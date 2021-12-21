"""
shared_cache
------------

An implementation of `diskcache` with three extra features:

1) It works on generators via the new "GeneratorCacheMixin"
   (and so chunks can be cached)
2) It downloads a previously prepared "template" diskcache from S3,
   with previous function calls already defined
3) The key used to hash the function call is changed from diskcache's
   default (which is basically the name of the function) to a hash of
   the function's source code. In effect this means that changes to
   the source code will trigger the function, rather than falling back
   on cached results, which is the behaviour developers would like
   without needing to clear the cache manually.
"""

import inspect
import diskcache
from diskcache.core import ENOVAL, DBNAME
import functools as ft
from hashlib import md5
from pathlib import Path
import logging
from collections import namedtuple
import sqlite3


from ojd_daps.dqa.shared_cache_utils import (
    resolve_base_diskcache,
    JSONDisk,
    prepare_bucket,
    prepare_local,
    upload_to_s3,
    download_from_s3,
    is_up_to_date,
    MARKER,
)

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def full_name(func):
    """
    Monkey patch diskcache's "full_name" function so that
    cache keys are a hash of the function source code
    (and therefore detects changes (useful for debugging))
    """
    raw_key = inspect.getsource(func)
    return md5(raw_key.encode("utf-8")).hexdigest()


# Monkey patch here
diskcache.core.full_name = full_name


class GeneratorCacheMixin:
    """Mixin for making a Cache which works with generators"""

    def memoize(self, name=None, typed=False, expire=None, tag=None, chunksize=None):
        """Overload the base memoize to allow for generators"""
        _decorator = super().memoize(name=name, typed=typed, expire=expire, tag=tag)

        def decorator(func):
            """Decorator created by memoize() for callable `func`."""
            # For regular (non-generator) functions, use the base memoize
            if not inspect.isgeneratorfunction(func):
                if chunksize is not None:
                    raise ValueError(
                        "The 'chunksize' argument is only valid for "
                        "memoizing generators, not regular functions"
                    )
                return _decorator(func)

            # Otherwise for generators use the following customisation
            base = full_name(func)

            @ft.wraps(func)
            def wrapper(*args, **kwargs):
                """Wrapper for callable to cache arguments and return values."""
                # Check whether a status has been registered for this function
                base_key = wrapper.__cache_key__(base, *args, **kwargs)
                status = self.get(base_key, default=ENOVAL, retry=True)

                # Null status: evaluate the generator until finished
                if status is ENOVAL:
                    logger.warning(
                        f" No cache found for {func.__name__} with "
                        f"args {args} and kwargs {kwargs}.\n"
                        "Evaluating this now, but it may take some time.\n"
                    )
                    i = 0
                    chunk = []
                    for result in func(*args, **kwargs):
                        chunk.append(result)
                        # Flush the chunk when required
                        if chunksize is None or len(chunk) == chunksize:
                            key = wrapper.__cache_key__(base + str(i), *args, **kwargs)
                            self.set(key, chunk, expire, tag=tag, retry=True)
                            i += 1
                            chunk = []
                        yield result
                    # One final flush
                    if chunk:
                        key = wrapper.__cache_key__(base + str(i), *args, **kwargs)
                        self.set(key, chunk, expire, tag=tag, retry=True)
                    # When finished mark status as DONE
                    if expire is None or expire > 0:
                        self.set(base_key, "DONE", expire, tag=tag, retry=True)
                # Non-null status: yield results from diskcache
                else:
                    i = 0
                    key = wrapper.__cache_key__(base + str(i), *args, **kwargs)
                    chunk = self.get(key, default=ENOVAL, retry=True)
                    while chunk is not ENOVAL:
                        yield from chunk
                        i += 1
                        key = wrapper.__cache_key__(base + str(i), *args, **kwargs)
                        chunk = self.get(key, default=ENOVAL, retry=True)

            def __cache_key__(base, *args, **kwargs):
                """Make key for cache given function arguments."""
                return diskcache.core.args_to_key((base,), args, kwargs, typed)

            wrapper.__cache_key__ = __cache_key__
            return wrapper

        return decorator


class SharedCacheMixin(GeneratorCacheMixin):
    """Mixin for making a Cache which which works with generators and syncs with S3"""

    def __init__(self, bucket, read_only=True, *args, **kwargs):
        self.bucket = bucket
        self.read_only = read_only
        # Instantiate the diskcache base class and then create the MARKER file
        cache_base_class = resolve_base_diskcache(self)
        cache_base_class.__init__(self, disk=JSONDisk, **kwargs)
        (Path(self.directory) / MARKER).touch()

        # For some reason diskcache/SQLite clears the S3 Cache table
        # if we don't interact with it first. Until someone has a better
        # idea of why that is happening, we need the following two lines:
        conn = sqlite3.connect(
            Path(self.directory) / DBNAME,
            timeout=self._timeout,
            isolation_level=None,
        )
        conn.cursor().execute("SELECT COUNT(key) FROM Cache")

        # Synchronise from S3 to local if not up to date
        if not is_up_to_date(self.directory, self.bucket):
            logger.info("Synchronising diskcache from S3 (this may take 5 minutes)")
            prepare_local(self.directory)
            download_from_s3(self.directory, self.bucket)
            # Re-initialise to pick up the new cache
            cache_base_class.__init__(self, disk=JSONDisk, **kwargs)

    def upload_to_s3(self):
        if self.read_only is True:  # i.e. type no-coersion
            raise IOError("Cannot upload in read_only mode.")
        logger.info("Synchronising diskcache to S3:")
        prepare_bucket(self.bucket)
        upload_to_s3(self.directory, self.bucket)


class SharedCache(SharedCacheMixin, diskcache.Cache):
    """
    Concrete implementation of Cache which works with generators and syncs with S3.
    """

    pass


class SharedFanoutCache(SharedCacheMixin, diskcache.FanoutCache):
    """
    Concrete implementation of FanoutCache which works with generators and syncs with S3.
    """

    pass


def FakeCache():
    """
    A mechanism for mocking up a cache. It only implements a mocked out version of
    the cache.memoize decorator, and does absolutely nothing. For example:

        cache = FakeCache()

        @cache.memoize(foo, bar)
        def func(something):
            return something


    and

        def func(something):
            return something

    have identical behaviour.
    """
    cls = namedtuple(
        "FakeCache",
        "memoize",
        defaults=[  # mocks out cache.memoize()
            lambda *_args, **_kwargs: (  # cache.memoize
                lambda func: (  # memoize.decorator
                    lambda *args, **kwargs: func(*args, **kwargs)  # decorator.wrapper
                )
            )
        ],
    )
    return cls()
