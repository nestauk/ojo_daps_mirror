"""
test_shared_diskcache_integration
---------------------------------
"""
from moto import mock_s3
from ojd_daps.dqa.shared_cache import (
    SharedCache,
)
from ojd_daps.dqa.shared_cache_utils import (
    LATEST,
    MARKER,
    get_all_paths_s3,
    get_all_paths_local,
    prepare_local,
)
import boto3
import time

BUCKET_NAME = "temp-testing-bucket-123"  # NB: not a real bucket
SLEEP_TIME = 3


def get_counts(_cache):
    """
    Helper function to retrieve object counts for validation

    Returns:
        n_s3_objs, n_local_objs, n_cache_keys
    """
    s3_objs = list(get_all_paths_s3(BUCKET_NAME, _cache.directory))
    local_objs = list(get_all_paths_local(_cache.directory))
    keys = list(_cache.iterkeys())
    return len(s3_objs), len(local_objs), len(keys)


def _foo():
    """
    Function to be wrapped by cache.memoize.

    Note that it generates a timestamp, which can therefore
    be used to check that the cache is working.

    The factor of 1000000 is to ensure that we have something sizeable to cache.
    """
    time.sleep(SLEEP_TIME)
    return str(time.time()) * 1000000


@mock_s3
def test_reproducible(cache, tmp_path):

    # Create a mocked up bucket
    s3 = boto3.resource("s3", region_name="us-east-1")  # NB: region is arbitrary
    s3.create_bucket(Bucket=BUCKET_NAME)
    # Mock up to look like a diskcache bucket
    s3.Object(BUCKET_NAME, f"{LATEST}/cache.db").put(Body="")
    s3.Object(BUCKET_NAME, f"{LATEST}/{MARKER}").put(Body="")

    # Create a shared cache object
    cache = SharedCache(bucket=BUCKET_NAME, directory=tmp_path, read_only=False)

    @cache.memoize()
    def foo():
        return _foo()

    ##################
    # 0 a) Check S3 cache is empty, local cache is empty
    n_s3, n_local, n_keys = get_counts(cache)
    assert n_s3 == 2  # 2 means empty, just a marker file and empty cache.db
    assert n_keys == 0
    assert n_local == 4

    ##################
    # 1 a) Check the function executes as expected
    before = time.time()
    value = foo()
    assert type(value) is str
    assert len(value)
    assert time.time() - before > 0.99 * SLEEP_TIME

    # 1 b) Check S3 is empty, local storage is not empty
    n_s3, n_local, n_keys = get_counts(cache)
    assert n_s3 == 2
    assert n_keys == 1
    assert n_local >= 4

    ##################
    # 2 a) Check that the cache has worked
    before = time.time()
    assert foo() == value  # value unchanged
    assert time.time() - before < SLEEP_TIME / 10  # i.e. no sleep

    # 2 b) Check S3 is *still* empty, local storage is unchanged
    n_s3, n_local, n_keys = get_counts(cache)
    assert n_s3 == 2
    assert n_keys == 1
    assert n_local >= 4

    ##################
    # 3 a) Upload the cache to S3, then clear
    cache.upload_to_s3()
    cache.clear()
    prepare_local(tmp_path)  # clears the local directory

    # 3 b) Check S3 is *not* empty, local storage *is now* empty
    n_s3, n_local, n_keys = get_counts(cache)
    assert n_s3 >= 4  # MARKER, cache.db, cache.db-shm, cache.db-wal, .val
    assert n_keys == 0
    assert n_local == 0

    # 3 c) Check that the function draws a new value because the cache is empty
    before = time.time()
    assert foo() != value
    assert time.time() - before > 0.99 * SLEEP_TIME
    del cache

    ##################
    # 4 a) Download the cache from S3 to local
    cache = SharedCache(bucket=BUCKET_NAME, directory=tmp_path)  # / "somewhere_else")

    @cache.memoize()
    def foo():
        return _foo()

    # 4 b) Check both S3 and local aren't empty
    n_s3, n_local, n_keys = get_counts(cache)
    assert n_s3 >= 4
    assert n_local >= 2  # MARKER, cache.db (wal and shm can get swallowed up on init)
    assert n_keys == 1

    ##################
    # 5 a) Check the cache is working again
    before = time.time()
    assert foo() == value  # Got the original value back!
    assert time.time() - before < SLEEP_TIME / 10  # i.e. no sleep
