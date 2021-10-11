"""
test_shared_diskcache
---------------------
"""
import pytest
from unittest import mock
from moto import mock_s3
from ojd_daps.dqa.shared_cache import (
    full_name,
    SharedCache,
    FakeCache
)
import time


DATE_FORMAT = "%d-%m-%Y"
SLEEP_TIME = 1.5

@pytest.fixture
def cache(tmp_path):
    """
    Create a temporary local diskcache, with a knockdown before and after.
    mock_s3 is used to disable boto3, useful for not destroying our S3 buckets.
    Mocking is_up_to_date avoids testing features which are tested in
    test_shared_diskcache_utils.py
    """
    with mock_s3(), mock.patch(
        "ojd_daps.dqa.shared_cache.is_up_to_date", return_value=True
    ):
        cache = SharedCache(bucket="dummy-bucket", directory=tmp_path)
        cache.clear()  # pre-knockdown
        yield cache
        cache.clear()  # post-knockdown


# Pseudo-fixtures
def generator_func(x, y):
    """A demo function to be redis-cached"""
    for i in range(x):
        for j in range(y):
            time.sleep(SLEEP_TIME)
            yield {"x": i, "y": j}


def regular_func(x, y):
    """A demo function to be redis-cached"""
    time.sleep(SLEEP_TIME)
    return {"x": x, "y": y}


def test_full_name():
    """Create a hash from a function and check that the properties are as expected"""

    def foo():
        return "bar"

    # Check reproducibility
    foobar_hash = full_name(foo)
    assert full_name(foo) == foobar_hash

    def foo():
        return "baz"

    # Check reproducibility
    foobaz_hash = full_name(foo)
    assert full_name(foo) == foobaz_hash

    # Check unique
    assert foobar_hash != foobaz_hash

    # Check properties
    for hash_ in (foobar_hash, foobaz_hash):
        assert type(hash_) is str
        assert len(hash_) == 32


def test_regular_caching(cache):
    @cache.memoize()
    def func(x, y):
        return regular_func(x, y)

    # Internal test: verify that the cache is empty
    keys = list(cache.iterkeys())
    assert len(keys) == 0

    # Check that sleeping has occurred
    before = time.time()
    assert func(x=1, y=2) == {"x": 1, "y": 2}
    assert func(x=2, y=3) == {"x": 2, "y": 3}
    assert time.time() - before > 0.99 * 2 * SLEEP_TIME  # 2 rows x SLEEP_TIME secs

    # Now check that the caching is working
    before = time.time()
    assert func(x=1, y=2) == {"x": 1, "y": 2}
    assert func(x=2, y=3) == {"x": 2, "y": 3}
    assert time.time() - before < 1  # Should be very quick

    # Finally verify by checking the cache
    keys = list(cache.iterkeys())
    assert len(keys) == 2  # One per row


def test_generator_caching(cache):
    @cache.memoize()
    def func(x, y):
        yield from generator_func(x, y)

    # Internal test: verify that the cache is empty
    keys = list(cache.iterkeys())
    assert len(keys) == 0

    # Check that sleeping has occurred
    before = time.time()
    assert list(func(x=1, y=2)) == [{"x": 0, "y": 0}, {"x": 0, "y": 1}]
    assert time.time() - before > 0.99 * 2 * SLEEP_TIME  # 2 rows x SLEEP_TIME secs

    # Now check that the caching is working
    before = time.time()
    assert list(func(x=1, y=2)) == [{"x": 0, "y": 0}, {"x": 0, "y": 1}]
    assert time.time() - before < 1  # Should be very quick

    # Finally verify by checking the cache
    keys = list(cache.iterkeys())
    assert len(keys) == 3  # One per row, plus one for the function


def test_generator_caching_chunks_inexact(cache):
    @cache.memoize(chunksize=2)
    def func(x, y):
        yield from generator_func(x, y)

    # Internal test: verify that the cache is empty
    keys = list(cache.iterkeys())
    assert len(keys) == 0

    # Check that sleeping has occurred
    before = time.time()
    assert list(func(x=1, y=3)) == [
        {"x": 0, "y": 0},
        {"x": 0, "y": 1},  # end of chunk one
        {"x": 0, "y": 2},  # premature of chunk two
    ]
    assert time.time() - before > 0.99 * 3 * SLEEP_TIME  # 3 rows x SLEEP_TIME secs

    # Now check that the caching is working
    before = time.time()
    assert list(func(x=1, y=3)) == [
        {"x": 0, "y": 0},
        {"x": 0, "y": 1},  # end of chunk one
        {"x": 0, "y": 2},  # premature of chunk two
    ]
    assert time.time() - before < 1  # Should be very quick

    # Finally verify by checking the cache
    keys = list(cache.iterkeys())
    assert len(keys) == 3  # One per chunk, plus one for the function


def test_generator_caching_chunks_exact(cache):
    @cache.memoize(chunksize=2)
    def func(x, y):
        yield from generator_func(x, y)

    # Internal test: verify that the cache is empty
    keys = list(cache.iterkeys())
    assert len(keys) == 0

    # Check that sleeping has occurred
    before = time.time()
    assert list(func(x=2, y=2)) == [
        {"x": 0, "y": 0},
        {"x": 0, "y": 1},  # end of chunk one
        {"x": 1, "y": 0},
        {"x": 1, "y": 1},  # end of chunk two
    ]
    assert time.time() - before > 0.99 * 3 * SLEEP_TIME  # 3 rows x SLEEP_TIME secs

    # Now check that the caching is working
    before = time.time()
    assert list(func(x=2, y=2)) == [
        {"x": 0, "y": 0},
        {"x": 0, "y": 1},  # end of chunk one
        {"x": 1, "y": 0},
        {"x": 1, "y": 1},  # end of chunk two
    ]
    assert time.time() - before < 1  # Should be very quick

    # Finally verify by checking the cache
    keys = list(cache.iterkeys())
    assert len(keys) == 3  # One per chunk, plus one for the function


def test_fake_cache_regular():
    cache = FakeCache()

    @cache.memoize()
    def func(x, y):
        return regular_func(x, y)

    before = time.time()
    assert func(x=1, y=2) == {"x": 1, "y": 2}
    assert func(x=2, y=3) == {"x": 2, "y": 3}
    assert time.time() - before > 0.99 * 2 * SLEEP_TIME  # 2 rows x SLEEP_TIME secs

    # Check the fake cache isn't doing anything
    before = time.time()
    assert func(x=1, y=2) == {"x": 1, "y": 2}
    assert func(x=2, y=3) == {"x": 2, "y": 3}
    assert time.time() - before > 0.99 * 2 * SLEEP_TIME  # 2 rows x SLEEP_TIME secs


def test_fake_cache_generator():
    cache = FakeCache()

    @cache.memoize()
    def func(x, y):
        yield from generator_func(x, y)

    before = time.time()
    assert list(func(x=1, y=2)) == [{"x": 0, "y": 0}, {"x": 0, "y": 1}]
    assert time.time() - before > 0.99 * 2 * SLEEP_TIME  # 2 rows x SLEEP_TIME secs

    # Check the fake cache isn't doing anything
    before = time.time()
    assert list(func(x=1, y=2)) == [{"x": 0, "y": 0}, {"x": 0, "y": 1}]
    assert time.time() - before > 0.99 * 2 * SLEEP_TIME  # 2 rows x SLEEP_TIME secs
