"""
Common functions
"""
import boto3
import itertools
import time
from random import randint
from pathlib import Path
from functools import lru_cache

BUCKET_NAME = "open-jobs-lake"


def flatten(iterable):
    """Flatten a list of iterables to a single flat list"""
    return list(itertools.chain(*iterable))


def get_chunks(_list, chunksize):
    """Chunk up a flat list into a list of lists, each with size `chunksize`"""
    chunks = [_list[x : x + chunksize] for x in range(0, len(_list), chunksize)]
    return chunks


def jiterange(*args, min_sleep=2, max_sleep=4, **kwargs):
    """Iterate through a range, and jitter sleep between iterations"""
    for item in range(*args, **kwargs):
        yield item
        time.sleep(randint(min_sleep, max_sleep))


def save_to_s3(s3_path, filename, contents):
    """Saves the contents to the filename in {BUCKET_NAME}/{S3_PATH}"""
    s3 = boto3.resource("s3")
    obj = s3.Object(BUCKET_NAME, str(Path(s3_path) / filename))
    obj.put(Body=contents)


@lru_cache()
def load_from_s3(s3_path, filename):
    """Loads the file contents from the filename at {BUCKET_NAME}/{S3_PATH}"""
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=str(Path(s3_path) / filename))
    return obj["Body"].read().decode()
