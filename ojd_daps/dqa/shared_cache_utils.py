import diskcache
from diskcache.core import DBNAME
from pathlib import Path
from decimal import Decimal
import time
import boto3
import logging
from tqdm import tqdm
import json
import zlib
from datetime import datetime as dt
from tenacity import retry, stop_after_attempt, wait_fixed, wait_random

MARKER = "MARKER"
DATE_FORMAT = "%Y-%m-%d"
LATEST = "latest"
BACKUPS = "backups"
ALLOWED_CACHE_TYPES = (diskcache.Cache, diskcache.FanoutCache)


class NotADiskcacheBucket(Exception):
    """
    To indicate when a nominated S3 diskcache bucket
    doesn't conform to the expected format
    """

    pass


class NotADiskcacheDir(Exception):
    """
    To indicate when a nominated local diskcache directory
    doesn't conform to the expected format
    """

    pass


class BadBaseCache(Exception):
    """
    To indicate when a class does not inherit from exactly
    one of ALLOWED_CACHE_TYPES
    """

    pass


def timestamp():
    return str(time.time_ns())


def get_all_paths_local(directory):
    """
    Yield all absolute file paths in the local directory `directory`, and
    also yield the relative file path, which can be interpretted as an S3 key.

    Args:
        directory (str): Absolute path to the target directory.

    Yields:
        (filepath, s3_key) (str, str): Absolute path to a file in the directory,
                                       and an S3 key (relative path of the file)
    """
    prefix = len(directory)
    for filepath in filter(Path.is_file, Path(directory).glob("**/*")):
        filepath = str(filepath.absolute())
        s3_key = filepath[prefix:]
        if s3_key.startswith("/"):
            s3_key = s3_key[1:]
        yield filepath, s3_key


def get_all_paths_s3(bucket, directory):
    """
    Yield all target file paths from S3 keys, absolute with respect to a
    local `directory`, and also the S3 keys.

    Args:
        bucket (str): S3 bucket name
        directory (str): Absolute path to the target directory.

    Yields:
        (filepath, s3_key) (str, str): Absolute path to a file in the directory,
                                       and an S3 key (relative path of the file)
    """
    s3 = boto3.client("s3")
    for obj in s3.list_objects(Bucket=bucket, Prefix=f"{LATEST}/").get("Contents", []):
        s3_key = obj["Key"][len(LATEST) + 1 :]  # Strip "latest/"
        filepath = Path(directory) / s3_key
        yield str(filepath), s3_key


@retry(stop=stop_after_attempt(10), wait=wait_fixed(3) + wait_random(0, 2))
def sync_file(filepath, bucket, key, operation_name):
    """
    Either upload or download a file to/from S3 using either
    s3.upload_file or s3_download_file.
    """
    s3 = boto3.client("s3")
    operation = getattr(s3, f"{operation_name}_file")
    operation(Bucket=bucket, Key=f"{LATEST}/{key}", Filename=filepath)


def upload_to_s3(directory, bucket):
    """Upload the contents of local `directory` to remote `bucket`."""
    logging.info("Synchronising diskcache to S3...")
    paths = list(get_all_paths_local(directory))
    for filepath, key in tqdm(paths):
        sync_file(filepath=filepath, bucket=bucket, key=key, operation_name="upload")
    # Also upload a MARKER file - to officially timestamp this upload
    boto3.client("s3").put_object(
        Body=timestamp(), Bucket=bucket, Key=f"{LATEST}/{MARKER}"
    )


def download_from_s3(directory, bucket):
    """Dowload the contents of remote `bucket` to local `directory`."""
    paths = list(get_all_paths_s3(bucket=bucket, directory=directory))
    logging.info("Synchronising diskcache from S3...")
    for filepath, key in tqdm(paths):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        sync_file(bucket=bucket, key=key, filepath=filepath, operation_name="download")
    # Also download the MARKER file - so that timestamps are sync'd
    filepath = str(Path(directory) / MARKER)
    sync_file(
        bucket=bucket,
        key=MARKER,  # Note that sync_file will prepend '{LATEST}/'
        filepath=filepath,
        operation_name="download",
    )


def is_diskcache_file(filepath):
    """
    Does this file look like an expected diskcache file? Note this method
    is used to make sure we don't interact with anything other than diskcache
    buckets and directories, as we will be performing destructive operations.
    """
    if filepath.name in (DBNAME, f"{DBNAME}-shm", f"{DBNAME}-wal", MARKER):
        return True
    if filepath.suffix in (".val",):
        return True
    logging.warning(f"{filepath} is not recognised as a valid diskcache file.")
    return False


def _is_diskcache_directory(filepaths):
    filenames = list(map(lambda path: path.name, filepaths))
    if len(filepaths) <= 1:  # Multiple files, not just the marker file
        logging.warning(
            f"Too few files found in cache directory: {len(filenames)} found."
        )
        return False
    elif not all(map(is_diskcache_file, filepaths)):
        return False
    elif filenames.count(MARKER) != 1:  # There only be one marker file
        logging.warning(
            f"Exactly one MARKER file is expected but {filenames.count(MARKER)} found."
        )
        return False
    return True


def is_diskcache_bucket(bucket):
    """Do all keys in this bucket look like diskache files?"""
    s3 = boto3.resource("s3")
    # Create pseudo-filepaths from the bucket contents
    filepaths = sorted(map(lambda obj: Path(obj.key), s3.Bucket(bucket).objects.all()))
    # The latest diskcache contents
    filepaths = list(filter(lambda fpath: len(fpath.parts) > 1, filepaths))
    latest = list(filter(lambda fpath: fpath.parts[0] == LATEST, filepaths))
    backups = list(filter(lambda fpath: fpath.parts[0] == BACKUPS, filepaths))
    if not latest:
        logging.warning(f"No subdirectory {LATEST}/ found in bucket '{bucket}'")
        return False
    elif sorted(latest + backups) != filepaths:
        logging.warning(f"Only {LATEST}/ and {BACKUPS}/ expected in bucket '{bucket}'")
        return False
    return _is_diskcache_directory(latest)


def is_diskcache_directory(directory):
    """Do all files in this directory look like diskache files?"""
    directory = Path(directory).resolve()
    if not directory.exists():
        return False
    filepaths = list(filter(Path.is_file, directory.glob("**/*")))
    return _is_diskcache_directory(filepaths)


def prepare_bucket(bucket):
    """
    Copy bucket/LATEST contents to a timestamped backup "folder"
    in the same bucket, and then clear the LATEST "folder" from the bucket.
    This is the boto3 equivalent of renaming keys.
    """
    # Don't change bucket contents if the bucket doesn't look like a diskcache bucket!
    if not is_diskcache_bucket(bucket):
        raise NotADiskcacheBucket(f"Bucket {bucket} is not a valid diskcache bucket.")
    # Rename every key in the directory
    s3 = boto3.resource("s3")
    obj = s3.Object(bucket_name=bucket, key=f"{LATEST}/{MARKER}")
    _timestamp = obj.get()["Body"].read().decode("utf-8")
    for obj in s3.Bucket(bucket).objects.filter(Prefix=f"{LATEST}/"):
        new_key = f"{BACKUPS}/{_timestamp}/{obj.key[len(LATEST)+1:]}"
        s3.Object(bucket, new_key).copy_from(CopySource=f"{bucket}/{obj.key}")
        obj.delete()


def prepare_local(directory):
    """
    Rename the local diskcache directory to {directory.name}-backup / {timestamp},
    and then recreate an empty diskcache directory in it's place.
    """
    # Don't change directory contents if the path isn't a diskcache path!
    directory = Path(directory)
    if not is_diskcache_directory(directory):
        raise NotADiskcacheDir(
            f"Directory {directory} is not a valid diskcache directory"
        )
    # Read the previous timestamp from MARKER
    with open(directory / MARKER) as f:
        _timestamp = f.read()
    if _timestamp == "":
        _timestamp = timestamp()
    # Move the directory contents to a new location
    backup_dir = directory.parent / f"{directory.name}-backup"
    backup_dir.mkdir(exist_ok=True)
    directory.rename(backup_dir / _timestamp)
    # Recreate the original directory, ready for downloading contents from S3
    directory.mkdir()


def is_up_to_date(directory, bucket):
    """
    Verifies whether the local diskcache directory is already in sync with
    the remote diskcache bucket.
    """
    # First check if a local marker exists, if it doesn't then it means that
    # we're definitely not up to date
    local_marker_path = Path(directory) / MARKER
    if not local_marker_path.exists():
        logging.debug(f"File {local_marker_path} doesn't exist")
        return False
    # Compare the local and remote markers
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket, Key=f"{LATEST}/{MARKER}")
    s3_marker = obj["Body"].read()
    with open(local_marker_path) as f:
        local_marker = f.read()
    return local_marker == s3_marker.decode("utf-8")


def resolve_base_diskcache(obj):
    """Resolve the Cache type of this object from it's base classes"""

    classes = list(filter(lambda cls: cls in obj.__class__.mro(), ALLOWED_CACHE_TYPES))
    if len(classes) > 1:
        raise BadBaseCache(
            f"Only one base Cache class is allowed from {ALLOWED_CACHE_TYPES}, "
            f"but {obj.__class__} has {len(classes)} ({classes})."
        )
    elif len(classes) == 0:
        raise BadBaseCache(
            f"None of the allowed Cache bases {ALLOWED_CACHE_TYPES} found in "
            f"the bases of {obj.__class__}"
        )
    return classes[0]


def compress(value):
    """Convert to JSON and then compress with zlib"""
    json_bytes = json.dumps(value, cls=CacheUtilsEncoder).encode("utf-8")
    return zlib.compress(json_bytes, level=6)


def decompress(data):
    """Decompress with zlib and then load data as JSON"""
    return json.loads(zlib.decompress(data).decode("utf-8"), object_hook=date_hook)


def date_hook(json_dict):
    """Try to coerce any object that matches the DATE_FORMAT to a datetime object"""
    for (key, value) in json_dict.items():
        try:
            json_dict[key] = dt.strptime(value, DATE_FORMAT)
        except (ValueError, TypeError):
            pass
    return json_dict


class CacheUtilsEncoder(json.JSONEncoder):
    """Custom JSONEncoder recipe for non-native python types"""

    def default(self, obj):
        if isinstance(obj, dt):
            return obj.strftime(DATE_FORMAT)
        elif isinstance(obj, Decimal):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


class JSONDisk(diskcache.Disk):
    """Custom Disk recipe which encodes to JSON using our own Encoder and hook"""

    def __init__(self, directory, **kwargs):
        super().__init__(directory, **kwargs)

    def put(self, key):
        data = compress(key)
        return super().put(data)

    def get(self, key, raw):
        data = super().get(key, raw)
        return decompress(data)

    def store(self, value, read, key=diskcache.core.UNKNOWN):
        if not read:
            value = compress(value)
        return super().store(value, read, key=key)

    def fetch(self, mode, filename, value, read):
        data = super().fetch(mode, filename, value, read)
        if not read:
            data = decompress(data)
        return data
