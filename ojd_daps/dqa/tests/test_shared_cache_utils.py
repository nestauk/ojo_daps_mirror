import pytest
from unittest import mock
from moto import mock_s3
from ojd_daps.dqa.shared_cache_utils import (
    get_all_paths_local,
    get_all_paths_s3,
    upload_to_s3,
    download_from_s3,
    is_diskcache_file,
    is_diskcache_bucket,
    is_diskcache_directory,
    prepare_bucket,
    prepare_local,
    is_up_to_date,
    resolve_base_diskcache,
    compress,
    decompress,
    CacheUtilsEncoder,
    date_hook,
)
from ojd_daps.dqa.shared_cache_utils import (
    LATEST,
    MARKER,
    boto3,
    Path,
    NotADiskcacheBucket,
    NotADiskcacheDir,
    diskcache,
    BadBaseCache,
    dt,
    DATE_FORMAT,
    json,
    Decimal,
)

BUCKET_NAME = "temp-testing-bucket-123"  # NB: not a real bucket


@pytest.fixture
def mocked_s3():
    with mock_s3():
        s3 = boto3.resource("s3", region_name="us-east-1")  # NB: region is arbitrary
        s3.create_bucket(Bucket=BUCKET_NAME)
        for i in range(3):
            for j in range(2):
                obj = s3.Object(BUCKET_NAME, f"{LATEST}/directory_{i}/file_{j}.txt")
                obj.put(Body="")
        yield s3


@pytest.fixture
def temp_path(tmp_path):
    for i in range(3):
        d = tmp_path / f"directory_{i}"
        d.mkdir()
        for j in range(2):
            p = d / f"file_{j}.txt"
            p.write_text("")
    return str(tmp_path.resolve())


@pytest.fixture
def a_date():
    return dt(year=2013, month=2, day=13)


@pytest.fixture
def a_date_str(a_date):
    return a_date.strftime(DATE_FORMAT)


def test_get_all_paths_local(temp_path):
    assert sorted(get_all_paths_local(temp_path)) == [
        (f"{temp_path}/directory_0/file_0.txt", "directory_0/file_0.txt"),
        (f"{temp_path}/directory_0/file_1.txt", "directory_0/file_1.txt"),
        (f"{temp_path}/directory_1/file_0.txt", "directory_1/file_0.txt"),
        (f"{temp_path}/directory_1/file_1.txt", "directory_1/file_1.txt"),
        (f"{temp_path}/directory_2/file_0.txt", "directory_2/file_0.txt"),
        (f"{temp_path}/directory_2/file_1.txt", "directory_2/file_1.txt"),
    ]


def test_get_all_paths_s3(mocked_s3):
    assert sorted(get_all_paths_s3(BUCKET_NAME, "temp_path")) == [
        ("temp_path/directory_0/file_0.txt", "directory_0/file_0.txt"),
        ("temp_path/directory_0/file_1.txt", "directory_0/file_1.txt"),
        ("temp_path/directory_1/file_0.txt", "directory_1/file_0.txt"),
        ("temp_path/directory_1/file_1.txt", "directory_1/file_1.txt"),
        ("temp_path/directory_2/file_0.txt", "directory_2/file_0.txt"),
        ("temp_path/directory_2/file_1.txt", "directory_2/file_1.txt"),
    ]


@mock_s3
def test_upload_to_s3(temp_path):
    s3 = boto3.resource("s3", region_name="us-east-1")  # NB: region is arbitrary
    s3.create_bucket(Bucket=BUCKET_NAME)

    # Sanity check: bucket is empty before we start
    assert sorted(get_all_paths_s3(bucket=BUCKET_NAME, directory=temp_path)) == []

    upload_to_s3(directory=temp_path, bucket=BUCKET_NAME)
    assert sorted(get_all_paths_s3(bucket=BUCKET_NAME, directory=temp_path)) == [
        (f"{temp_path}/{MARKER}", MARKER),
        (f"{temp_path}/directory_0/file_0.txt", "directory_0/file_0.txt"),
        (f"{temp_path}/directory_0/file_1.txt", "directory_0/file_1.txt"),
        (f"{temp_path}/directory_1/file_0.txt", "directory_1/file_0.txt"),
        (f"{temp_path}/directory_1/file_1.txt", "directory_1/file_1.txt"),
        (f"{temp_path}/directory_2/file_0.txt", "directory_2/file_0.txt"),
        (f"{temp_path}/directory_2/file_1.txt", "directory_2/file_1.txt"),
    ]


def test_download_from_s3(mocked_s3, tmp_path):  # NB: pytest.tmp_path not temp_path
    tmp_path = str(tmp_path)

    # Add in a MARKER file, which is expected by 'download_from_s3'
    obj = mocked_s3.Object(bucket_name=BUCKET_NAME, key=f"{LATEST}/{MARKER}")
    obj.put(Body="")

    # Sanity check: tmp path is empty before we start
    assert sorted(get_all_paths_local(tmp_path)) == []

    download_from_s3(directory=tmp_path, bucket=BUCKET_NAME)
    assert sorted(get_all_paths_local(tmp_path)) == [
        (f"{tmp_path}/{MARKER}", MARKER),
        (f"{tmp_path}/directory_0/file_0.txt", "directory_0/file_0.txt"),
        (f"{tmp_path}/directory_0/file_1.txt", "directory_0/file_1.txt"),
        (f"{tmp_path}/directory_1/file_0.txt", "directory_1/file_0.txt"),
        (f"{tmp_path}/directory_1/file_1.txt", "directory_1/file_1.txt"),
        (f"{tmp_path}/directory_2/file_0.txt", "directory_2/file_0.txt"),
        (f"{tmp_path}/directory_2/file_1.txt", "directory_2/file_1.txt"),
    ]


@pytest.mark.parametrize("filepath", ("cache.db", MARKER, "foo.val", "bar.val"))
def test_is_diskcache_file(filepath):
    for dirname in ["", "aaa", "bbb"]:
        _filepath = Path(dirname) / filepath
        assert is_diskcache_file(_filepath)


def test_is_diskcache_directory(tmp_path):

    # 0) Check that the directory isn't valid until it contains files
    assert not is_diskcache_directory(tmp_path)  # works with Path or str
    assert not is_diskcache_directory(str(tmp_path))  # works with Path or str

    # 1) Add valid files and check passes
    # 1a) make dirs
    (tmp_path / "foo").mkdir(parents=True)
    (tmp_path / "bar").mkdir(parents=True)
    (tmp_path / "foo" / "bar").mkdir(parents=True)
    # 1b) make files
    (tmp_path / "foo" / "cache.db").touch()
    (tmp_path / "bar" / "cache.db").touch()
    (tmp_path / "cache.db").touch()
    (tmp_path / "foo" / "bar.val").touch()
    (tmp_path / "foo" / "bar" / "baz.val").touch()
    (tmp_path / "bar.val").touch()
    (tmp_path / MARKER).touch()
    # 1c) check that these are valid
    assert is_diskcache_directory(tmp_path)  # works with Path or str
    assert is_diskcache_directory(str(tmp_path))

    # 2) Add an invalid file and check fails
    (tmp_path / "something.else").touch()
    assert not is_diskcache_directory(tmp_path)  # works with Path or str
    assert not is_diskcache_directory(str(tmp_path))


@mock_s3
def test_is_diskcache_bucket():
    s3 = boto3.resource("s3", region_name="us-east-1")  # NB: region is arbitrary
    s3.create_bucket(Bucket=BUCKET_NAME)

    # 0) Check that empty isn't valid
    assert not is_diskcache_bucket(BUCKET_NAME)

    # 1) Add valid files and check passes
    # 1a) make objects
    for k in (
        "foo/cache.db",
        "bar/cache.db",
        "cache.db",
        "foo/bar.val",
        "foo/bar/baz.val",
        "bar.val",
        MARKER,
    ):
        s3.Object(BUCKET_NAME, f"latest/{k}").put(Body="")
        s3.Object(BUCKET_NAME, f"backups/123/{k}").put(Body="")
    # 1b) check that these are valid
    assert is_diskcache_bucket(BUCKET_NAME)

    # 2) Add an invalid key and check fails
    s3.Object(BUCKET_NAME, "foo/bar/something.else").put(Body="")
    assert not is_diskcache_bucket(BUCKET_NAME)


@mock_s3
def test_prepare_bucket_invalid():
    s3 = boto3.resource("s3", region_name="us-east-1")  # NB: region is arbitrary
    s3.create_bucket(Bucket=BUCKET_NAME)

    # 0) Check that can't prepare an empty bucket
    with pytest.raises(NotADiskcacheBucket):
        prepare_bucket(BUCKET_NAME)

    # 1) Check that can't prepare a non-diskcache bucket
    s3.Object(BUCKET_NAME, "foo/cache.db").put(Body="")
    s3.Object(BUCKET_NAME, "foo/bar/something.else").put(Body="")
    with pytest.raises(NotADiskcacheBucket):
        prepare_bucket(BUCKET_NAME)

    # Verify that not modified
    assert len(list(s3.Bucket(BUCKET_NAME).objects.all())) == 2


@mock_s3
def test_prepare_bucket():
    s3 = boto3.resource("s3", region_name="us-east-1")  # NB: region is arbitrary
    s3.create_bucket(Bucket=BUCKET_NAME)

    # Sanity check: no keys in the bucket
    assert len(list(s3.Bucket(BUCKET_NAME).objects.all())) == 0

    # Fill up the bucket with valid keys
    for k in (
        "foo/cache.db",
        "bar/cache.db",
        "cache.db",
        "foo/bar.val",
        "foo/bar/baz.val",
        "bar.val",
        MARKER,
    ):
        s3.Object(BUCKET_NAME, f"latest/{k}").put(Body="")
        s3.Object(BUCKET_NAME, f"backups/abc/{k}").put(Body="")

    # Seven keys in the bucket, zero after
    assert len(list(s3.Bucket(BUCKET_NAME).objects.filter(Prefix="latest/"))) == 7
    assert len(list(s3.Bucket(BUCKET_NAME).objects.filter(Prefix="backups/"))) == 7
    # with mock.patch.object(boto3, "resource", return_value=s3):
    prepare_bucket(BUCKET_NAME)
    assert len(list(s3.Bucket(BUCKET_NAME).objects.filter(Prefix="latest/"))) == 0

    # But all 7 keys can be found in the backup folder
    assert len(list(s3.Bucket(BUCKET_NAME).objects.filter(Prefix="backups/"))) == 14


def test_prepare_local_invalid(tmp_path):
    # Sanity check: no files in the directory
    assert len(list(tmp_path.glob("**/*"))) == 0

    # 0) Check that can't prepare an empty directory
    with pytest.raises(NotADiskcacheDir):
        prepare_local(tmp_path)

    # 1) Check that can't prepare a non-diskcache directory
    # make dirs
    (tmp_path / "foo" / "bar").mkdir(parents=True)
    # make non-diskcache files
    (tmp_path / "foo" / "cachedb").touch()
    (tmp_path / "foo" / "bar" / "cache.DB").touch()
    with pytest.raises(NotADiskcacheDir):
        prepare_local(tmp_path)

    # Verify that not modified
    assert len(list(filter(Path.is_file, tmp_path.glob("**/*")))) == 2


def test_prepare_local(tmp_path):

    # Sanity check: no files in the directory
    assert len(list(tmp_path.glob("**/*"))) == 0

    # Add valid files and check passes
    # make dirs
    (tmp_path / "foo").mkdir(parents=True)
    (tmp_path / "bar").mkdir(parents=True)
    (tmp_path / "foo" / "bar").mkdir(parents=True)
    # make files
    (tmp_path / "foo" / "cache.db").touch()
    (tmp_path / "bar" / "cache.db").touch()
    (tmp_path / "cache.db").touch()
    (tmp_path / "foo" / "bar.val").touch()
    (tmp_path / "foo" / "bar" / "baz.val").touch()
    (tmp_path / "bar.val").touch()
    (tmp_path / MARKER).touch()

    # Seven files in the directory before, zero after
    assert len(list(filter(Path.is_file, tmp_path.glob("**/*")))) == 7
    prepare_local(tmp_path)
    assert len(list(tmp_path.glob("**/*"))) == 0

    # But check that all 7 files can be found in a new backup
    backup_path = tmp_path.parent / f"{tmp_path.name}-backup"
    assert len(list(filter(Path.is_file, backup_path.glob("**/*")))) == 7


def test_is_up_to_date_no_marker(tmp_path):
    # tmp_path doesn't contain MARKER and so return False
    (tmp_path / "bar.val").touch()
    assert not is_up_to_date(tmp_path, "bucket")


@mock_s3
def test_is_up_to_date_marker_out_of_sync(tmp_path):
    # Local marker = 123
    with open(tmp_path / MARKER, "w") as f:
        f.write("123")
    # Remote marker = abc
    s3 = boto3.resource("s3", region_name="us-east-1")  # NB: region is arbitrary
    s3.create_bucket(Bucket=BUCKET_NAME)
    s3.Object(BUCKET_NAME, f"{LATEST}/{MARKER}").put(Body="ABC")
    assert not is_up_to_date(tmp_path, BUCKET_NAME)


@mock_s3
def test_is_up_to_date_marker_in_sync(tmp_path):
    # Local marker = 123
    with open(tmp_path / MARKER, "w") as f:
        f.write("123")
    # Remote marker = 123
    s3 = boto3.resource("s3", region_name="us-east-1")  # NB: region is arbitrary
    s3.create_bucket(Bucket=BUCKET_NAME)
    s3.Object(BUCKET_NAME, f"{LATEST}/{MARKER}").put(Body="123")
    assert is_up_to_date(tmp_path, BUCKET_NAME)


def test_resolve_base_diskcache():
    class A:
        pass

    class B:
        pass

    class NewFanoutCache(A, diskcache.FanoutCache, B):
        pass

    class NewCache(A, diskcache.Cache, B):
        pass

    assert resolve_base_diskcache(NewFanoutCache()) is diskcache.FanoutCache
    assert resolve_base_diskcache(NewCache()) is diskcache.Cache


def test_resolve_base_diskcache_too_many_bases():
    class A:
        pass

    class B:
        pass

    class NewCache(A, diskcache.Cache, diskcache.FanoutCache, B):
        pass

    with pytest.raises(BadBaseCache):
        resolve_base_diskcache(NewCache())


def test_resolve_base_diskcache_too_few_bases():
    class A:
        pass

    class B:
        pass

    class NewCache(A, B):
        pass

    with pytest.raises(BadBaseCache):
        resolve_base_diskcache(NewCache())


def test_compress_decompress(a_date):
    value = [{"a date": a_date, "a value": "ABcD"}] * 1000

    # 1) Check that the compression is better than just dumping the string
    compressed = compress(value)
    json_str = json.dumps(value, cls=CacheUtilsEncoder)
    assert type(compressed) is bytes
    assert len(json_str.encode("utf-8")) > 10 * len(compressed)

    # 2) Check that decompression returns the opposite of compression
    decompressed = decompress(compressed)
    assert decompressed == value


def test_date_hook(a_date, a_date_str):
    json_dict = {"a date": a_date_str, "a value": "ABcD"}
    assert date_hook(json_dict) == {
        "a date": a_date,
        "a value": "ABcD",
    }


def test_CacheUtilsEncoder(a_date, a_date_str):
    json_dict = {"a date": a_date, "a value": ["ABcD"], "a decimal": Decimal("12.3")}

    encoded = CacheUtilsEncoder().encode(json_dict)
    assert (
        encoded
        == f'{{"a date": "{a_date_str}", "a value": ["ABcD"], "a decimal": 12.3}}'
    )
