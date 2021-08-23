from unittest import mock

from ojd_daps.dqa.data_getters import (
    get_s3_job_ads,
    get_db_job_ads,
    get_locations,
    get_location_lookup,
    get_salaries,
    get_skills,
    get_requires_degree,
)

from ojd_daps.dqa.data_getters import Decimal


PATH = "ojd_daps.dqa.data_getters.{}"


@mock.patch(PATH.format("get_metaflow_bucket"))
@mock.patch(PATH.format("boto3"))
def test_get_s3_job_ads_default(mocked_boto, mocked_get_bkt):
    mocked_boto.resource().Bucket().objects.filter.return_value = [
        mock.MagicMock()
    ] * 10
    items = list(get_s3_job_ads("reed"))
    assert len(items) == 10
    assert all(item["body"] is not None for item in items)


@mock.patch(PATH.format("get_metaflow_bucket"))
@mock.patch(PATH.format("boto3"))
def test_get_s3_job_ads_sample(mocked_boto, mocked_get_bkt):
    mocked_boto.resource().Bucket().objects.filter.return_value = [
        mock.MagicMock()
    ] * 10
    item_lens = []
    for i in range(10):
        items = list(get_s3_job_ads("reed", sample_ratio=0.2, read_body=False))
        assert all(item["body"] is None for item in items)
        item_lens.append(len(items))
    mean = sum(item_lens) / len(item_lens)

    assert all(_len < 10 for _len in item_lens)  # i.e. all sampled
    assert mean > 0 and mean < 4  # i.e. around 0.2 x 10


@mock.patch(PATH.format("db"))
def test_get_db_job_ads_limit(mocked_db):
    mocked_db.object_as_dict.side_effect = lambda x: {"id": x.id}
    session = mocked_db.db_session().__enter__()
    query = session.query().order_by()
    # Create objects with an id attribute
    objs = []
    for i in range(0, 10):
        obj = mock.Mock()
        obj.id = i
        objs.append(obj)
    query.limit.return_value = objs  # First chunk
    query.filter().limit.return_value = objs  # All other chunks

    # Check limit works
    for limit in range(7, 213):
        print(limit)
        ad_iter = get_db_job_ads(chunksize=10, limit=limit)
        assert len(list(ad_iter)) == limit


@mock.patch(PATH.format("db"))
def test_get_db_job_ads_chunksize(mocked_db):
    mocked_db.object_as_dict.side_effect = lambda x: {"id": x.id}
    session = mocked_db.db_session().__enter__()
    query = session.query().order_by()

    # Create objects with an id attribute
    objs = []
    for i in range(0, 10):
        obj = mock.Mock()
        obj.id = i
        objs.append(obj)
    query.limit.return_value = objs  # len = chunksize
    query.filter().limit.return_value = objs[:8]  # len != chunksize

    # Check limit works
    ad_iter = get_db_job_ads(chunksize=10, limit=12)
    assert len(list(ad_iter)) == 12

    # Check chunksize works
    ad_iter = get_db_job_ads(chunksize=10)  # no limit
    assert len(list(ad_iter)) == 18


@mock.patch(
    PATH.format("get_location_lookup"),
    return_value={"location_1": "foo", "location_3": "bar"},
)
@mock.patch(PATH.format("db"))
def test_get_locations(mocked_db, mocked_get_location_lookup):
    session = mocked_db.db_session().__enter__()
    query = session.query().distinct().outerjoin()
    query.all.return_value = [
        ("id_1", "location_1"),
        ("id_2", None),
        ("id_3", "location_3"),
    ]
    assert list(get_locations("nuts_2", do_lookup=True)) == [
        {"job_id": "id_1", "nuts_2_code": "location_1", "nuts_2_name": "foo"},
        {"job_id": "id_3", "nuts_2_code": "location_3", "nuts_2_name": "bar"},
    ]


@mock.patch(
    PATH.format("get_location_lookup"),
    return_value={"location_1": "foo", "location_3": "bar"},
)
@mock.patch(PATH.format("db"))
def test_get_locations_no_lookup(mocked_db, mocked_get_location_lookup):
    session = mocked_db.db_session().__enter__()
    query = session.query().distinct().outerjoin()
    query.all.return_value = [
        ("id_1", "location_1"),
        ("id_2", None),
        ("id_3", "location_3"),
    ]
    assert list(get_locations("nuts_2", do_lookup=False)) == [
        {"job_id": "id_1", "nuts_2_code": "location_1"},
        {"job_id": "id_3", "nuts_2_code": "location_3"},
    ]


@mock.patch(PATH.format("db"))
def test_get_location_lookup(mocked_db):
    mocked_db.object_as_dict.side_effect = lambda x: x
    session = mocked_db.db_session().__enter__()
    session.query().all.return_value = [
        {
            "foo_code": "123",
            "foo_name": "abc",
            "ipn_code": "1234",  # IPN will be ignore
            "ipn_name": "abcd",
            "bar_code": "123",  # duplicate
            "bar_name": "ab",  # but take this value because it's the shortest
        },
        {
            "foo_code": "456",
            "foo_name": "abcdef",
            "ipn_code": "1234",  # IPN will be ignore
            "ipn_name": "abcd",
            "bar_code": "123",  # another duplicate
            "bar_name": "abcdefg",
        },
        {
            "foo_code": "456",
            "foo_name": "",  # Ignore empty
        },
    ]
    assert get_location_lookup() == {"123": "ab", "456": "abcdef"}


@mock.patch(PATH.format("db"))
def test_get_salaries(mocked_db):
    session = mocked_db.db_session().__enter__()
    session.query().all.return_value = [
        {
            "id": "foo",
            "__version__": "bar",
            "a_decimal": Decimal("12.3"),
            "a_float": 23.4,
            "a_str": "45.6",
        }
    ]
    mocked_db.object_as_dict.side_effect = lambda x: x
    assert list(get_salaries()) == [
        {"job_id": "foo", "a_decimal": 12.3, "a_float": 23.4, "a_str": "45.6"}
    ]


@mock.patch(PATH.format("db"))
def test_get_skills(mocked_db):
    session = mocked_db.db_session().__enter__()
    session.query().all.return_value = [
        {
            "job_data_source": "foo",
            "job_id": 123,
            "__version__": "bar",
            "a_decimal": Decimal("12.3"),
            "a_float": 23.4,
            "a_str": "45.6",
        }
    ]
    mocked_db.object_as_dict.side_effect = lambda x: x
    assert list(get_skills()) == [
        {"job_id": "123", "a_decimal": 12.3, "a_float": 23.4, "a_str": "45.6"}
    ]


@mock.patch(PATH.format("db"))
def test_get_requires_degree(mocked_db):
    session = mocked_db.db_session().__enter__()
    session.query().all.return_value = [
        {
            "id": "1234",
            "__version__": "bar",
            "a_decimal": Decimal("12.3"),
            "a_float": 23.4,
            "a_str": "45.6",
        }
    ]
    mocked_db.object_as_dict.side_effect = lambda x: x

    assert list(get_requires_degree()) == [
        {
            "job_id": "1234",
            "a_decimal": Decimal("12.3"),
            "a_float": 23.4,
            "a_str": "45.6",
        }
    ]
