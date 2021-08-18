from unittest import mock

from ojd_daps.dqa.data_getters import (
    get_s3_job_ads,
    get_db_job_ads,
    get_locations,
    get_location_lookup,
    query_salary,
    get_salaries,
)


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
    query = session.query().filter().order_by()

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
        ad_iter = get_db_job_ads("reed", chunksize=10, limit=limit)
        assert len(list(ad_iter)) == limit


@mock.patch(PATH.format("db"))
def test_get_db_job_ads_chunksize(mocked_db):
    mocked_db.object_as_dict.side_effect = lambda x: {"id": x.id}
    session = mocked_db.db_session().__enter__()
    query = session.query().filter().order_by()

    # Create objects with an id attribute
    objs = []
    for i in range(0, 10):
        obj = mock.Mock()
        obj.id = i
        objs.append(obj)
    query.limit.return_value = objs  # len = chunksize
    query.filter().limit.return_value = objs[:8]  # len != chunksize

    # Check limit works
    ad_iter = get_db_job_ads("reed", chunksize=10, limit=12)
    assert len(list(ad_iter)) == 12

    # Check chunksize works
    ad_iter = get_db_job_ads("reed", chunksize=10)  # no limit
    assert len(list(ad_iter)) == 18


@mock.patch(PATH.format("db"))
def test_get_locations(mocked_db):
    session = mocked_db.db_session().__enter__()
    query = session.query().distinct().filter().outerjoin()
    query.all.return_value = [(1, 2), ("3", "4")]

    assert get_locations("dummy", "blah") == [
        {"job_id": 1, "blah_code": 2},
        {"job_id": "3", "blah_code": "4"},
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
def test_query_salary(mocked_db):
    mocked_db.object_as_dict.side_effect = lambda x: x
    session = mocked_db.db_session().__enter__()
    query = session.query().filter()
    query.all.return_value = [1, 2, 3, "4"]
    assert list(query_salary()) == [1, 2, 3, "4"]


@mock.patch(PATH.format("query_salary"))
def test_get_salaries(mocked_query_salary):
    mocked_query_salary.side_effect = lambda rate: [{"id": "1234", "salary": "100"}] * 2

    assert get_salaries() == [
        {"id": "1234", "salary": 100, "rate": "per annum"},
        {"id": "1234", "salary": 100, "rate": "per annum"},
        {"id": "1234", "salary": 100 * 52 * 5, "rate": "per day"},
        {"id": "1234", "salary": 100 * 52 * 5, "rate": "per day"},
        {"id": "1234", "salary": 100 * 52 * 37.5, "rate": "per hour"},
        {"id": "1234", "salary": 100 * 52 * 37.5, "rate": "per hour"},
    ]
