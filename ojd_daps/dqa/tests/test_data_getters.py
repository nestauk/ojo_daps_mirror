import pytest
from unittest import mock
import os
from datetime import datetime

os.environ["DATA_GETTERS_DISKCACHE"] = "0"

from ojd_daps.dqa.data_getters import (
    _get_skills,
    get_s3_job_ads,
    get_db_job_ads,
    get_locations,
    get_location_lookup,
    get_salaries,
    get_skills,
    get_requires_degree,
    get_duplicate_ids,
    get_duplicate_subgraphs,
    get_subgraphs_by_location,
    identify_duplicates,
    get_entity_chunks,
    monday_of_week,
    iterdates,
    get_snapshot_ads,
)

from ojd_daps.dqa.data_getters import Decimal


PATH = "ojd_daps.dqa.data_getters.{}"


def strptime(date):
    return datetime.strptime(date, "%d-%m-%Y")


@pytest.mark.parametrize(
    "date,monday",
    [
        ("07-11-2021", "01-11-2021"),
        ("08-11-2021", "08-11-2021"),  # A Monday
        ("09-11-2021", "08-11-2021"),
        ("10-11-2021", "08-11-2021"),
        ("11-11-2021", "08-11-2021"),
        ("12-11-2021", "08-11-2021"),
        ("13-11-2021", "08-11-2021"),
        ("14-11-2021", "08-11-2021"),
        ("15-11-2021", "15-11-2021"),
        ("16-11-2021", "15-11-2021"),
        ("17-11-2021", "15-11-2021"),
        ("05-01-2012", "02-01-2012"),  # A couple of other dates
        ("17-02-1997", "17-02-1997"),
    ],
)
def test_monday_of_week(date, monday):
    date = strptime(date)
    monday = strptime(monday)
    assert monday.weekday() == 0
    assert monday_of_week(date) == monday


def test_iterdates():
    assert list(
        iterdates(
            start_date=datetime(2021, 2, 1),
            end_date=datetime(2021, 3, 1),
            timespan_weeks=2,
        )
    ) == [
        (datetime(2021, 1, 18, 0, 0), datetime(2021, 2, 1, 0, 0)),
        (datetime(2021, 1, 25, 0, 0), datetime(2021, 2, 8, 0, 0)),
        (datetime(2021, 2, 1, 0, 0), datetime(2021, 2, 15, 0, 0)),
        (datetime(2021, 2, 8, 0, 0), datetime(2021, 2, 22, 0, 0)),
        (datetime(2021, 2, 15, 0, 0), datetime(2021, 3, 1, 0, 0)),
    ]


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
        print(limit)
        ad_iter = get_db_job_ads(chunksize=10, limit=limit)
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
def test_get_entity_chunks(mocked_db):
    session = mocked_db.db_session().__enter__()
    session.query().group_by().order_by.return_value = (
        ("foo", 8),  # will combine with eggs since 8 + 1 <= 10
        ("bar", 7),  # will combine with spam since 7 + 3 <= 10
        ("spam", 3),
        ("eggs", 1),
    )
    assert get_entity_chunks(chunksize=10) == [["foo", "eggs"], ["bar", "spam"]]


@mock.patch(PATH.format("get_entity_chunks"), return_value=[[None, None]])
@mock.patch(PATH.format("db"))
def test__get_skills(mocked_db, mocked_get_chunks):
    session = mocked_db.db_session().__enter__()
    session.query().order_by().filter().limit().offset().all.side_effect = [
        [["a job id", "an entity"], ["another id", "another entity"]],  # First chunk
        [],  # N'th chunk --> StopIteration
    ]
    mocked_db.object_as_dict.side_effect = lambda x: x
    assert list(_get_skills()) == [
        {
            "job_id": "a job id",
            "entity": "an entity",
        },
        {
            "job_id": "another id",
            "entity": "another entity",
        },
    ]


@mock.patch(PATH.format("_get_skills"))
@mock.patch(PATH.format("get_skills_lookup"))
def test_get_skills(mocked_lookup, mocked__get_skills):
    mocked_lookup.return_value = {"foo": {"value": "spam"}, "bar": {"value": "eggs"}}
    mocked__get_skills.return_value = (
        {"job_id": "123", "entity": "foo"},
        {"job_id": "234", "entity": "foo"},
        {"job_id": "234", "entity": "bar"},
        {"job_id": "345", "entity": "bar"},
    )

    assert list(get_skills()) == [
        {"job_id": "123", "skills": [{"entity": "foo", "value": "spam"}]},
        {
            "job_id": "234",
            "skills": [
                {"entity": "foo", "value": "spam"},
                {"entity": "bar", "value": "eggs"},
            ],
        },
        {"job_id": "345", "skills": [{"entity": "bar", "value": "eggs"}]},
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


@mock.patch(PATH.format("make_date_filter"))
@mock.patch(PATH.format("identify_duplicates"), side_effect=lambda ids, **kwargs: ids)
@mock.patch(PATH.format("db"))
def test_get_duplicate_ids(mocked_db, mocked_identify, mocked_filter):
    session = mocked_db.db_session().__enter__()
    session.query().filter().all.return_value = [(1,), (2,)]
    assert (
        set(
            get_duplicate_ids(
                min_weight="min_weight",
                max_weight="max_weight",
                split_by_location="split_by_location",
                from_date="from_date",
                to_date="to_date",
            )
        )
        == {1, 2}
    )


@mock.patch(PATH.format("db"))
def test_get_duplicate_subgraphs(mocked_db):
    # Subgraph 1: 1-2-3-4-5
    edges = [(1, 2), (2, 3), (1, 4), (2, 5)]
    # Subgraph 2: 6-7-10
    edges += [(6, 10), (10, 7)]
    # Subgraph 3: 8-9
    edges += [(8, 9), (8, 9)]  # Note: contains duplicate
    session = mocked_db.db_session().__enter__()
    session.query().filter().all.return_value = edges

    subgraphs = list(map(set, get_duplicate_subgraphs(min_weight=1, max_weight=1)))
    assert subgraphs == [{1, 2, 3, 4, 5}, {6, 7, 10}, {8, 9}]


@mock.patch(PATH.format("get_duplicate_subgraphs"))
@mock.patch(PATH.format("db"))
def test_get_subgraphs_by_location(mocked_db, mocked_subgraphs):
    mocked_subgraphs.return_value = [{1, 2, 3, 4, 5}, {6, 7, 10}, {8, 9}]

    session = mocked_db.db_session().__enter__()
    session.query().all.return_value = [
        # return in the form (job_id, location, description_length)
        # and so locations here are "foo" and "bar"
        # and description_lengths are in the range 1 - 8, noting that
        # < 5 will be rejected as "too short"
        (1, "foo", 6),
        (2, "foo", 1),  # too short
        (3, "foo", 8),
        (4, "foo", 4),  # too short
        (5, "bar", 6),  # only bar in the group
        (6, "foo", 6),  # only foo in the group
        (7, "bar", 6),
        (10, "bar", 6),
        (9, "foo", 1),  # too short
        (8, "foo", 6),  # only foo in the group
    ]

    subgraphs = list(map(set, get_subgraphs_by_location(min_weight=1, max_weight=1)))
    assert subgraphs == [{1, 3}, {7, 10}]


@mock.patch(
    PATH.format("get_subgraphs_by_location"), return_value=[{1, 2, 3}, {7, 10, 11}]
)
@mock.patch(
    PATH.format("get_duplicate_subgraphs"), return_value=[{1, 2, 3}, {7, 10, 11}]
)
def test_identify_duplicates_split(mocked_dupe_subgraphs, mocked_subgraphs_by_loc):
    dupe_ids = set(
        identify_duplicates(
            ids={7, 11, 3, 10}, min_weight=1, max_weight=1, split_by_location=True
        )
    )
    # 7 dropped as an exemplar, 3 is the only id in its group
    assert dupe_ids == {11, 10}
    # split_by_location == True
    assert mocked_dupe_subgraphs.call_count == 0
    assert mocked_subgraphs_by_loc.call_count == 1


@mock.patch(
    PATH.format("get_subgraphs_by_location"), return_value=[{1, 2, 3}, {7, 10, 11}]
)
@mock.patch(
    PATH.format("get_duplicate_subgraphs"), return_value=[{1, 2, 3}, {7, 10, 11}]
)
def test_identify_duplicates_no_split(mocked_dupe_subgraphs, mocked_subgraphs_by_loc):
    dupe_ids = set(
        identify_duplicates(
            ids={7, 11, 3, 10}, min_weight=1, max_weight=1, split_by_location=False
        )
    )
    # 7 dropped as an exemplar, 3 is the only id in its group
    assert dupe_ids == {11, 10}
    # split_by_location == False
    assert mocked_dupe_subgraphs.call_count == 1
    assert mocked_subgraphs_by_loc.call_count == 0
