import pytest
from unittest import mock
import datetime

from ojd_daps.flows.enrich.deduplication_utils import (
    get_sliding_windows,
    query_and_bundle,
    prefill_inputs,
    read_data,
    query_to_str,
    create_base_query,
    download_vectors,
    generate_window_query_chunks,
)
from ojd_daps.flows.enrich.deduplication_utils import VECTOR_DIM, np, json, CHUNKSIZE

PATH = "ojd_daps.flows.enrich.deduplication_utils.{}"
START_DATE = datetime.datetime(2020, 10, 16, 0, 0)
END_DATE = datetime.datetime(2020, 12, 1, 0, 0)


def test_sliding_window_getter_4_week():
    expected = [
        [str(START_DATE), "2020-11-13 00:00:00"],
        ["2020-10-30 00:00:00", "2020-11-27 00:00:00"],
        ["2020-11-13 00:00:00", "2020-12-11 00:00:00"],
        ["2020-11-27 00:00:00", "2020-12-25 00:00:00"],
    ]
    assert get_sliding_windows(START_DATE, END_DATE, interval=4 * 7) == expected


def test_sliding_window_getter_6_week():
    expected = [
        ["2020-10-16 00:00:00", "2020-11-27 00:00:00"],
        ["2020-11-06 00:00:00", "2020-12-18 00:00:00"],
        ["2020-11-27 00:00:00", "2021-01-08 00:00:00"],
    ]
    assert get_sliding_windows(START_DATE, END_DATE, interval=6 * 7) == expected


@mock.patch(PATH.format("VECTOR_DIM"), 2)
def test_query_and_bundle():
    mocked_session = mock.Mock()
    mocked_session.execute().fetchall.return_value = [
        ("aa", '["111", "222"]'),
        ("bb", '["222", "333"]'),
        ("cc", '["333", "444"]'),
        ("dd", '["444", "555"]'),
        ("ee", '["555", "666"]'),
    ]
    _ids, _vectors = query_and_bundle(mocked_session, "query")
    assert _ids.flatten().tolist() == ["aa", "bb", "cc", "dd", "ee"]
    assert _vectors.tolist() == [
        [111.0, 222.0],
        [222.0, 333.0],
        [333.0, 444.0],
        [444.0, 555.0],
        [555.0, 666.0],
    ]


def test_prefill_inputs():
    for i in range(10, 20):
        data, ids = prefill_inputs(i)
        assert sum(map(len, ids)) == 0
        assert data.shape == (i, VECTOR_DIM)
        assert ids.shape == (i,)


@mock.patch(PATH.format("VECTOR_DIM"), 3)  # 3 elements per vector
@mock.patch(PATH.format("CHUNKSIZE"), 2)  # 2 chunks of 2 rows
@mock.patch(PATH.format("query_and_bundle"))
def test_read_data(mocked_query_and_bundle):
    mocked_query_and_bundle.side_effect = [
        # ids [1,4], data [3,4]
        (np.array(["a", "c"]), np.array([[1.0, 2.0, 3.0], [4.0, 6.0, 8.0]])),
        (np.array(["f", "b"]), np.array([[34.0, 23.0, 0.23], [-1.0, 1.0, 10.0]])),
    ]
    data, ids = prefill_inputs(4)  # 4 rows of data
    read_data(data, ids, "session", ["query"] * 2)

    np.testing.assert_almost_equal(
        data.tolist(),
        [
            [1.0, 2.0, 3.0],
            [4.0, 6.0, 8.0],
            [34.0, 23.0, 0.23],
            [-1.0, 1.0, 10.0],
        ],
    )
    assert ids.tolist() == ["a", "c", "f", "b"]


def test_query_to_str():
    from ojd_daps.orms.raw_jobs import RawJobAd
    from sqlalchemy.orm import Query

    tblname = RawJobAd.__tablename__
    q = Query(RawJobAd.id).filter(RawJobAd.id >= 123)
    expected = f"SELECT {tblname}.id \nFROM {tblname} \nWHERE {tblname}.id >= 123"
    assert query_to_str(q) == expected


def test_create_base_query():
    from sqlalchemy.orm import Query
    from ojd_daps.orms.raw_jobs import RawJobAd
    from ojd_daps.orms.raw_jobs import JobAdDescriptionVector as Vector

    mocked_session = mock.Mock()
    mocked_session.query.side_effect = lambda *fields: Query(fields)
    query = create_base_query(mocked_session, [Vector.id, Vector.vector], "now", "then")
    query_str = query_to_str(query)

    vectabl = Vector.__tablename__
    tblname = RawJobAd.__tablename__
    assert query_str == (
        f"SELECT {vectabl}.id, {vectabl}.vector \n"
        f"FROM {vectabl} JOIN {tblname} ON "
        f"{vectabl}.id = {tblname}.id \n"
        f"WHERE {tblname}.created BETWEEN 'now' AND 'then'"
    )


@mock.patch(PATH.format("prefill_inputs"), return_value=(None, None))
@mock.patch(PATH.format("read_data"))
def test_download_vectors(mocked_read, mocked_prefill):
    mocked_read.side_effect = [json.JSONDecodeError("a", "b", 1)] * 3 + [None] * 1000
    download_vectors("session", "queries", "count")
    assert mocked_read.call_count == 4  # all json.JSONDecodeError then first None


@mock.patch(PATH.format("prefill_inputs"), return_value=(None, None))
@mock.patch(PATH.format("read_data"))
def test_download_vectors_too_much_fail(mocked_read, mocked_prefill):
    mocked_read.side_effect = [json.JSONDecodeError("a", "b", 1)] * 3 + [None] * 1000
    with pytest.raises(json.JSONDecodeError):
        download_vectors("session", "queries", "count", max_errors=2)
    assert mocked_read.call_count == 2


@mock.patch(PATH.format("create_base_query"))
def test_generate_window_query_chunks(mocked_base_query):
    from sqlalchemy.orm import Query
    import re

    mock_vector_query = Query("entities")
    mock_ids_query = mock.Mock()
    mocked_base_query.side_effect = [mock_vector_query, mock_ids_query]

    mock_ids_query.count.return_value = CHUNKSIZE * 12.3
    mock_ids_query.limit().one.return_value = ("pk",)
    mock_ids_query.filter().offset().limit().one.return_value = ("pk",)
    queries, count = generate_window_query_chunks("session", START_DATE, END_DATE)

    assert len(queries) == 13

    RE_long = "SELECT (.*) \nFROM (.*) \nWHERE (.*) >= (.*) AND (.*) < (.*)"
    RE_short = "SELECT (.*) \nFROM (.*) \nWHERE (.*) >= (.*)"
    for q in queries[:12]:
        assert re.match(RE_long, q) is not None
    assert re.match(RE_short, queries[-1]) is not None
