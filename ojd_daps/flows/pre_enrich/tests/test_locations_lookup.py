from unittest import mock

from ojd_daps.flows.pre_enrich.location_lookup import (
    stream_metadata,
    read_csv,
    extract_fields,
    suffix_lookup,
    NutsFinder,
    find_nuts,
    impute_nuts,
    dedupe,
    transform_metadata,
    StringIO,
)

PATH = "ojd_daps.flows.pre_enrich.location_lookup.{}"
CSV_DATA = StringIO('header1,header2\n"value1","value2"\n"value3","value4"')


@mock.patch(PATH.format("boto3"))
def test_stream_metadata(mocked_boto):
    s3 = mocked_boto.client()
    s3.get_object().__getitem__().read.return_value = b"A' Chr\xecon L\xe0raich"
    assert stream_metadata("bucket", "key").read() == "A' Chrìon Làraich"


@mock.patch(PATH.format("stream_metadata"), return_value=CSV_DATA)
def test_read_csv(mocked_stream):
    assert list(read_csv("bucket", "key")) == [
        {"header1": "value1", "header2": "value2"},
        {"header1": "value3", "header2": "value4"},
    ]


def test_extract_fields():
    row = {
        "abc": 123,
        "cde": 345,
        "place1824nm": 456,
        "ctry1cd": 56,
        "hlth183cd": 6,
        "rgn183nm": 67,
    }

    assert extract_fields(row, ignore_fields=[]) == {
        "ipn_1824_name": 456,
        "country_1_code": 56,
        "health_183_code": 6,
        "region_183_name": 67,
    }

    assert extract_fields(row, ignore_fields=["abc", "cde"]) == {
        "abc": 123,
        "cde": 345,
        "ipn_1824_name": 456,
        "country_1_code": 56,
        "health_183_code": 6,
        "region_183_name": 67,
    }


def test_find_nuts():
    nutses = list(find_nuts(57.057719, -4.500908))
    assert len(nutses) > 0
    for key, value in nutses:
        assert value not in (None, "")
        start, middle, end = key.split("_")
        assert start == "nuts"
        assert middle.isnumeric()
        assert end in ("code", "name")


@mock.patch(PATH.format("find_nuts"))
def test_impute_nuts(mocked_nuts):
    mocked_nuts.return_value = [("key", "value"), ("another key", "another value")]
    row = {"lat": 123, "long": 234, "other": "something"}
    assert impute_nuts(row) == {
        "other": "something",
        "key": "value",
        "another key": "another value",
    }


def test_dedupe():
    data = [{"a": 123}, {"a": 1234}, {"a": 1234}, {"a": 12}, {"a": 123}]
    assert list(dedupe(data, primary_key="a")) == [{"a": 123}, {"a": 1234}, {"a": 12}]


@mock.patch(PATH.format("extract_fields"))
@mock.patch(PATH.format("impute_nuts"))
def test_transform_metadata(mocked_impute, mocked_nuts):
    mocked_impute.side_effect = lambda x: x
    mocked_nuts.side_effect = lambda x: x
    row = {"foo": "bar", "baz": None, "abc": 123}
    assert transform_metadata(row) == {"foo": "bar", "abc": 123}
