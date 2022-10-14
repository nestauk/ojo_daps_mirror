from unittest import mock

from ojd_daps.flows.pre_enrich.location_lookup import (
    stream_metadata,
    read_csv,
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


def test_dedupe():
    data = [{"a": 123}, {"a": 1234}, {"a": 1234}, {"a": 12}, {"a": 123}]
    assert list(dedupe(data, primary_key="a")) == [{"a": 123}, {"a": 1234}, {"a": 12}]
