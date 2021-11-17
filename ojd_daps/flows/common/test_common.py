from unittest import mock
from ojd_daps.flows.common import (
    load_from_s3,
    save_to_s3,
    flatten,
    get_chunks,
)

PATH = "ojd_daps.flows.common.common.{}"


@mock.patch(PATH.format("boto3"))
def test_load_from_s3(mocked_boto):
    s3 = mocked_boto.client()
    file_obj = s3.get_object().__getitem__().read()
    file_obj.decode.return_value = "hello"
    assert load_from_s3("s3path", "blah") == "hello"


@mock.patch(PATH.format("boto3"))
def test_save_to_s3(mocked_boto):
    s3 = mocked_boto.resource()
    obj = s3.Object()
    save_to_s3("s3path", "blah", "the content!")
    args, kwargs = obj.put.call_args
    assert kwargs == {"Body": "the content!"}


def test_flatten():
    assert flatten([["a", "b"], ["c", "d"]]) == ["a", "b", "c", "d"]
    assert flatten([["e", "f", "g"]]) == ["e", "f", "g"]


def test_get_chunks():
    assert get_chunks(["a", "b"], 1) == [["a"], ["b"]]
    assert get_chunks(["a", "b"], 2) == [["a", "b"]]
