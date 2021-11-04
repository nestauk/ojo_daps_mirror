from unittest import mock
from ojd_daps.flows.enrich.labs.locations.regex.regex_utils import (
    regex_model,
    BUCKET_NAME,
    S3_PATH,
    save_to_s3,
    load_from_s3,
    save_model,
    load_model,
    apply_model,
)

PATH = "ojd_daps.flows.enrich.labs.locations.regex.regex_utils.{}"


def test_regex_model():
    model = regex_model("[A-Z]{1,2}[0-9][0-9A-Z]?\s?", "BOILERPLATE")
    assert model("Telford, TF10") == "TF10"
    assert model("Telford, TF10 7XY") == "TF10"
    assert model("London, South East England") == "london"
    assert model("London BOILERPLATE, South East England") == "london"
    assert model("London TOWN BOILERPLATE, South East England") == "london_town"
    assert model("In London TOWN BOILERPLATE, South East England") == "in_london_town"
    assert model("In London TOWN BOILERPLATE, TF10 7XY") == "TF10"


@mock.patch(PATH.format("boto3"))
def test_save_to_s3(mocked_boto):
    mocked_s3 = mocked_boto.resource()
    mocked_obj = mocked_s3.Object()

    filename = "test.test"
    contents = "the contents"
    save_to_s3(filename, contents)

    args, kwargs = mocked_s3.Object.call_args
    assert args == (BUCKET_NAME, S3_PATH.format(filename))

    args, kwargs = mocked_obj.put.call_args
    assert args == tuple()
    assert kwargs == {"Body": contents}


@mock.patch(PATH.format("boto3"))
def test_load_from_s3(mocked_boto):
    mocked_s3 = mocked_boto.client()
    mocked_obj = mocked_s3.get_object()

    filename = "test.test"
    assert load_from_s3(filename) == mocked_obj["Body"].read().decode()

    args, kwargs = mocked_s3.get_object.call_args
    assert args == tuple()
    assert kwargs == {"Bucket": BUCKET_NAME, "Key": S3_PATH.format(filename)}


@mock.patch(PATH.format("save_to_s3"))
def test_save_model(mocked_save_to_s3):
    save_model("a", "b")
    ((args1, kwargs1), (args2, kwargs2)) = mocked_save_to_s3.call_args_list
    assert args1 == ("outcode_regex.txt", "a")
    assert args2 == ("boilerplate_text.txt", "b")


@mock.patch(PATH.format("load_from_s3"), side_effect=lambda x: x)
@mock.patch(PATH.format("regex_model"), side_effect=lambda x, y: (x, y))
def test_load_model(mocked_s3, mocked_model):
    assert load_model() is load_model()  # Check the cache is "working"
    assert load_model() == ("outcode_regex.txt", "boilerplate_text.txt")


@mock.patch(PATH.format("load_model"), side_effect=lambda: (lambda x: x.upper()))
def test_apply_model(mocked_load):
    row = {"job_location_raw": "hello"}
    assert apply_model(row) == "HELLO"
