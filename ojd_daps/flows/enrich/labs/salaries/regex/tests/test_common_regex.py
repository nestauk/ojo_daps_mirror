from unittest import mock
from ojd_daps.flows.enrich.labs.salaries.regex.common import (
    BUCKET_NAME,
    S3_PATH,
    guess_rate,
    save_to_s3,
    load_from_s3,
)

PATH = "ojd_daps.flows.enrich.labs.salaries.regex.common.{}"


def test_guess_rate():
    for salary in (1, 10, 49.99):
        assert guess_rate(salary) == "per hour"
    for salary in (50, 1000, 9999.99):
        assert guess_rate(salary) == "per day"
    for salary in range(10000, 100000):
        assert guess_rate(salary) == "per annum"


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
