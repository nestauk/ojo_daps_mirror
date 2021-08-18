from unittest import mock
from pytest import raises
from ojd_daps.flows.enrich.labs.deduplication.faiss_utils import \
    BUCKET_NAME, S3_PATH, save_to_s3, load_from_s3, save_model,\
    class_for_name

PATH = "ojd_daps.flows.enrich.labs.deduplication.faiss_utils.{}"

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
    save_model("a", "b", "c", "d", "e")
    ((args1, kwargs1), (args2, kwargs2),
    (args3, kwargs3), (args4, kwargs4),
    (args5, kwargs5)) = mocked_save_to_s3.call_args_list
    assert args1 == ("k.txt", "a")
    assert args2 == ("k_large.txt", "b")
    assert args3 == ("n_clusters.txt", "c")
    assert args4 == ("score_threshold.txt", "d")
    assert args5 == ("metric.txt", "e")

def test_class_for_name():
    assert isinstance(class_for_name('faiss','METRIC_L1'), int)
    with raises(AttributeError):
        class_for_name('faiss','METRIC_WRONG')
