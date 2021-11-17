from unittest import mock
from pytest import raises
from ojd_daps.flows.enrich.labs.deduplication.faiss_utils import (
    save_model,
    class_for_name,
    S3_PATH,
)

PATH = "ojd_daps.flows.enrich.labs.deduplication.faiss_utils.{}"


@mock.patch(PATH.format("save_to_s3"))
def test_save_model(mocked_save_to_s3):
    save_model("a", "b", "c", "d", "e")
    (
        (args1, kwargs1),
        (args2, kwargs2),
        (args3, kwargs3),
        (args4, kwargs4),
        (args5, kwargs5),
    ) = mocked_save_to_s3.call_args_list
    assert args1 == (S3_PATH, "k.txt", "a")
    assert args2 == (S3_PATH, "k_large.txt", "b")
    assert args3 == (S3_PATH, "n_clusters.txt", "c")
    assert args4 == (S3_PATH, "score_threshold.txt", "d")
    assert args5 == (S3_PATH, "metric.txt", "e")


def test_class_for_name():
    assert isinstance(class_for_name("faiss", "METRIC_L1"), int)
    with raises(AttributeError):
        class_for_name("faiss", "METRIC_WRONG")
