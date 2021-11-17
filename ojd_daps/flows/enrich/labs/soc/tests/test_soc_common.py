from unittest import mock
import pytest
from ojd_daps.flows.enrich.labs.soc.common import (
    S3_PATH,
    save_json_to_s3,
    load_json_from_s3,
    _load_metadata,
    load_metadata,
    remove_digits,
    replace_punctuation,
    standardise_text,
    replace_or_remove,
    remove_prefix,
    clean_raw_job_title,
)

PATH = "ojd_daps.flows.enrich.labs.soc.common.{}"


@pytest.fixture
def sample_text():
    return "thiES23989 Are 12321.12?Some12.21424 232Digits!"


@mock.patch(PATH.format("save_to_s3"))
def test_save_json_to_s3(mocked_save):
    save_json_to_s3("something", {"some data": "a value"})
    args, kwargs = mocked_save.call_args
    assert args == (S3_PATH, "something.json", '{"some data": "a value"}')


@mock.patch(PATH.format("load_from_s3"), side_effect=lambda _, x: rf'{{"x": "{x}"}}')
def test_load_json_from_s3(mocked_load):
    assert load_json_from_s3("something") is load_json_from_s3("something")
    assert load_json_from_s3("abc") == {"x": "abc.json"}  # <-- note suffix


def test_load_metadata():
    """Two in one: _load_metadata and load_metadata"""
    metadata = _load_metadata()
    assert len(metadata.keys()) > 3
    for key in metadata.keys():
        _metadata = load_metadata(key)
        assert len(_metadata) > 0


def test_remove_digits(sample_text):
    assert remove_digits(sample_text) == "thiES Are .?Some. Digits!"


def test_replace_punctuation(sample_text):
    assert (
        replace_punctuation(sample_text)
        == "thiES23989 Are 12321 12 Some12 21424 232Digits"
    )


def test_standardise_text(sample_text):
    standardise_text(sample_text) == "thies are some digits"


def test_replace_remove_remove_mode():
    """Running in 'remove' mode (i.e. with list)"""
    text_to_change = "this is the text to change the meaning"
    words_to_remove = ["to", "the", "change"]
    expected = "this is text meaning"  # Note removed 'the' twice
    replace_or_remove(text_to_change, words_to_remove) == expected


def test_replace_remove_replace_mode():
    """Running in 'replace' mode (i.e. with dict)"""
    text_to_change = "this is the text to change the meaning"
    words_to_remove = {"to": "2", "the": "THE", "change": "alter"}
    expected = "this is THE text 2 alter THE meaning"  # Note replace 'the' twice
    replace_or_remove(text_to_change, words_to_remove) == expected


def test_remove_prefix():
    assert remove_prefix("account manager", "account") == "manager"
    assert remove_prefix("account manager", "manager") == "account manager"
    assert remove_prefix("accountant manager", "account") == "accountant manager"
    assert remove_prefix("the account manager", "account") == "the account manager"


@mock.patch(PATH.format("load_json_from_s3"))
@mock.patch(PATH.format("load_metadata"), return_value=["thies"])
def test_clean_raw_job_title(mocked_load, mocked_json, sample_text):
    clean_raw_job_title(sample_text) == "are some digits"
