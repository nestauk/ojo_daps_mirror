from unittest import mock
from ojd_daps.flows.enrich.labs.salaries.regex.single_regex_utils import (
    regex_model,
    save_model,
    load_model,
    apply_model,
)

PATH = "ojd_daps.flows.enrich.labs.salaries.regex.single_regex_utils.{}"


def test_regex_model():
    regex = r"(\d*[.]?\d*)"

    model = regex_model(regex, "min")
    assert model("100.0 to 1000.0") == 100.0
    assert model("1000 and 7777.234 - 88.123") == 88.123

    model = regex_model(regex, "max")
    assert model("100.0 to 1000.0") == 1000.0
    assert model("1000 and 7777.234 - 88.123") == 7777.234

    model = regex_model(regex, "mean")
    assert model("100.0 to 1000.0") == 550.0
    assert model("1000 and 7777.234 - 88.123") == 2955.119

    model = regex_model(regex, "median")
    assert model("100.0 to 1000.0") == 550.0
    assert model("1000 and 7777.234 - 88.123") == 1000.0


@mock.patch(PATH.format("save_to_s3"))
def test_save_model(mocked_save_to_s3):
    save_model("a", "b")
    ((args1, kwargs1), (args2, kwargs2)) = mocked_save_to_s3.call_args_list
    assert args1 == ("regex.txt", "a")
    assert args2 == ("picker.txt", "b")


@mock.patch(PATH.format("load_from_s3"), side_effect=lambda x: x)
@mock.patch(PATH.format("regex_model"), side_effect=lambda x, y: (x, y))
def test_load_model(mocked_s3, mocked_model):
    assert load_model() is load_model()  # Check the cache is "working"
    assert load_model() == ("regex.txt", "picker.txt")


@mock.patch(PATH.format("load_model"), side_effect=lambda: (lambda x: x.upper()))
@mock.patch(PATH.format("guess_rate"), side_effect=lambda x: x.lower().title())
def test_apply_model(mocked_load, mocked_guess):
    row = {"job_salary_raw": "hello"}
    assert apply_model(row) == ("HELLO", "Hello")
