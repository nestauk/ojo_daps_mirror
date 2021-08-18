from unittest import mock
from ojd_daps.flows.enrich.labs.salaries.regex.multi_regex_utils import (
    regex_model,
    annualised_salary,
    extract_salary,
    save_model,
    load_model,
    apply_model,
)

PATH = "ojd_daps.flows.enrich.labs.salaries.regex.multi_regex_utils.{}"


def test_regex_model():
    model = regex_model(r"(\d*[.]?\d*)")
    assert model("100.0 to 1000.0") == [100.0, 1000.0]
    assert model("1000 and 7777.234 - 88.123") == [1000, 7777.234, 88.123]
    assert model("22000.0000-27000.0000") == [22000.0, 27000.0]
    assert model("no salary") == []


def test_annualised_salary():
    assert annualised_salary(225, "per day") == 58500
    assert annualised_salary(9.5, "per hour") == 18525
    assert annualised_salary(66000, "per annum") == 66000


def test_extract_salary():
    salaries = [[22000.0, 27000.0], [30.0, 500.0], [24.46, 24.46]]
    extracts = [
        {
            "min_salary": 22000.0,
            "max_salary": 27000.0,
            "rate": "per annum",
            "min_annualised_salary": 22000.0,
            "max_annualised_salary": 27000.0,
        },
        {
            "min_salary": 30.0,
            "max_salary": 500.0,
            "rate": "per day",
            "min_annualised_salary": 7800.0,
            "max_annualised_salary": 130000.0,
        },
        {
            "min_salary": 24.46,
            "max_salary": 24.46,
            "rate": "per hour",
            "min_annualised_salary": 47697.0,
            "max_annualised_salary": 47697.0,
        },
    ]
    for salary, extract in zip(salaries, extracts):
        assert extract_salary(salary) == extract


@mock.patch(PATH.format("save_to_s3"))
def test_save_model(mocked_save_to_s3):
    save_model("a")
    args, _ = mocked_save_to_s3.call_args
    assert args == ("regex.txt", "a")


@mock.patch(PATH.format("load_from_s3"), side_effect=lambda x: x)
@mock.patch(PATH.format("regex_model"), side_effect=lambda x: x)
def test_load_model(mocked_s3, mocked_model):
    assert load_model() is load_model()  # Check the cache is "working"
    assert load_model() == ("regex.txt")


@mock.patch(PATH.format("load_model"), side_effect=lambda: (lambda x: x.upper()))
@mock.patch(PATH.format("extract_salary"), side_effect=lambda x: x * 2)
def test_apply_model(mocked_load, mocked_extract):
    row = {"job_salary_raw": "hello"}
    assert apply_model(row) == "HELLOHELLO"
    row = {"job_salary_raw": None}
    assert apply_model(row) == None
