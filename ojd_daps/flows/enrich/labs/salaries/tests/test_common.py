import pytest
from ojd_daps.flows.enrich.labs.salaries.common import (
    remove_null_values,
    extract_salary,
)
from ojd_daps.flows.enrich.labs.salaries.common import (
    PER_ANNUM_RATE,
    MIN_SALARY,
    MIN_SALARY_DISPARITY,
    MAX_SALARY,
)


def get_job_ad():
    return {
        "raw_salary_unit": "YEAR",
        "raw_salary_currency": "GBP",
        "raw_salary": 15_000.0,
        "raw_min_salary": 10_000.0,
        "raw_max_salary": 20_000.0,
    }


@pytest.fixture
def job_ad_yearly():
    job_ad = get_job_ad()
    return job_ad


@pytest.fixture
def job_ad_daily():
    job_ad = get_job_ad()
    job_ad["raw_salary_unit"] = "DAY"
    for field in ["raw_salary", "raw_min_salary", "raw_max_salary"]:
        job_ad[field] = job_ad[field] / PER_ANNUM_RATE["DAY"]
    return job_ad


@pytest.fixture
def job_ad_hourly():
    job_ad = get_job_ad()
    job_ad["raw_salary_unit"] = "HOUR"
    for field in ["raw_salary", "raw_min_salary", "raw_max_salary"]:
        job_ad[field] = job_ad[field] / PER_ANNUM_RATE["HOUR"]
    return job_ad


def test_rate():
    # To guard against changes
    assert PER_ANNUM_RATE == {
        "DAY": 5 * 52,
        "HOUR": 37.5 * 52,
        "YEAR": 1,
    }


def test_min_salary():
    # To guard against changes
    assert 8_000 < MIN_SALARY < 10_000


def test_disparity():
    # To guard against changes
    assert MIN_SALARY_DISPARITY >= 10


def test_max_salary():
    assert MAX_SALARY > 250_000


def test_remove_null_values():
    in_dict = {"foo": "bar", "baz": None, "ham": "eggs"}
    out_dict = {"foo": "bar", "ham": "eggs"}
    n = len(in_dict)
    assert remove_null_values(in_dict) == out_dict
    assert len(in_dict) == n  # i.e. unchanged


def test_extract_salary(job_ad_yearly, job_ad_daily, job_ad_hourly):
    assert extract_salary(job_ad_yearly) == {
        "min_salary": 10_000.0,
        "max_salary": 20_000.0,
        "min_annualised_salary": 10_000.0,
        "max_annualised_salary": 20_000.0,
        "rate": "YEAR",
    }

    assert extract_salary(job_ad_daily) == {
        "min_salary": 38.46,
        "max_salary": 76.92,
        "min_annualised_salary": 10_000.0,
        "max_annualised_salary": 20_000.0,
        "rate": "DAY",
    }

    assert extract_salary(job_ad_hourly) == {
        "min_salary": 5.13,
        "max_salary": 10.26,
        "min_annualised_salary": 10_000.0,
        "max_annualised_salary": 20_000.0,
        "rate": "HOUR",
    }


def test_extract_salary_no_min_max(job_ad_yearly):
    job_ad_yearly["raw_min_salary"] = None
    job_ad_yearly["raw_max_salary"] = None
    assert extract_salary(job_ad_yearly) == {
        "min_salary": 15_000.0,
        "max_salary": 15_000.0,
        "min_annualised_salary": 15_000.0,
        "max_annualised_salary": 15_000.0,
        "rate": "YEAR",
    }


def test_extract_salary_wrong_min_max_order(job_ad_yearly):
    job_ad_yearly["raw_min_salary"] = 25_000.0
    job_ad_yearly["raw_max_salary"] = 15_000.0
    assert extract_salary(job_ad_yearly) == {
        "min_salary": 15_000.0,
        "max_salary": 25_000.0,
        "min_annualised_salary": 15_000.0,
        "max_annualised_salary": 25_000.0,
        "rate": "YEAR",
    }


def test_extract_salary_no_currency(job_ad_yearly):
    job_ad_yearly["raw_salary_currency"] = None
    assert extract_salary(job_ad_yearly) == {
        "min_salary": 10_000.0,
        "max_salary": 20_000.0,
        "min_annualised_salary": 10_000.0,
        "max_annualised_salary": 20_000.0,
        "rate": "YEAR",
    }


def test_extract_salary_other_currency(job_ad_yearly):
    for currency in ("EUR", "USD", "other"):
        job_ad_yearly["raw_salary_currency"] = "EUR"
        assert extract_salary(job_ad_yearly) is None


def test_extract_salary_no_rate(job_ad_yearly):
    job_ad_yearly["raw_salary_unit"] = None
    assert extract_salary(job_ad_yearly) is None


def test_extract_salary_max_too_low(job_ad_yearly):
    job_ad_yearly["raw_min_salary"] = 0
    job_ad_yearly["raw_max_salary"] = 7_000
    assert extract_salary(job_ad_yearly) is None


def test_extract_salary_min_too_low(job_ad_yearly):
    job_ad_yearly["raw_min_salary"] = 7_000.0
    job_ad_yearly["raw_max_salary"] = 12_000.0
    assert extract_salary(job_ad_yearly) == {
        "min_salary": 12_000.0,
        "max_salary": 12_000.0,
        "min_annualised_salary": 12_000.0,
        "max_annualised_salary": 12_000.0,
        "rate": "YEAR",
    }


def test_extract_salary_disparity_big_max(job_ad_yearly):
    job_ad_yearly["raw_min_salary"] = 9_000.0
    job_ad_yearly["raw_max_salary"] = 1_000_000.0  # million
    assert extract_salary(job_ad_yearly) == {
        "min_salary": 9_000.0,
        "max_salary": 9_000.0,
        "min_annualised_salary": 9_000.0,
        "max_annualised_salary": 9_000.0,
        "rate": "YEAR",
    }


def test_extract_salary_disparity_small_min(job_ad_yearly):
    job_ad_yearly["raw_min_salary"] = 9_000.0
    job_ad_yearly["raw_max_salary"] = 400_000.0  # under the threshold
    assert extract_salary(job_ad_yearly) == {
        "min_salary": 400_000.0,
        "max_salary": 400_000.0,
        "min_annualised_salary": 400_000.0,
        "max_annualised_salary": 400_000.0,
        "rate": "YEAR",
    }


def test_extract_salary_disparity_big_min(job_ad_yearly):
    job_ad_yearly["raw_min_salary"] = 600_000.0  # over the threshold
    job_ad_yearly["raw_max_salary"] = 7_000_000.0  # over the threshold
    assert extract_salary(job_ad_yearly) is None
