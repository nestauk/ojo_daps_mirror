"""
salaries.regex.multi_regex_utils
--------------

Regex to pull out min and max original and annualised
salaries, with a hard-coded guess at whether the
salary rate is hourly, daily or pa.

Do `apply_model(row)` to load and apply the model to data.

Run `multi_regex.py` in `jupyter lab` (with `jupytext` installed)
to see the model in action (it's surprisingly good).

NB: the regex I decided on was '(\\d*[.]?\\d*)'
"""
from functools import lru_cache
import re

from .common import (
    guess_rate,
    save_to_s3,
    load_from_s3,
)

PER_ANNUM_TO_RATE = {
    "per day": 5 * 52,
    "per hour": 37.5 * 52,
    "per annum": 1,
}

MIN_SALARY = 4.30  # Minimum wage for apprenticeships (Sept 2021)
MIN_SALARY_DISPARITY = 50  # Max salary cannot be more than this time min salary


def annualised_salary(salary, rate):
    """Calculates annual salary"""
    return round(salary * PER_ANNUM_TO_RATE[rate], 2)


def extract_salary(salaries):
    """Make salary extraction dictionary"""
    min_salary, max_salary = sorted(salaries)
    # Edge cases
    if max_salary < MIN_SALARY:
        return None  # explicit null value
    if (min_salary < MIN_SALARY) or (max_salary / min_salary > MIN_SALARY_DISPARITY):
        min_salary = max_salary
    rate = guess_rate(max_salary)
    return {
        "min_salary": min_salary,
        "max_salary": max_salary,
        "min_annualised_salary": annualised_salary(min_salary, rate),
        "max_annualised_salary": annualised_salary(max_salary, rate),
        "rate": rate,
    }


def regex_model(regex):
    """Returns a function which extracts list of numbers from a raw salary

    Args:
        regex (str): The regex strategy for picking out numbers from the raw salary.
    """
    re_ = re.compile(regex)
    return lambda raw_salary: [
        float(result) for result in re_.findall(raw_salary) if len(result) > 0
    ]


def save_model(regex):
    """Saves the model"""
    save_to_s3("regex.txt", regex)


@lru_cache()  # <--- Important
def load_model():
    """Loads the model"""
    regex = load_from_s3("regex.txt")
    return regex_model(regex)


def apply_model(row):
    """Loads and applies the model to the given row"""
    raw_salary = row["job_salary_raw"]
    if raw_salary is None:
        return None
    model = load_model()  # NB: lru_cached
    salaries = model(raw_salary)
    return extract_salary(salaries)
