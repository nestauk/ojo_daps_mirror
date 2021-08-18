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


def annualised_salary(salary, rate):
    """Calculates annual salary"""
    if rate == "per day":
        return salary * 5 * 52
    elif rate == "per hour":
        return salary * 37.5 * 52
    else:
        return salary


def extract_salary(salary):
    """Make salary extraction dictionary"""
    salary_ext = {}
    salary_ext["min_salary"] = min(salary)
    salary_ext["max_salary"] = max(salary)
    salary_ext["rate"] = guess_rate(
        salary_ext["min_salary"] + salary_ext["max_salary"] / 2
    )  # for some the range is very large so use average
    salary_ext["min_annualised_salary"] = round(
        annualised_salary(salary_ext["min_salary"], salary_ext["rate"]), 2
    )
    salary_ext["max_annualised_salary"] = round(
        annualised_salary(salary_ext["max_salary"], salary_ext["rate"]), 2
    )
    return salary_ext


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
    salary = model(raw_salary)
    return extract_salary(salary)
