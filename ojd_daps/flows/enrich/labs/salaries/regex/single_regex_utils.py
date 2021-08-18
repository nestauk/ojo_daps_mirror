"""
salaries.regex.single_regex_utils
--------------

Regex to pull out salaries, with a hard-coded
guess at whether the salary rate is hourly, daily or pa.

Do `apply_model(row)` to load and apply the model to data.

Run `regex.py` in `jupyter lab` (with `jupytext` installed)
to see the model in action (it's surprisingly good).

NB: the regex I decided on was '(\\d*[.]?\\d*)' and the 'picker'
is "max": i.e. I pick the highest number found in the salary string.
"""
from functools import lru_cache
import re
import numpy as np
from ojd_daps.flows.enrich.labs.salaries.regex.common import (
    guess_rate,
    save_to_s3,
    load_from_s3,
)


def regex_model(regex, picker_name):
    """Returns a function which extracts a number from a raw salary.

    Args:
        regex (str): The regex strategy for picking out numbers from the raw salary.
        picker_name (str): Name of the picker, assumed to be one of the
                           numpy functions like 'min', 'median', or 'max'.
    """
    re_ = re.compile(regex)
    picker = getattr(np, picker_name)
    return lambda raw_salary: picker(
            np.array(
                [result for result in re_.findall(raw_salary) if len(result) > 0],
                dtype=float
                )
            )


def save_model(regex, picker_name):
    """Save the model config to s3.

    Args:
        regex (str): The regex strategy for picking out numbers from the raw
                     salary.
        picker_name (str): Name of the picker, assumed to be one of the
                           numpy functions like 'min', 'median', or 'max'.
    """
    save_to_s3("regex.txt", regex)
    save_to_s3("picker.txt", picker_name)


@lru_cache()  # <--- Important
def load_model():
    """Loads the model"""
    regex = load_from_s3("regex.txt")
    picker_name = load_from_s3("picker.txt")
    return regex_model(regex, picker_name)


def apply_model(row):
    """Loads and applies the model to the given row"""
    model = load_model()  # NB: lru_cached
    salary = model(row["job_salary_raw"])
    rate = guess_rate(salary)
    return salary, rate
