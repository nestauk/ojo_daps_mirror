"""
labs.salaries.common
--------------------

Common utils for extracting salary info from raw job ads in the database.
"""
from copy import deepcopy

WEEKS_IN_YEAR = 52
WORKDAYS_PER_WEEK = 5
HOURS_PER_WEEK = 37.5  # Standard UK weekly hours
PER_ANNUM_RATE = {  # Conversion to annual from other rates
    "DAY": WORKDAYS_PER_WEEK * WEEKS_IN_YEAR,
    "HOUR": HOURS_PER_WEEK * WEEKS_IN_YEAR,
    "YEAR": 1,
}
MIN_SALARY = (
    4.20 * PER_ANNUM_RATE["HOUR"]
)  # Minimum wage in GBP for apprenticeships (Sept 2021)
MIN_SALARY_DISPARITY = 10  # Max salary cannot be more than this times the min salary
MAX_SALARY = 500_000  # "Crude" max salary cut-off in GBP


def remove_null_values(_dict):
    """
    Remove all keys with values equal to None. This allows the `extract_salary`
    to take advantage of dict.get returning None anyway in the case that the key
    doesn't exist whilst offering an alternative dfault value.
    """
    _dict_copy = deepcopy(_dict)
    for k in _dict.keys():
        if _dict[k] is None:
            _dict_copy.pop(k)
    return _dict_copy


def extract_salary(job_ad):
    """Make salary extraction dictionary"""
    # Remove keys from null values
    job_ad = remove_null_values(job_ad)

    # Exclude if a salary hasn't been found
    rate = job_ad.get("raw_salary_unit")
    if rate is None:
        return None

    # Exclude non-GBP
    currency = job_ad.get("raw_salary_currency", "GBP")
    if currency != "GBP":
        return None

    # Assign min and max salaries, and make sure they're in order
    salary = job_ad.get("raw_salary")
    min_salary = job_ad.get("raw_min_salary", salary)
    max_salary = job_ad.get("raw_max_salary", salary)
    min_salary, max_salary = map(float, sorted((min_salary, max_salary)))

    # Convert to annual
    min_annual = min_salary * PER_ANNUM_RATE[rate]
    max_annual = max_salary * PER_ANNUM_RATE[rate]

    # At least minimum wage
    if max_annual < MIN_SALARY:
        return None

    # Symettrise the salaries if the min salary doesn't make sense
    if min_annual < MIN_SALARY:
        min_annual = max_annual

    # If the salary range is vast
    if max_annual / min_annual > MIN_SALARY_DISPARITY:
        # Lower the upper band if extreme
        if max_annual > MAX_SALARY:
            max_annual = min_annual
        # Otherwise raise the lower band
        else:
            min_annual = max_annual

    # Exclude extreme salaries that remain after the above corrections
    if max_annual > MAX_SALARY:
        return None

    return {
        "min_salary": round(min_annual / PER_ANNUM_RATE[rate], 2),
        "max_salary": round(max_annual / PER_ANNUM_RATE[rate], 2),
        "min_annualised_salary": round(min_annual, 2),
        "max_annualised_salary": round(max_annual, 2),
        "rate": rate,
    }
