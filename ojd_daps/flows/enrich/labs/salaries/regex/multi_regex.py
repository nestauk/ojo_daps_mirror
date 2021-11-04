# %%
from daps_utils.db import object_as_dict, db_session
from ojd_daps.orms.raw_jobs import RawJobAd
from functools import lru_cache
import boto3
import re

############
# Following lines are needed until issue is fixed in daps_utils
from daps_utils import db
import ojd_daps

db.CALLER_PKG = ojd_daps
db_session = db.db_session
############

S3_PATH = "labs/salaries/regex/{}"
BUCKET_NAME = "open-jobs-lake"


# %%
def guess_rate(salary):
    """Guess whether the given salary is hourly, daily or pa"""
    if salary < 50:
        return "per hour"
    elif salary < 10000:
        return "per day"
    else:
        return "per annum"


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
    """Returns a function which extracts list of numbers from a raw salary"""
    re_ = re.compile(regex)
    return lambda raw_salary: [
        float(result) for result in re_.findall(raw_salary) if len(result) > 0
    ]


def save_to_s3(filename, contents):
    """Saves the contents to the filename in {BUCKET_NAME}/{S3_PATH}"""
    s3 = boto3.resource("s3")
    obj = s3.Object(BUCKET_NAME, S3_PATH.format(filename))
    obj.put(Body=contents)


def load_from_s3(filename):
    """Loads the file contents from the filename at {BUCKET_NAME}/{S3_PATH}"""
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=S3_PATH.format(filename))
    return obj["Body"].read().decode()


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


# %%
save_model("(\d*[.]?\d*)")  # <--- After all of my hard work, I'll save my model config


# %%
def load_jobs(limit=100):
    with db_session("dev") as session:
        for ad in session.query(RawJobAd).limit(limit):
            yield object_as_dict(ad)


# %%
# Example of applying my model
fields_to_print = ("job_title_raw", "contract_type_raw", "job_salary_raw")
for job_ad in load_jobs():
    prediction = apply_model(job_ad)
    print(*(job_ad[x] for x in fields_to_print), prediction, sep="\n")
    print()

# %%
