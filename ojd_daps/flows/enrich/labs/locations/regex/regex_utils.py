"""
salaries.regex
--------------
Regex to pull out locations, which attempts to extract a
postcode outcode, then if not found, tidies up the raw
location for place name matching.
Do `apply_model(row)` to load and apply the model to data.
Run `regex.py` in `jupyter lab` (with `jupytext` installed)
to see the model in action.
"""
from functools import lru_cache
import boto3
import re


S3_PATH = "labs/locations/regex/{}"
BUCKET_NAME = "open-jobs-lake"


def regex_model(outcode_regex, boilerplate_text):
    """Produces function which extracts a postcode area from a location
    or a cleaned location if none found.
    """
    re_ = re.compile(outcode_regex)
    clean_ = boilerplate_text
    return (
        lambda raw_location: raw_location.split(",")[0]
        .replace(clean_, "")
        .lower()
        .replace(" ", "_")
        .lstrip("_")
        .rstrip("_")
        if len(re_.findall(raw_location)) == 0
        else re_.findall(raw_location)[0].replace(" ", "")
    )


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


def save_model(outcode_regex, boilerplate_text):
    """Save the model config to s3.
    Args:
        outcode_regex (str): The regex strategy for postcode area
        boilerplate_text (str): The regex strategy for basic cleaning
    """
    save_to_s3("outcode_regex.txt", outcode_regex)
    save_to_s3("boilerplate_text.txt", boilerplate_text)


@lru_cache()  # <--- Important
def load_model():
    """Loads the model"""
    outcode_regex = load_from_s3("outcode_regex.txt")
    boilerplate_text = load_from_s3("boilerplate_text.txt")
    return regex_model(outcode_regex, boilerplate_text)


def apply_model(row):
    """Loads and applies the model to the given row"""
    model = load_model()  # NB: lru_cached
    location = model(row["job_location_raw"])
    return location
