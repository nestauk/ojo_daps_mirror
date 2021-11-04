"""s3 and database inferface support functions for requires_degree regex model."""
import boto3

from sqlalchemy.exc import NoInspectionAvailable
from sqlalchemy import inspect
from daps_utils.db import db_session, object_as_dict

# >>> Workaround for batch
try:
    from ojd_daps.orms.raw_jobs import RawJobAd
except ModuleNotFoundError:
    pass

############
# Following lines are needed until issue is fixed in daps_utils
from daps_utils import db

try:
    import ojd_daps
except ModuleNotFoundError:
    ojd_daps = None

db.CALLER_PKG = ojd_daps
db_session = db.db_session
############

S3_PATH = "labs/requires_degree/regex/{}"
BUCKET_NAME = "open-jobs-lake"


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


def load_jobs(limit=10, database="production", columns=None):
    """Return a generator that returns jobs with a *non-empty* job description (the
    model is undefined for jobs without description field."""
    if columns is None:
        # return all
        columns = (RawJobAd,)
    with db_session(database) as session:
        # records with a job description
        for ad in (
            session.query(*columns).filter(RawJobAd.description != "[]").limit(limit)
        ):
            yield object_as_dict(ad)
