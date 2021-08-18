import boto3

S3_PATH = "labs/salaries/regex/{}"
BUCKET_NAME = "open-jobs-lake"


def guess_rate(salary):
    """Guess whether the given salary is hourly, daily or pa"""
    if salary < 50:
        return "per hour"
    elif salary < 10000:
        return "per day"
    else:
        return "per annum"


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
