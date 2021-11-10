"""
Flow to extract raw job advert information from reed ad
html files.
"""

import os

os.system(
    f"pip install -r {os.path.dirname(os.path.realpath(__file__))}/requirements.txt 1> /dev/null"
)
import json
import re
import boto3

import lxml
import lxml.html
from bs4 import BeautifulSoup as _BeautifulSoup
from daps_utils import talk_to_luigi, DapsFlowMixin

from metaflow import S3, FlowSpec, Parameter, batch, step, retry

BUCKET = "open-jobs-lake"
PREFIX = "most_recent_jobs/production/reed/"
CHUNKSIZE = 1000
TEST_OFFSET = 800000  # Skip Jan/Feb/March

# Mapping of orm fields to field names in the raw HTML
KEY_MAP = {
    "id": "jobId",
    "created": "jobPostedDate",
    "job_title_raw": "jobTitle",
    "job_location_raw": "jobLocation",
    "company_raw": "jobRecruiterName",
    "contract_type_raw": "jobType",
    "type": "jobRecruiterType",
    "sector": "jobSector",
    "parent_sector": "jobParentSector",
    "knowledge_domain": "jobKnowledgeDomain",
    "occupation": "jobOccupationL3",
}
SALARY_KEY_MAP = {
    "raw_salary": "value",
    "raw_min_salary": "minValue",
    "raw_max_salary": "maxValue",
    "raw_salary_unit": "unitText",
    "raw_salary_currency": "currency",
}


def BeautifulSoup(text):
    """
    Hardcode html parser to lxml. Breaking with funciton naming conventions
    (lower_snake --> UpperCamel) to shadow the original class BeautifulSoup.
    """
    return _BeautifulSoup(text, "lxml")


def get_meta(span, itemprop):
    """Extract 'baz' from <span itemprop='foo'> <meta itemprop='bar' content='baz'>"""
    meta = span.find("meta", itemprop=itemprop)
    if meta is not None:
        meta = meta.get("content")  # None if doesn't exist
    return meta  # None if either <meta> or <meta content=""> don't exist


def get_salary_info(soup):
    """Identify salary info from the <span itemprop='baseSalary'> span."""
    # Determine whether a salary is officially listed
    span = soup.find("span", itemprop="baseSalary")
    raw_span = str(span).lower()
    is_competitive = "competitive" in raw_span
    is_negotiable = "negotiable" in raw_span
    has_salary = not (is_competitive or is_negotiable)
    # Extract salary info
    salary_info = {
        orm_field: get_meta(span, itemprop) if has_salary else None
        for orm_field, itemprop in SALARY_KEY_MAP.items()
    }
    # Set boolean flags
    salary_info["salary_competitive"] = is_competitive
    salary_info["salary_negotiable"] = is_negotiable
    return salary_info


def get_reed_details(text):
    """Parses details from a reed job advert.

    Parameters
    ----------
    ad : str
        Content of a job ad

    Returns:
    ----------
    job_details : dict
        Dictionary of job details
    """
    # job ads without 'baseSalary' are probably legacy listing pages
    # rather than job ads per-se, so ignore them
    if "baseSalary" not in text:
        return None
    # Extract fields from the 'dataLayer' section of the job ad
    job_details = {
        orm_field: reed_detail_parser(reed_field, text)
        for orm_field, reed_field in KEY_MAP.items()
    }
    # Assign other fields
    soup = BeautifulSoup(text)
    description = strip_html(str(soup.find_all("span", itemprop="description")))
    job_details["description"] = description
    job_details["data_source"] = "Reed"
    job_details["closing_date_raw"] = None  # Reed don't include closing dates
    salary_info = get_salary_info(soup)
    job_details.update(salary_info)
    return job_details


def reed_detail_parser(field, text):
    """Parses details from a reed job advert.

    Parameters
    ----------
    field : str
        Variable to be extracted from the advert
    text : str
        Variable containing the job advert data layer

    Returns:
    ----------
    value : str
        If present, the value, empty string if not
    """
    try:
        value = re.search(f"{field}: (.*),", text).group(1).replace("'", "")
    except AttributeError:
        value = None
    return value


def strip_html(text):
    """Strips html from a string.

    Parameters
    ----------
    text : str
        String that may contain html

    Returns:
    ----------
    stripped_html : str
        Text stripped of html
    """
    stripped_html = lxml.html.fromstring(text).text_content() if text else " "
    return stripped_html


def get_keys(max_keys=1000, limit=None, offset=0):
    client = boto3.client("s3")
    kwargs = dict(Bucket=BUCKET, Prefix=PREFIX, MaxKeys=max_keys)
    response = {"NextContinuationToken": None}
    keys = []
    while "NextContinuationToken" in response and (limit is None or len(keys) < limit):
        response = client.list_objects_v2(**kwargs)
        kwargs["ContinuationToken"] = response.get("NextContinuationToken")
        n_keys = response["KeyCount"]
        if offset <= 0:
            keys += [obj["Key"] for obj in response["Contents"]]
        offset -= n_keys
    return keys


@talk_to_luigi
class ReedAdCurateFlow(FlowSpec, DapsFlowMixin):
    @step
    def start(self):
        """
        Starts the flow.
        """
        self.next(self.get_job_ads)

    @retry
    @step
    def get_job_ads(self):
        """
        Define the ads to be processed.
        """
        from common import get_chunks

        limit = None if self.production else 10 * CHUNKSIZE
        offset = 0 if self.production else TEST_OFFSET
        keys = get_keys(limit=limit, offset=offset)
        self.chunks = get_chunks(keys, CHUNKSIZE)
        self.next(self.extract_ad_details, foreach="chunks")

    @batch
    @retry
    @step
    def extract_ad_details(self):
        """
        Process each chunk to extract relevant details.
        """
        ad_details = []
        last_id, first_id = None, None
        for key in self.input:
            with S3(s3root=f"s3://{BUCKET}") as s3:
                job_ad = s3.get(key=key).text
                reed_details = get_reed_details(job_ad)
                if reed_details is None:
                    continue
            reed_details["s3_location"] = key
            ad_details.append(reed_details)
            if first_id is None:
                first_id = reed_details["id"]
            last_id = reed_details["id"]

        # Save to S3
        ids = f"{first_id}-{last_id}"
        filename = f"extract-reed_test-{self.test}_{ids}.json"
        with S3(run=self) as s3:
            if len(ad_details) > 0:
                data = json.dumps(ad_details)
                s3.put(filename, data)
        self.next(self.join_data)

    @step
    def join_data(self, inputs):
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    ReedAdCurateFlow()
