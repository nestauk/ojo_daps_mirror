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

import lxml
import lxml.html
from bs4 import BeautifulSoup
from daps_utils import talk_to_luigi

from metaflow import S3, FlowSpec, Parameter, batch, step, retry

S3_PATH = "s3://open-jobs-lake/most_recent_jobs/{level}/reed/"
CHUNKSIZE = 1000
CAP = 100


def get_reed_details(ad):
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
    soup = BeautifulSoup(ad, "lxml")
    try:
        data_text = str(soup)
        job_details = {
            "id": reed_detail_parser("jobId", data_text),
            "data_source": "Reed",
            "created": reed_detail_parser("jobPostedDate", data_text),
            "job_title_raw": reed_detail_parser("jobTitle", data_text),
            "job_location_raw": reed_detail_parser("jobLocation", data_text),
            "job_salary_raw": regex_search("jobSalaryBand: (.*?),", str(soup)),
            "company_raw": reed_detail_parser("jobRecruiterName", data_text),
            "contract_type_raw": reed_detail_parser("jobType", data_text),
            "description": strip_html(
                str(soup.find_all("span", itemprop="description"))
            ),
            "closing_date_raw": None,  # Reed don't include closing dates
            "type": reed_detail_parser("jobRecruiterType", data_text),
            "sector": reed_detail_parser("jobSector", data_text),
            "parent_sector": reed_detail_parser("jobParentSector", data_text),
            "knowledge_domain": reed_detail_parser("jobKnowledgeDomain", data_text),
            "occupation": reed_detail_parser("jobOccupationL3", data_text),
        }
        return job_details
    except IndexError:
        pass


def regex_search(regex, text):
    """Searches the text and extracts if the regex is found.

    Parameters
    ----------
    regex : str
        Regular expression to be found
    text : str
        Text to be searched

    Returns
    -------
    value : str
        Value if search successful, None if not
    """
    try:
        value = re.search(regex, text).group(1).replace("'", "")
    except:
        value = None
    return value


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


@talk_to_luigi
class ReedAdCurateFlow(FlowSpec):
    production = Parameter("production", help="Run in production mode?", default=False)
    job_board = Parameter("job_board", help="Which job board?", default="reed")

    @property
    def test(self):
        return not self.production

    @property
    def s3_path(self):
        level = "test" if self.test else "production"
        return S3_PATH.format(level=level)

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
        with S3(s3root=self.s3_path) as s3:
            self.job_ads = [item.key for item in s3.list_recursive()]
        self.next(self.generate_ad_chunks)

    @step
    def generate_ad_chunks(self):
        """
        Convert the ads into chunks of 1000.
        """
        from common import get_chunks

        limit = min(CAP, len(self.job_ads)) if self.test else None
        self.job_ads = self.job_ads[:limit]
        self.chunks = get_chunks(self.job_ads, CHUNKSIZE)
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
        for ad in self.input:
            with S3(s3root=self.s3_path) as s3:
                job_ad = s3.get(ad).text
            try:
                reed_details = get_reed_details(job_ad)
            except TypeError:
                pass
            reed_details["s3_location"] = ad
            ad_details.append(reed_details)
            if first_id is None:
                first_id = reed_details["id"]
            last_id = reed_details["id"]
        # Save to S3
        ids = f"{first_id}-{last_id}"
        filename = f"extract-{self.job_board}_test-{self.test}_{ids}.json"
        with S3(run=self) as s3:
            data = json.dumps(ad_details)
            url = s3.put(filename, data)
        self.next(self.join_data)

    @step
    def join_data(self, inputs):
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    ReedAdCurateFlow()
