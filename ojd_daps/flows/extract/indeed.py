"""
Flow to extract raw job advert information from indeed ad
html files.
"""

import os

os.system(
    f"pip install -r {os.path.dirname(os.path.realpath(__file__))}/requirements.txt 1> /dev/null"
)
import json
import lxml
import re

from bs4 import BeautifulSoup
from datetime import datetime
from metaflow import FlowSpec, step, S3, resources, Parameter, batch

from daps_utils import talk_to_luigi

S3_PATH = "s3://open-jobs-lake/most_recent_jobs/{level}/indeed/"
CHUNKSIZE = 1000
CAP = 100


def get_indeed_details(ad):
    """Parses details from a indeed job advert.

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
    text = str(soup).encode("utf8").decode("unicode_escape", errors="ignore")
    job_details = {
        "id": regex_search('"jobKey":(.*?),', text),  # .group(1).replace('"', ""),
        "data_source": "Indeed",
        "url": indeed_detail_parser(
            soup, "span", "indeed-apply-widget", "data-indeed-apply-joburl"
        ),
        "created": str(datetime.today().strftime("%Y-%m-%d")),
        "job_title_raw": regex_search('"jobTitle":(.*?),', text),
        "job_location_raw": regex_search('"jobLocation":(.*?),', text),
        "company_raw": regex_search('ompanyName":(.*?),', text),
        "contract_type_raw": None,  # Indeed don't include structured contract type
        "description": indeed_detail_parser(
            soup, "div", "jobsearch-jobDescriptionText"
        ),
        "closing_date_raw": None,  # Indeed don't include structured closing dates
        "job_salary_raw": regex_search('"salaryText":(.*?),', text),
    }
    return job_details


def indeed_detail_parser(soup, tag, name, element=None):
    """Parses details from a indeed job advert.

    Parameters
    ----------
    soup : bs4.BeautifulSoup
        Soup class of advert
    tag : str
        Type of HTML tag where the value resides
    name : str
        Name of the tag
    element_bool : Boolean
        Whether the desired value is contained within an element
    element : str
        Name of the element if required

    Returns:
    ----------
    value : str
        If present, the value, empty string if not
    """
    try:
        item = soup.find(tag, {"class": name})
    except (AttributeError, TypeError, KeyError):
        value = None
    try:
        value = item.text if element is None else item[element]
    except (AttributeError, TypeError, KeyError):
        value = None
    return value


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
        value = re.search(regex, text).group(1).replace('"', "")
    except:
        value = None
    return value


@talk_to_luigi
class IndeedAdCurateFlow(FlowSpec):
    production = Parameter("production", help="Run in production mode?", default=False)
    job_board = Parameter("job_board", help="Which job board?", default="indeed")

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

    @step
    def get_job_ads(self):
        """
        Define the ads to be processed.
        """
        with S3(s3root=self.s3_path) as s3:
            self.job_ads = []
            for key in s3.list_recursive():
                self.job_ads.append(key.key)
        self.next(self.generate_ad_chunks)

    @step
    def generate_ad_chunks(self):
        """
        Convert the ads into chunks of 1000.
        """
        from common import get_chunks

        limit = min(CAP, len(self.job_ads)) if self.test else None
        self.chunks = get_chunks(self.job_ads, CHUNKSIZE)
        self.next(self.extract_ad_details, foreach="chunks")

    @step
    def extract_ad_details(self):
        """
        Process each chunk to extract relevant details.
        """
        self.ad_details = []
        for ad in self.input:
            with S3(s3root=self.s3) as s3:
                job_ad = s3.get(ad).text
                indeed_details = get_indeed_details(job_ad)
                indeed_details["s3_location"] = ad
                self.ad_details.append(indeed_details)
        self.next(self.join_data)

    @step
    def join_data(self, inputs):
        """
        Join the outputs of the processing.
        """
        from common import flatten

        self.data = flatten(input.ad_details for input in inputs)
        self.next(self.end)

    @step
    def end(self):
        """
        Save the data to the data lake.
        """
        filename = f"extract-{self.job_board}_test-{self.test}.json"
        with S3(run=self) as s3:
            data = json.dumps(self.data)
            url = s3.put(filename, data)


if __name__ == "__main__":
    IndeedAdCurateFlow()
