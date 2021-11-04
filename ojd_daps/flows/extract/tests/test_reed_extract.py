import pytest

from pathlib import Path

from ojd_daps.flows.extract.reed import (
    get_reed_details,
    reed_detail_parser,
    strip_html,
    regex_search,
)


PATH = Path(__file__).parent
EXAMPLE_QUERY = PATH / "resources" / "example_reed_ad.html"
EXPECTED_DETAILS = {
    "job_id": "41339969",
    "data_source": "Reed",
    "created": "09/11/2020",
}
TEST_HTML = "jobId: '41339969',\r\n"
TEST_UNSTRIPPED_HTML = '<td><a href="http://www.fakewebsite.com">Unstripped HTML</a>'


def test_get_reed_details():
    f = open(EXAMPLE_QUERY, "r")
    content = f.read()
    ad_dict = get_reed_details(content)
    assert ad_dict["id"] == EXPECTED_DETAILS["job_id"]
    assert ad_dict["data_source"] == EXPECTED_DETAILS["data_source"]
    assert ad_dict["created"] == EXPECTED_DETAILS["created"]


def test_reed_detail_parser():
    assert reed_detail_parser("jobId", TEST_HTML) == "41339969"


def test_strip_html():
    assert strip_html(TEST_UNSTRIPPED_HTML) == "Unstripped HTML"


def test_regex_search():
    f = open(EXAMPLE_QUERY, "r")
    content = f.read()
    assert regex_search("jobSalaryBand: (.*?),", content) == "52000.0000-55000.0000"
