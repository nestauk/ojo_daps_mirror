import pytest

from pathlib import Path

from ojd_daps.flows.extract.reed import (
    get_salary_info,
    get_reed_details,
    reed_detail_parser,
    strip_html,
    get_meta,
)

from ojd_daps.flows.extract.reed import BeautifulSoup


PATH = Path(__file__).parent
EXAMPLE_QUERY = PATH / "resources" / "example_reed_ad.html"
EXPECTED_DETAILS = {
    "job_id": "41339969",
    "data_source": "Reed",
    "created": "09/11/2020",
}
TEST_HTML = "jobId: '41339969',\r\n"
TEST_UNSTRIPPED_HTML = '<td><a href="http://www.fakewebsite.com">Unstripped HTML</a>'


@pytest.fixture
def html_and_soup():
    with open(EXAMPLE_QUERY) as f:
        text = f.read()
    soup = BeautifulSoup(text)
    return text, soup


def test_get_meta():
    span = BeautifulSoup("<span itemprop='foo'> <meta itemprop='bar' content='baz'>")
    assert get_meta(span, itemprop="bar") == "baz"


def test_get_salary_info(html_and_soup):
    _, soup = html_and_soup
    salary_info = get_salary_info(soup)
    assert salary_info == {
        "raw_salary": "52000.0000",
        "raw_min_salary": "52000.0000",
        "raw_max_salary": "55000.0000",
        "raw_salary_unit": "YEAR",
        "raw_salary_currency": "GBP",
        "salary_competitive": False,
        "salary_negotiable": False,
    }


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
