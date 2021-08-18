import pytest

from bs4 import BeautifulSoup
from datetime import datetime
from pathlib import Path

from ojd_daps.flows.extract.indeed import get_indeed_details, indeed_detail_parser,\
    regex_search


PATH = Path(__file__).parent
EXAMPLE_QUERY = PATH / "resources" / "example_indeed_ad.html"
AD_CONTENT = open(EXAMPLE_QUERY, "r").read()
EXPECTED_DETAILS = {
                "id":"837b8affdf31b9e7",
                "data_source":"Indeed",
                "created":str(datetime.today().strftime('%Y-%m-%d'))
                }


def test_get_indeed_details():
    ad_dict = get_indeed_details(AD_CONTENT)
    assert ad_dict['id'] == EXPECTED_DETAILS['id']
    assert ad_dict['data_source'] == EXPECTED_DETAILS['data_source']
    assert ad_dict['created'] == EXPECTED_DETAILS['created']

def test_indeed_detail_parser():
    soup = BeautifulSoup(AD_CONTENT, "lxml")
    assert indeed_detail_parser(soup, 'span', 'icl-u-xs-mr--xs')\
        == '£9 an hour'
    assert indeed_detail_parser(soup, 'span', 'indeed-apply-widget',\
                                'data-indeed-apply-joburl')\
        == 'https://www.indeed.co.uk/viewjob?jk=837b8affdf31b9e7'

def test_regex_search():
    assert regex_search('"jobTitle":(.*?),', AD_CONTENT) == "Warehouse Labourer"
