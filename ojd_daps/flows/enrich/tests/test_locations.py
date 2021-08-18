from unittest import mock

from ojd_daps.flows.enrich.locations import (
    process_location,
    define_processed_location,
    location_lookup,
)

OUTCODE_LOCATION = "Sheffield, S1"
LOCATION = "Telford, Shropshire"


def test_process_location():
    assert process_location(LOCATION) == "telford"


def test_define_processed_location():
    assert define_processed_location(LOCATION) == "telford"
    assert define_processed_location(OUTCODE_LOCATION) == "S1"


@mock.patch("ojd_daps.flows.enrich.locations.db_session")
def test_location_lookup(mocked_session):
    data = [("code1", "name1"), ("code2", "name1"), ("code3", "name2")]
    mocked_session().__enter__().query().all.return_value = data
    assert dict(location_lookup()) == {"name1": ["code1", "code2"], "name2": ["code3"]}
