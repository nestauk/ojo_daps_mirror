import pytest
from unittest import mock
from ojd_daps.flows.aggregate.common import monday_of_week, iterdates, get_snapshot_ads
from ojd_daps.flows.aggregate.common import datetime


def strftime(date):
    return datetime.strftime(date, "%d-%m-%Y")


@pytest.mark.parametrize(
    "date,monday",
    [
        ("07-11-2021", "01-11-2021"),
        ("08-11-2021", "08-11-2021"),  # A Monday
        ("09-11-2021", "08-11-2021"),
        ("10-11-2021", "08-11-2021"),
        ("11-11-2021", "08-11-2021"),
        ("12-11-2021", "08-11-2021"),
        ("13-11-2021", "08-11-2021"),
        ("14-11-2021", "08-11-2021"),
        ("15-11-2021", "15-11-2021"),
        ("16-11-2021", "15-11-2021"),
        ("17-11-2021", "15-11-2021"),
        ("05-01-2012", "02-01-2012"),  # A couple of other dates
        ("17-02-1997", "17-20-1997"),
    ],
)
def test_monday_of_week(date, monday):
    date = strftime(date)
    monday = strftime(monday)
    assert monday.weekday() == 1
    assert monday_of_week(date) == monday


def test_iterdates():
    assert (
        list(
            iterdates(
                start_date=datetime(2021, 2, 1),
                end_date=datetime(2021, 3, 1),
                timespan_weeks=2,
            )
        )
        == []
    )
