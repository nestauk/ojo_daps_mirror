import pytest
import datetime

from ojd_daps.flows.enrich.deduplication import sliding_window_getter

TEST_JOB_ADS = [
    {'id': '111', 'created': datetime.datetime(2020, 12, 16, 0, 0)},
    {'id': '112', 'created': datetime.datetime(2020, 10, 16, 0, 0)}
]
EXPECTED_WINDOWS = [
    [datetime.datetime(2020, 10, 16, 0, 0),
    datetime.datetime(2020, 12, 13, 0, 0)],
    [datetime.datetime(2020, 11, 14, 0, 0),
    datetime.datetime(2021, 1, 11, 0, 0)],
    [datetime.datetime(2020, 12, 13, 0, 0),
    datetime.datetime(2021, 2, 9, 0, 0)]
]

def test_sliding_window_getter():
    assert sliding_window_getter(job_ads=TEST_JOB_ADS, interval=58) == \
        EXPECTED_WINDOWS