# %%
from unittest import mock
from ojd_daps.dqa.s3_counts_utils import (
    find_category,
    find_when_posted,
    get_job_ads_posted_date,
    timestamp_to_isoweek,
    isoweek_to_universal_week,
    timestamp_to_universal_week,
    timestamp_to_isomonth,
    isomonth_to_universal_month,
    timestamp_to_universal_month,
    datetime,
    find_description,
)


# %%
TIMESTAMP = datetime(year=1988, month=2, day=1)
ISOWEEK = (1988, 5)
ISOMONTH = (2, 1988)
UNIVERSAL_WEEK = 105316  # 53*(1988-1) + 5
UNIVERSAL_MONTH = 23846  # 12*(1988-1) + 2

# %%
def test_find_description():
    body = (
        b"<span itemprop='description'>THIS IS THE DESCRIPTION</span>"
        b"<span itemprop='not description'>THIS IS NOT THE DESCRIPTION</span>"
    )
    assert find_description(body) == "THIS IS THE DESCRIPTION"


# %%
def test_find_category():
    body = (
        b"pageCategory: other stuff pageCategory: "
        b"'THIS IS THE CATEGORY' other stuff pageCategory:"
    )
    assert find_category(body) == b"THIS IS THE CATEGORY"


# %%
def test_find_category_fails():
    body = (
        b"pageCategory: other stuff pageCategory: "
        b"THIS IS THE CATEGORY other stuff pageCategory:"
    )
    assert find_category(body) is None


# %%
def test_find_when_posted():
    body = (
        b"jobPostedDate: other stuff jobPostedDate: "
        b"'01/02/1988' other stuff jobPostedDate:"
    )
    date = find_when_posted(body)
    assert date.day == 1
    assert date.month == 2
    assert date.year == 1988


# %%
PATH = "ojd_daps.dqa.s3_counts_utils.{}"


# %%
@mock.patch(PATH.format("get_s3_job_ads"))
@mock.patch(PATH.format("find_category"))
@mock.patch(PATH.format("find_when_posted"))
def test_get_job_ads_posted_date(mocked_find_when, mocked_find_cat, mocked_get_ads):
    mocked_find_cat.return_value = b"jobseeker-jobdetails-mobile"
    mocked_find_when.return_value = "today"
    mocked_get_ads.return_value = [
        {"body": "something"},
        {"body": "something else"},
    ]

    # No limit
    cats, ads = get_job_ads_posted_date("dummy", 0.1, None)
    assert cats == [b"jobseeker-jobdetails-mobile", b"jobseeker-jobdetails-mobile"]
    assert ads == [{"posted": "today"}, {"posted": "today"}]

    # Limit = 1
    mocked_get_ads.return_value = [
        {"body": "something"},
        {"body": "something else"},
    ]
    cats, ads = get_job_ads_posted_date("dummy", 0.1, n_max=1)
    assert cats == [b"jobseeker-jobdetails-mobile"]
    assert ads == [{"posted": "today"}]


# %%
def test_timestamp_to_isoweek():
    timestamp_to_isoweek(TIMESTAMP) == ISOWEEK


# %%
def test_isoweek_to_universal_week():
    isoweek_to_universal_week(ISOWEEK) == UNIVERSAL_WEEK


# %%
def test_timestamp_to_universal_week():
    timestamp_to_universal_week(TIMESTAMP) == UNIVERSAL_WEEK


# %%
def test_timestamp_to_isomonth():
    timestamp_to_isomonth(TIMESTAMP) == ISOMONTH


# %%
def test_isomonth_to_universal_month():
    isomonth_to_universal_month(ISOMONTH) == UNIVERSAL_MONTH


# %%
def test_timestamp_to_universal_month():
    timestamp_to_universal_month(TIMESTAMP) == UNIVERSAL_MONTH
