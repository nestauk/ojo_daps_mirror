# %%
"""
s3 counts utils
---------------

General utils for making plots of counts with the 'raw' S3 data.
This has been factored out from s3_counts.py
"""

# %%
import re
from datetime import datetime
from functools import lru_cache

# %%
from ojd_daps.dqa.data_getters import get_s3_job_ads

# %%
# globals
DATE_FINDER = re.compile(br"jobPostedDate: '(.*)/(.*)/(.*)'")
CATEGORY_FINDER = re.compile(b"pageCategory: '(.*)'")
DESCRIPTION_FINDER = re.compile('<span itemprop="description">(.*?)</span>')
SAMPLE_RATIO = 0.01
JOB_BOARD = "reed"


# %%
def find_category(body):
    """
    Extract the 'pageCategory' field from the raw HTML

    Some job ads don't have data in them, so I wanted to understand
    why this was. On inspection if 'pageCategory != jobseeker-jobdetails-mobile'
    then the procedure fails, and so I collect the pageCategory for
    understanding if there are many of these edge cases in our data.
    """
    try:
        (cat,) = CATEGORY_FINDER.search(body).groups()
    except:
        cat = None
    return cat


# %%
def find_when_posted(body):
    """
    Extract the 'jobPostedDate' field from the raw HTML and convert to datetime.
    """
    (dd, mm, yyyy) = DATE_FINDER.search(body).groups()
    posted = datetime(day=int(dd), month=int(mm), year=int(yyyy))
    return posted


# %%
@lru_cache()
def get_job_ads_posted_date(job_board, sample_ratio, n_max=None, get_body=False):
    """
    Sample jobs from S3 and extract the date on which they were posted,
    and also the "category" of the HTML page. Note that this procedure is very
    "reed" specific, but could be generalised in the future.
    """
    categories = []
    job_ads = []
    for job_ad in get_s3_job_ads(job_board, sample_ratio=sample_ratio):
        if get_body:
            body = job_ad["body"]
        else:
            body = job_ad.pop("body")  # Don't keep a hold of the memory intensive body
        cat = find_category(body)  # Ignore categories which aren't 'j*mobile'
        categories.append(cat)
        if cat != b"jobseeker-jobdetails-mobile":
            continue
        # Determine the posted date then continue
        job_ad["posted"] = find_when_posted(body)
        job_ads.append(job_ad)
        if len(job_ads) == n_max:
            break
    return categories, job_ads


# %%
def timestamp_to_isoweek(ts):
    """Convert a datetime object to tuple (year, week number this year)"""
    year, week, _ = ts.isocalendar()
    return (year, week)


# %%
def isoweek_to_universal_week(isoweek):
    """Convert (year, week number this year) to week number since year zero"""
    year, week = isoweek
    return (year - 1) * 53 + week


# %%
def timestamp_to_universal_week(ts):
    """Convert timestamp to week number since year zero"""
    isoweek = timestamp_to_isoweek(ts)
    return isoweek_to_universal_week(isoweek)


# %%
def timestamp_to_isomonth(ts):
    """Convert timestamp to (month, year)"""
    return (ts.month, ts.year)


# %%
def isomonth_to_universal_month(isomonth):
    """Convert (month, year) to months since year zero"""
    month, year = isomonth
    return (year - 1) * 12 + month


# %%
def timestamp_to_universal_month(ts):
    """Convert timestamp to months since year zero"""
    isomonth = timestamp_to_isomonth(ts)
    return isomonth_to_universal_month(isomonth)


# %%
def find_description(body):
    """
    Extract the 'description' field from the raw HTML.
    Not all adverts contain a description.
    """
    decoded = body.decode()
    try:
        return re.findall(DESCRIPTION_FINDER, decoded)[0]
    except:
        return None
