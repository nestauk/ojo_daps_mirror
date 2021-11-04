# %% [markdown]
# # Raw counts of files in S3 by date of job posting
# https://github.com/nestauk/ojd_daps/issues/113
#
# Author: Joel K
#
# ### TL;DR
#
# * It's really slow to read the raw data from S3
# * I've implemented a procedure for getting the "job posted" date from raw S3 files
# * Only a random sample of 1% of the S3 files is used
# * With this setup I have estimated the counts of reed files collected per week (around 60k per week)
# * The number of files collected from reed during 2021 was very consistent
# * Expect many fewer job ads posted on weekends
# * The number of job ads collected each month has been pretty constant
#
# ### Issues raised
#
# * ```PySpark``` is needed to analyse the raw data properly (see [issue](https://github.com/nestauk/ojd_daps/issues/117))
# * Add tests to ensure that the ```dqa.plots``` modules comply to the required specifications (see [issue](https://github.com/nestauk/ojd_daps/issues/118))
#
# ### Plotting code refactored to:
#
# * ```dqa.plots.s3_counts_weekly```
# * ```dqa.plots.s3_counts_by_weekday_monthly```
# * ```dqa.plots.s3_counts_cumulative_monthly```
#
# ## Preamble (imports, globals, utils)
# *these should be useful for other dqa*

# %%
# imports
# %matplotlib inline
from matplotlib import pyplot as plt
from datetime import datetime
from itertools import groupby
import calendar
from collections import defaultdict, Counter
import re
from functools import lru_cache

from ojd_daps.dqa.data_getters import get_s3_job_ads

# globals
DATE_FINDER = re.compile(br"jobPostedDate: '(.*)/(.*)/(.*)'")
CATEGORY_FINDER = re.compile(b"pageCategory: '(.*)'")
SAMPLE_RATIO = 0.01
JOB_BOARD = "reed"

# utils
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


def find_when_posted(body):
    """
    Extract the 'jobPostedDate' field from the raw HTML and convert to datetime.
    """
    (dd, mm, yyyy) = DATE_FINDER.search(body).groups()
    posted = datetime(day=int(dd), month=int(mm), year=int(yyyy))
    return posted


# %% [markdown]
# ## Other utils
#
# *these are **probably not** useful for other dqa*

# %%
@lru_cache()
def get_job_ads_posted_date(job_board, sample_ratio, n_max=None):
    """
    Sample jobs from S3 and extract the date on which they were posted,
    and also the "category" of the HTML page. Note that this procedure is very
    "reed" specific, but could be generalised in the future.
    """
    categories = []
    job_ads = []
    for job_ad in get_s3_job_ads(job_board, sample_ratio=sample_ratio):
        body = job_ad["body"]
        cat = find_category(body)
        categories.append(cat)
        if cat != b"jobseeker-jobdetails-mobile":
            continue
        job_ad["posted"] = find_when_posted(body)
        job_ads.append(job_ad)
        if len(job_ads) == n_max:
            break
    return categories, job_ads


def timestamp_to_isoweek(ts):
    """Convert a datetime object to tuple (year, week number this year)"""
    year, week, _ = ts.isocalendar()
    return (year, week)


def isoweek_to_universal_week(isoweek):
    """Convert (year, week number this year) to week number since year zero"""
    year, week = isoweek
    return (year - 1) * 53 + week


def timestamp_to_universal_week(ts):
    """Convert timestamp to week number since year zero"""
    isoweek = timestamp_to_isoweek(ts)
    return isoweek_to_universal_week(isoweek)


def timestamp_to_isomonth(ts):
    """Convert timestamp to (month, year)"""
    return (ts.month, ts.year)


def isomonth_to_universal_month(isomonth):
    """Convert (month, year) to months since year zero"""
    month, year = isomonth
    return (year - 1) * 12 + month


def timestamp_to_universal_month(ts):
    """Convert timestamp to months since year zero"""
    isomonth = timestamp_to_isomonth(ts)
    return isomonth_to_universal_month(isomonth)


# %% [markdown]
# ## Main DQA (no fixed format from this point)
#
# Count total number of files without reading the body, since reading the body is really slow

# %%
total_files = sum(1 for _ in get_s3_job_ads(JOB_BOARD, read_body=False))

# %%
print(total_files)

# %% [markdown]
# Randomly sample 1% of the data from S3 to speed things up

# %%
categories, job_ads = get_job_ads_posted_date(
    job_board=JOB_BOARD, sample_ratio=SAMPLE_RATIO, n_max=10
)

# %%
job_ads[0]["body"].decode()

# %% [markdown]
# ### Analyse the outputs

# %%
Counter(categories).most_common()

# %% [markdown]
# ### Count of job ads by week number

# %%
# Process the data in generators
posted_dates = (job_ad["posted"] for job_ad in job_ads)
sorted_dates = sorted(posted_dates, key=timestamp_to_universal_week, reverse=True)
grouped_dates = groupby(sorted_dates, key=lambda ts: timestamp_to_isoweek(ts))

# Launch the generators and fill the plotting data
plot_data = defaultdict(list)
for isoweek, chunk in grouped_dates:
    plot_data["label"].append(
        tuple(reversed(isoweek))
    )  # Human-readable iso week label (week, year - rather than - year, week)
    plot_data["numeric"].append(
        isoweek_to_universal_week(isoweek)
    )  # Weeks since year zero
    plot_data["count"].append(len((*chunk,)) / SAMPLE_RATIO)  # Estimated count of files

# %%
fig, ax = plt.subplots(figsize=(20, 6))
ax.scatter(x=plot_data["numeric"], y=plot_data["count"])
ax.set_xticks(plot_data["numeric"])
ax.set_xticklabels(plot_data["label"])
ax.set_title(
    "Number of files in 'most_recent_jobs/production/reed/'\n"
    f"(estimated from {int(100*SAMPLE_RATIO)}% sampling)",
    fontsize=20,
    pad=20,
)
_ = ax.set_xlabel("(Week number, Year)", fontsize=20, labelpad=20)

# %% [markdown]
# ### Total counts by weekday, by month

# %%
# Process the data in generators
posted_dates = (job_ad["posted"] for job_ad in job_ads)
sorted_dates = sorted(posted_dates, key=lambda ts: ts.weekday())
grouped_dates = groupby(sorted_dates, key=lambda ts: ts.weekday())

# Partially execute the generators
weekday_data = defaultdict(lambda: defaultdict(list))
for weekday, weekday_chunk in grouped_dates:
    weekday_name = calendar.day_abbr[weekday]
    # A second group by, more generators
    sorted_chunk = sorted(weekday_chunk, key=timestamp_to_universal_month, reverse=True)
    grouped_chunk = groupby(sorted_chunk, key=timestamp_to_isomonth)
    # Launch the full set of generators to fill the plotting data
    for isomonth, month_chunk in grouped_chunk:
        weekday_data[weekday_name]["label"].append(isomonth)
        weekday_data[weekday_name]["count"].append(len((*month_chunk,)) / SAMPLE_RATIO)
        weekday_data[weekday_name]["numeric"].append(
            isomonth_to_universal_month(isomonth)
        )

# %%
fig, ax = plt.subplots(figsize=(20, 6))

# Scatter by weekday
for weekday, plot_data in weekday_data.items():
    ax.scatter(x=plot_data["numeric"], y=plot_data["count"], label=weekday)
    ax.plot(plot_data["numeric"], plot_data["count"], alpha=0.5)
    # Only set up the style first time
    if weekday != "Mon":
        continue
    ax.set_xticks(plot_data["numeric"])
    ax.set_xticklabels(plot_data["label"])

# Other styling
ax.legend(fontsize=15)
ax.set_title(
    "Number of files in 'most_recent_jobs/production/reed/'\n"
    f"(estimated from {int(100*SAMPLE_RATIO)}% sampling)",
    fontsize=20,
    pad=20,
)
_ = ax.set_xlabel("(Month, Year)", fontsize=20, labelpad=20)

# %% [markdown]
# ### Cumulative count, by month

# %%
# Process the data in generators
posted_dates = (job_ad["posted"] for job_ad in job_ads)
sorted_dates = sorted(posted_dates, key=timestamp_to_universal_month)
grouped_dates = groupby(sorted_dates, key=timestamp_to_isomonth)

# Execute the generators
plot_data = defaultdict(list)
total_count = 0
for isomonth, chunk in grouped_dates:
    total_count += len((*chunk,)) / SAMPLE_RATIO
    plot_data["label"].append(isomonth)
    plot_data["numeric"].append(isomonth_to_universal_month(isomonth))
    plot_data["count"].append(total_count)

# %%
fig, ax = plt.subplots(figsize=(20, 6))
number_ypad = 20000
number_xpad = -0.1

ax.scatter(x=plot_data["numeric"], y=plot_data["count"])
ax.plot(plot_data["numeric"], plot_data["count"], alpha=0.5)
for x, y in zip(plot_data["numeric"], plot_data["count"]):
    ax.text(x + number_xpad, y + number_ypad, "{:,}".format(int(y)), fontsize=20)
ax.set_xticks(plot_data["numeric"])
ax.set_xticklabels(plot_data["label"])
y_min, y_max = ax.get_ylim()
ax.set_ylim(y_min, y_max + number_ypad * 2)

# Other styling
ax.set_title(
    "Cumulative files in 'most_recent_jobs/production/reed/'\n"
    f"(estimated from {int(100*SAMPLE_RATIO)}% sampling)",
    fontsize=20,
    pad=20,
)
_ = ax.set_xlabel("(Month, Year)", fontsize=20, labelpad=20)
# %%
