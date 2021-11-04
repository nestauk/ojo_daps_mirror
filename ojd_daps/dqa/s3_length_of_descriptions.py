# -*- coding: utf-8 -*-
# %% [markdown]
# # The percentage of job adverts with descriptions and the approximate word count of descriptions, in S3, by week of job posting
# https://github.com/nestauk/ojd_daps/issues/119
#
# Author: Cath S
#
# ### Summary
# * It's really slow to read the raw data from S3. Only a random sample of 0.5% of the S3 files is used
# * It's only in the first few months that adverts were missing descriptions.
# * The reason that adverts from the 7th week of 2021 look long is due to the release of many (long) adverts for Census Interviewers.
# * Some of the very short descriptions are due to a fault in Reed's website that cuts a description off in mid sentence. Quite a few of the cut-off ads were from the same employer - NHS Business Services Authority.
#
# ### Plotting code refactored to:
#
# * ```dqa.plots.s3_counts_description_weekly``` (line chart)
# * ```dqa.plots.s3_counts_description_length_weekly``` (box plot)
#
# ## Preamble (imports, globals, utils)
# *these should be useful for other dqa*

# %%
# imports
# %matplotlib inline
from matplotlib import pyplot as plt
from datetime import datetime
import calendar
from collections import defaultdict
import re
from functools import lru_cache
import string
from ojd_daps.dqa.data_getters import get_s3_job_ads
from ojd_daps.dqa.s3_counts_utils import find_category
from ojd_daps.dqa.s3_counts_utils import find_when_posted
from ojd_daps.dqa.s3_counts_utils import get_job_ads_posted_date
from ojd_daps.dqa.s3_counts_utils import timestamp_to_isoweek
from ojd_daps.dqa.s3_counts_utils import find_description


# globals
DATE_FINDER = re.compile(br"jobPostedDate: '(.*)/(.*)/(.*)'")
CATEGORY_FINDER = re.compile(b"pageCategory: '(.*)'")
DESCRIPTION_FINDER = re.compile("<span itemprop='description'>(.*?)</span>")
SAMPLE_RATIO = 0.005
JOB_BOARD = "reed"
# The maximum number of previous weeks to show in the plot
MAX_WEEKS_SHOW = 15


def count_approx_words(description):
    """
    Counts the approximate number of words in the
    description of one job advert.
    """
    # Remove html tags from description
    descrip_stripped = re.compile(r"<.*?>").sub("", description)

    # Remove punctuation from description
    descrip_no_punc = descrip_stripped.translate(
        str.maketrans("", "", string.punctuation)
    )

    # Split string into words and count number of words
    approx_no_words = len(descrip_no_punc.split())

    return approx_no_words


# %% [markdown]
# ## Collect a sample of adverts

# %% [markdown]
# Randomly sample 0.5% of the data from S3

# %%
_, job_ads = get_job_ads_posted_date(
    job_board=JOB_BOARD, sample_ratio=SAMPLE_RATIO, get_body=True
)

# %%
print("Number of adverts collected: {}".format(len(job_ads)))

# %% [markdown]
# ## Calculate the number of words in each advert and its week

# %%
# Extract the week that the advert was posted,
# record if the advert is missing a description,
# and count the approximate number of words in
# the advert's description if it is present

# To capture results
descriptions_by_week = defaultdict(
    lambda: {"total": 0, "missing": 0, "approx_no_of_words": []}
)

# Loop over job adverts
for index, one_advert in enumerate(job_ads):

    # Extract the isoweek for the date when the advert was posted
    timestamp = find_when_posted(one_advert["body"])
    isoweek = timestamp_to_isoweek(timestamp)

    # Add to total count
    descriptions_by_week[isoweek]["total"] += 1

    # Extract the description of the advert
    # (not all adverts contain descriptions)
    description = find_description(one_advert["body"])
    if description is None:
        descriptions_by_week[isoweek]["missing"] += 1
    else:
        # Count the approximate number of words in the advert
        count = count_approx_words(description)
        descriptions_by_week[isoweek]["approx_no_of_words"].append(count)


# %%
## Percentage of adverts with a description

# By week
for isoweek, data in descriptions_by_week.items():
    total_adverts = data["total"]
    total_with_description = len(data["approx_no_of_words"])
    data["percent_with_description"] = 100 * total_with_description / total_adverts

# Over all weeks
total_adverts = sum([data["total"] for data in descriptions_by_week.values()])
total_with_description = sum(
    [data["total"] - data["missing"] for data in descriptions_by_week.values()]
)
total_percent_with_description = 100 * total_with_description / total_adverts
print(
    "Percentage of adverts with a description is {0:.0f} percent".format(
        total_percent_with_description
    )
)


# %% [markdown]
# ## Organise data for the plots

# %%
# Space to hold plot data
plot_data = defaultdict(list)

# Record total number of adverts with descriptions (for the plot's title)
plot_data["total_adverts"] = total_adverts

# Record percentage of adverts with descriptions (also for the title)
plot_data["total_percent_with_description"] = total_percent_with_description

# %%
# List of unique isoweeks (for x axis)
plot_data["isoweeks"] = sorted(set(descriptions_by_week.keys()))

# %%
print("Number of unique weeks is {}".format(len(plot_data["isoweeks"])))

# %%
# For the line chart
plot_data["percent_with_description"] = [
    data["percent_with_description"] for data in descriptions_by_week.values()
]

# For the box plot
plot_data["approx_no_of_words"] = [
    data["approx_no_of_words"] for data in descriptions_by_week.values()
]


# %% [markdown]
# ## Plots

# %% [markdown]
# ### Line plot for percentage of adverts with a description

# %%
fig, ax = plt.subplots(figsize=(20, 6))
ax.set_title(
    (
        "Percentage of adverts with \n descriptions "
        "by week posted \n (overall: {0:.0f}% "
        "based on {1} adverts)".format(
            plot_data["total_percent_with_description"], plot_data["total_adverts"]
        )
    ),
    fontsize=20,
)
ax.plot(plot_data["percent_with_description"][-MAX_WEEKS_SHOW:])
ax.set_xticks([value for value in range(0, MAX_WEEKS_SHOW)])
ax.set_xticklabels(plot_data["isoweeks"][-MAX_WEEKS_SHOW:])
_ = ax.set_xlabel("(Year, week number)", fontsize=16, labelpad=20)
ax.tick_params(axis="both", which="major", labelsize=14)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_visible(False)

plt.show()

# %% [markdown]
# ### Boxplot for length of advert

# %%
# Draw a series of box plots (one per week)
fig, ax = plt.subplots(figsize=(20, 6))
ax.set_title(
    (
        "Approximate number of words in advert descriptions "
        " \n by week posted, based on {0} adverts".format(plot_data["total_adverts"])
    ),
    fontsize=20,
)
ax.boxplot(plot_data["approx_no_of_words"][-MAX_WEEKS_SHOW:])
ax.set_xticklabels(plot_data["isoweeks"][-MAX_WEEKS_SHOW:])
_ = ax.set_xlabel("(Year, week number)", fontsize=16, labelpad=20)
ax.tick_params(axis="both", which="major", labelsize=14)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_visible(False)

plt.show()
