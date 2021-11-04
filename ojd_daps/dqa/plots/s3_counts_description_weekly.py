# %%
"""
s3 counts description weekly
----------------------------

Counts of descriptions in the 'most_recent_jobs' S3 bucket, grouped
by the "job posted" date in the main body of the HTML.

Creates a line chart showing the percentage of adverts with a
description by week.

Note that the data is only sampled (see SAMPLE_RATIO) as
the data processing rate from S3 is too slow without e.g. Spark

For the full commented code see the notebook 's3_length_of_descriptions'
"""

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
TEST_SAMPLE_RATIO = SAMPLE_RATIO / 10


def make_plot_data(test=False):
    """Makes the plot data for this module.

    Args:
        test (bool): Reduces the sample ratio size to make the function run in less than a few mins.
    Returns:
        plot_data (dict): Data ready for plotting in `make_plot`
    """

    sample_ratio = TEST_SAMPLE_RATIO if test else SAMPLE_RATIO
    _, job_ads = get_job_ads_posted_date(
        job_board=JOB_BOARD, sample_ratio=sample_ratio, get_body=True
    )

    descriptions_by_week = defaultdict(lambda: {"total": 0, "missing": 0})
    for index, one_advert in enumerate(job_ads):

        timestamp = find_when_posted(one_advert["body"])
        isoweek = timestamp_to_isoweek(timestamp)

        descriptions_by_week[isoweek]["total"] += 1
        description = find_description(one_advert["body"])
        if description is None:
            descriptions_by_week[isoweek]["missing"] += 1

    for isoweek, data in descriptions_by_week.items():
        data["percent_with_description"] = (
            100 * (data["total"] - data["missing"]) / data["total"]
        )

    total_adverts = sum([data["total"] for data in descriptions_by_week.values()])
    total_with_description = sum(
        [data["total"] - data["missing"] for data in descriptions_by_week.values()]
    )
    total_percent_with_description = 100 * total_with_description / total_adverts

    plot_data = defaultdict(list)
    plot_data["total_adverts"] = total_adverts
    plot_data["total_percent_with_description"] = total_percent_with_description
    plot_data["isoweeks"] = sorted(set(descriptions_by_week.keys()))
    plot_data["percent_with_description"] = [
        data["percent_with_description"] for data in descriptions_by_week.values()
    ]

    return plot_data


def make_plot(plot_data, test=False):
    """Plots the plot data for this module.

    Args:
        plot_data (dict): Data ready for plotting.
        test (bool): Reduces the sample ratio size to make the function run in less than a few mins.
    Returns:
        ax (matplotlib.Ax): Matplotlib ax object, containing a plot.
    """
    sample_ratio = TEST_SAMPLE_RATIO if test else SAMPLE_RATIO

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

    return ax


# %%
if __name__ == "__main__":
    # %matplotlib inline
    plot_data = make_plot_data(test=True)
    ax = make_plot(plot_data=plot_data, test=True)

# %%
