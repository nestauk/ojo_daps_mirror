# %%
"""
s3 counts by weekday, by month
-------------------------------

Counts of files in the 'most_recent_jobs' S3 bucket, grouped
by the "job posted" date in the main body of the HTML.

Note that the data is only sampled (see SAMPLE_RATIO) as
the data processing rate from S3 is too slow without e.g. Spark
"""

from ojd_daps.dqa.s3_counts_utils import get_job_ads_posted_date  # lru_cached
from ojd_daps.dqa.s3_counts_utils import JOB_BOARD, SAMPLE_RATIO
from ojd_daps.dqa.s3_counts_utils import (
    timestamp_to_universal_month,
    isomonth_to_universal_month,
    timestamp_to_isomonth,
)
from itertools import groupby
from collections import defaultdict
from matplotlib import pyplot as plt
import calendar

TEST_SAMPLE_RATIO = SAMPLE_RATIO / 100


def make_plot_data(test=False):
    """Makes the plot data for this module.

    Args:
        test (bool): Reduces the sample ratio size to make the function run in less than a few mins.
    Returns:
        weekday_data (dict of dict of list): Data ready for plotting in `make_plot`, pre-grouped by weekday.
    """
    sample_ratio = TEST_SAMPLE_RATIO if test else SAMPLE_RATIO

    _, job_ads = get_job_ads_posted_date(job_board=JOB_BOARD, sample_ratio=sample_ratio)

    # Process the data in generators
    posted_dates = (job_ad["posted"] for job_ad in job_ads)
    sorted_dates = sorted(posted_dates, key=lambda ts: ts.weekday())
    grouped_dates = groupby(sorted_dates, key=lambda ts: ts.weekday())

    # Partially execute the generators
    weekday_data = defaultdict(lambda: defaultdict(list))
    for weekday, weekday_chunk in grouped_dates:
        weekday_name = calendar.day_abbr[weekday]
        # A second group by, more generators
        sorted_chunk = sorted(
            weekday_chunk, key=timestamp_to_universal_month, reverse=True
        )
        grouped_chunk = groupby(sorted_chunk, key=timestamp_to_isomonth)
        # Launch the full set of generators to fill the plotting data
        for isomonth, month_chunk in grouped_chunk:
            weekday_data[weekday_name]["label"].append(isomonth)
            weekday_data[weekday_name]["count"].append(
                len((*month_chunk,)) / sample_ratio
            )
            weekday_data[weekday_name]["numeric"].append(
                isomonth_to_universal_month(isomonth)
            )
    return weekday_data


def make_plot(weekday_data, test=False):
    """Plots the plot data for this module.

    Args:
        weekday_data (dict of dict of list): Data ready for plotting in `make_plot`, pre-grouped by weekday.
        test (bool): Reduces the sample ratio size to make the function run in less than a few mins.
    Returns:
        ax (matplotlib.Ax): Matplotlib ax object, containing a plot.
    """
    sample_ratio = TEST_SAMPLE_RATIO if test else SAMPLE_RATIO

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
        f"(estimated from {int(100*sample_ratio)}% sampling)",
        fontsize=20,
        pad=20,
    )
    ax.set_xlabel("(Month, Year)", fontsize=20, labelpad=20)
    return ax


# %%
if __name__ == "__main__":
    # %matplotlib inline
    weekday_data = make_plot_data(test=True)
    ax = make_plot(weekday_data=weekday_data, test=True)
