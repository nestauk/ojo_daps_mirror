# +
"""
Cumulative s3 counts by month
-----------------------------

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

TEST_SAMPLE_RATIO = SAMPLE_RATIO / 100


def make_plot_data(test=False):
    """Makes the plot data for this module.

    Args:
        test (bool): Reduces the sample ratio size to make the function run in less than a few mins.
    Returns:
        plot_data (list of dict): Data ready for plotting in `make_plot`
    """
    sample_ratio = TEST_SAMPLE_RATIO if test else SAMPLE_RATIO

    _, job_ads = get_job_ads_posted_date(job_board=JOB_BOARD, sample_ratio=sample_ratio)

    # Process the data in generators
    posted_dates = (job_ad["posted"] for job_ad in job_ads)
    sorted_dates = sorted(posted_dates, key=timestamp_to_universal_month)
    grouped_dates = groupby(sorted_dates, key=timestamp_to_isomonth)

    # Execute the generators
    plot_data = defaultdict(list)
    total_count = 0
    for isomonth, chunk in grouped_dates:
        total_count += len((*chunk,)) / sample_ratio
        plot_data["label"].append(isomonth)
        plot_data["numeric"].append(isomonth_to_universal_month(isomonth))
        plot_data["count"].append(total_count)
    return plot_data


def make_plot(plot_data, test=False):
    """Plots the plot data for this module.

    Args:
        plot_data (list of dict): Data ready for plotting.
        test (bool): Reduces the sample ratio size to make the function run in less than a few mins.
    Returns:
        ax (matplotlib.Ax): Matplotlib ax object, containing a plot.
    """
    sample_ratio = TEST_SAMPLE_RATIO if test else SAMPLE_RATIO

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
        f"(estimated from {int(100*sample_ratio)}% sampling)",
        fontsize=20,
        pad=20,
    )
    ax.set_xlabel("(Month, Year)", fontsize=20, labelpad=20)
    return ax


# -

if __name__ == "__main__":
    # %matplotlib inline
    plot_data = make_plot_data(test=True)
    ax = make_plot(plot_data=plot_data, test=True)
