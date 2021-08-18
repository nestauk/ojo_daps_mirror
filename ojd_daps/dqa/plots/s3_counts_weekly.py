# %%
"""
s3 counts weekly
----------------

Counts of files in the 'most_recent_jobs' S3 bucket, grouped
by the "job posted" date in the main body of the HTML.

Note that the data is only sampled (see SAMPLE_RATIO) as 
the data processing rate from S3 is too slow without e.g. Spark
"""

from ojd_daps.dqa.s3_counts_utils import get_job_ads_posted_date  # lru_cached
from ojd_daps.dqa.s3_counts_utils import JOB_BOARD, SAMPLE_RATIO
from ojd_daps.dqa.s3_counts_utils import (
    timestamp_to_universal_week,
    isoweek_to_universal_week,
    timestamp_to_isoweek,
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
        plot_data["count"].append(
            len((*chunk,)) / sample_ratio
        )  # Estimated count of files
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
    ax.scatter(x=plot_data["numeric"], y=plot_data["count"])
    ax.set_xticks(plot_data["numeric"])
    ax.set_xticklabels(plot_data["label"])
    ax.set_title(
        f"Number of files in 'most_recent_jobs/production/{JOB_BOARD}/'\n"
        f"(estimated from {int(100*sample_ratio)}% sampling)",
        fontsize=20,
        pad=20,
    )
    ax.set_xlabel("(Week number, Year)", fontsize=20, labelpad=20)
    return ax


# %%
if __name__ == "__main__":
    # %matplotlib inline
    plot_data = make_plot_data(test=True)
    ax = make_plot(plot_data=plot_data, test=True)

