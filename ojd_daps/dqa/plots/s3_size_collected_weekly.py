# +
"""
s3 size collected weekly
----------------

Total size of files in the 'most_recent_jobs' S3 bucket, grouped
by the "job posted" date in the main body of the HTML.

Note that the data is only sampled (see SAMPLE_RATIO) as 
the data processing rate from S3 is too slow without e.g. Spark
"""

from ojd_daps.dqa.s3_utils import get_job_ads_posted_date  # lru_cached
from ojd_daps.dqa.s3_utils import JOB_BOARD, SAMPLE_RATIO
from ojd_daps.dqa.s3_utils import timestamp_to_universal_week, isoweek_to_universal_week, timestamp_to_isoweek
from itertools import groupby
from collections import defaultdict
from matplotlib import pyplot as plt

TEST_SAMPLE_RATIO = SAMPLE_RATIO/100

def make_plot_data(test=False):
    """Makes the plot data for this module.
    
    Args:
        test (bool): Reduces the sample ratio size to make the function run in less than a few mins.
    Returns:
        plot_data (list of dict): Data ready for plotting in `make_plot`
    """
    sample_ratio = TEST_SAMPLE_RATIO if test else SAMPLE_RATIO
    
    _, job_ads = get_job_ads_posted_date(job_board=JOB_BOARD, sample_ratio=sample_ratio)
    
    # Generate the plot data
    filesizes = []
    dates = []
    for advert in job_ads:
        filesizes.append(advert['filesize'])
        dates.append(timestamp_to_universal_week(advert['posted']))
        data = list(zip(dates, filesizes))
    plot_data = [(key, sum(j for i, j in group))
    for key, group in groupby(data, key=lambda x: x[0])]
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
    x, y = zip(*plot_data)
    fig, ax = plt.subplots(figsize=(20, 6))
    ax.scatter(x=x, y=y)
    ax.set_title(f"Total size of files in 'most_recent_jobs/production/{JOB_BOARD}/'\n"
                 f"(estimated from {int(100*sample_ratio)}% sampling)", fontsize=20, pad=20)
    ax.set_xlabel("(Week number)", fontsize=20, labelpad=20)
    ax.set_ylabel("(Total Filesize)", fontsize=20, labelpad=20)
    return ax


# -

if __name__ == "__main__":
    # %matplotlib inline
    plot_data = make_plot_data(test=True)
    ax = make_plot(plot_data=plot_data, test=True)


