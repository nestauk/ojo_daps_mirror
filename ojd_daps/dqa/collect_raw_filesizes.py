# # *File sizes of files in S3 by date of job posting*
# *link to the issue [here](https://github.com/nestauk/ojd_daps/issues/116)*
#
# Author: *Jack Vines*
#
# ### TL;DR
#
# * *Bimodal distribution of file sizes*
# * *Over time, distributions are converging to the 2 modes*
# * *Consistent total filesize over the weeks*
#
# ### Issues raised
#
# * *Add any GH issues raised because of this work here*  (see [issue](https://github.com/nestauk/ojd_daps/issues/117))
#
# ### Plotting code refactored to:
#
# * ```dqa.plots.s3_filesizes_weekly.py```
# * ```dqa.plots.s3_size_collected_weekly.py```
#
# Note that plotting modules should be named according to:
#
# ```
# {s3, db}_{<lower_camel_case_description>}_{weekly, monthly, all}.py
# ```
#
# ## Preamble (imports, globals, utils)
# *these should be useful for other dqa*

#%%
# imports
# %matplotlib inline
from matplotlib import pyplot as plt
from datetime import datetime
from itertools import groupby
import calendar
from collections import defaultdict, Counter
import re
from functools import lru_cache
import numpy as np

from ojd_daps.dqa.data_getters import get_s3_job_ads
from ojd_daps.dqa.s3_utils import *

#%%
# globals
# reused utils globals

#%%
# utils

# -

#%%
# ## Other utils
#
# *these are **probably not** useful for other dqa*

#%%
# ## Main DQA (no fixed format from this point)
total_files = sum(1 for _ in get_s3_job_ads(JOB_BOARD, read_body=False))

total_files
#%%
# Randomly sample 1% of the data from S3 to speed things up

job_ads = get_job_ads_posted_date(job_board=JOB_BOARD, sample_ratio=0.01)

len(job_ads)

#%%
# Generate a list of filesizes
filesizes = []
for advert in job_ads[1]:
    filesizes.append(advert['filesize'])
#%%
# Plot a histogram of filesizes
plt.hist(filesizes, bins = 100)

#%%
# Plot scatter of filesizes across weeks
dates = []
for advert in job_ads[1]:
    dates.append(timestamp_to_universal_week(advert['posted']))

data = list(zip(dates, filesizes))

x, y = zip(*data)
plt.scatter(x,y)


# %%
# Plot scatter of sum of filesize across weeks

agg = [(key, sum(j for i, j in group)) for key, group in groupby(data, key=lambda x: x[0])]

x,y = zip(*agg)
plt.scatter(x,y)
