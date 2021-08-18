# %% [markdown]
# # Raw counts from the database
# https://github.com/nestauk/ojd_daps/issues/113
#
# Author: Joel K
#
# ### TL;DR
#
# * This is just an example of how to use the database really
#
# ### Issues raised
#
# * 
#
# ### Plotting code refactored to:
#
# * 
#
#
# ## Preamble (imports, globals, utils)
# *these should be useful for other dqa*

# %%
# imports
from data_getters import get_db_job_ads

# %% [markdown]
# ## Other utils
#
# *these are **probably not** useful for other dqa*

# %%
#@lru_cache()  # remember to use lru_cache when fetching data
# def get_data():
#     """
#     A little description
#     """
#     do_something()
#     return

# %% [markdown]
# ## Main DQA (no fixed format from this point)

# %%
# %%time
for i, job_ad in enumerate(get_db_job_ads('reed', chunksize=10000)):
    pass

print('Got', i+1, "job ads")
print(job_ad)

# %% [markdown]
# *make sure to put in notes*
