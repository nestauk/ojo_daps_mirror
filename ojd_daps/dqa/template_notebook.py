# # *Verbose title of the issue name*
# *link to the issue here*
#
# Author: *your name*
#
# ### TL;DR
#
# * *headlines from your work*
# * *one headline per line*
#
# ### Issues raised
#
# * *Add any GH issues raised because of this work here*  (see [issue](https://github.com/nestauk/ojd_daps/issues/117))
# * *Add other issues here etc* (see [issue](https://github.com/nestauk/ojd_daps/issues/118))
#
# ### Plotting code refactored to:
#
# * ```dqa.plots.<plot module name>```
# * ```dqa.plots.<another plot name>```
#
# Note that plotting modules should be named according to:
#
# ```
# {s3, db}_{<lower_camel_case_description>}_{weekly, monthly, all}.py
# ```
#
# ## Preamble (imports, globals, utils)
# *these should be useful for other dqa*

# +
# imports
# %matplotlib inline
from matplotlib import pyplot as plt

# globals
SOMETHING = None

# utils
def somekind_of_util(arg):    
    """
    A little description
    """
    do_something()
    return 


# -

# ## Other utils
#
# *these are **probably not** useful for other dqa*

#@lru_cache()  # remember to use lru_cache when fetching data
def get_data():
    """
    A little description
    """
    do_something()
    return

# ## Main DQA (no fixed format from this point)



# *make sure to put in notes*
