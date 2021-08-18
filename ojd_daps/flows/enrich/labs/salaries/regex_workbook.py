# +
from daps_utils.db import object_as_dict, db_session
from ojd_daps.orms.raw_jobs import RawJobAd
from functools import lru_cache
import boto3
import re
import numpy as np
from regex import single_regex_utils

############
# Following lines are needed until issue is fixed in daps_utils
from daps_utils import db
import ojd_daps
db.CALLER_PKG = ojd_daps
db_session = db.db_session
############

single_regex_utils.save_model('(\d*[.]?\d*)', 'max')  # <--- After all of my hard work, I'll save my model config


# +
def load_jobs(limit=10):
    with db_session('production') as session:
        for ad in session.query(RawJobAd).limit(limit):
            yield object_as_dict(ad)

# Example of applying my model
fields_to_print = ('job_title_raw', 'contract_type_raw', 'job_salary_raw')
for job_ad in load_jobs():
    prediction = single_regex_utils.apply_model(job_ad)
    print(*(job_ad[x] for x in fields_to_print), prediction, sep="\n")
    print()
# -




