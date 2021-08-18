# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.10.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
"""
Runs skills detection across all job ads from December to May.
NB: Not optimised at all (e.g. no parallelisation), so runs for about ~16 hrs!

"""

import pickle
from tqdm import tqdm
from ojd_daps.flows.enrich.labs.skills.helper_utils import DATA_PATH
from ojd_daps.flows.enrich.labs.skills.skills_detection_utils import (
  load_model, 
  setup_spacy_model,
  detect_skills, 
  clean_text)

JOBFILE = "job_ads_DEC2020_MAY2021"
MODEL_NAME = "v01_1"
TEST = False

filepath = DATA_PATH / f'raw/job_ads/{JOBFILE}.pickle'


# %%
if __name__ == "__main__":
    # Load model
    model = load_model(MODEL_NAME, from_local=True)
    nlp = setup_spacy_model(model["nlp"])

    # Import the job adverts
    jobs = pickle.load(open(filepath,'rb')) 

    # Detect skills
    detected_skills = []

    for i, job in tqdm(enumerate(jobs), total=len(jobs)):

        text = job['description']
        
        skills_dict = detect_skills(clean_text(text), model, nlp, debug=True, return_dict=True)

        detected_skills.append(skills_dict)

        if TEST and (i > 10):
            break

    # Save the detected skills
    savefile = f"{JOBFILE}_detected_skills_{model['name']}"
    savepath = DATA_PATH / f"processed/detected_skills/{savefile}.pickle"
    pickle.dump(detected_skills, open(savepath, 'wb'))

# %%
