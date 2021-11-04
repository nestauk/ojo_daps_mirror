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

# %% [markdown]
# # Manual refinements of the surface form list

# %%
import json
import pandas as pd
from pathlib import Path
import numpy as np

from ojd_daps.flows.enrich.labs.skills.helper_utils import DATA_PATH, get_esco_skills

MANUAL_CHECK_DIR = DATA_PATH / "aux/manual_checking"
skills = get_esco_skills()

# Outputs path
OUTPUTS_PATH = f"{DATA_PATH}/aux/manual_adjustments.json"


# %%
# Helper functions
def print_list(list_of_elements):
    """Helper function for organising the lists"""
    for element in sorted(list_of_elements):
        print(f"'{element}',")


def create_assign_dict_list(manual_assign):
    """Process the manual reassignments for outputting"""
    return [
        {"entity": e, "surface_form": s, "surface_form_type": "manual"}
        for s, e in manual_assign
    ]


def get_assigned_surface_forms(manual_assign):
    """Get a list of surface forms that will be assigned manually"""
    return [s[0] for s in manual_assign]


def get_manual_reassignments(fpath=MANUAL_CHECK_DIR):
    """Fetches manual reassignments surface forms' entities"""
    columns = ["surface_form", "entity", "new_entity"]
    checks = pd.read_excel(fpath / "v01_r1_hackweek_extract_counts.xlsx")[columns]
    checks = checks[-checks.new_entity.isnull()]
    return checks


def get_manual_review_results(fpath=MANUAL_CHECK_DIR, exclude=[2]):
    """Fetches results from manual review (exclude = [2] to remove ambiguous entries)"""
    # First manual check
    checks_1 = pd.read_excel(fpath / "v01_reed_extract_counts.xlsx")
    # Second manual check
    checks_2 = pd.read_excel(fpath / "v01_r1_hackweek_extract_counts.xlsx")

    # Combine both checks
    columns = [
        "surface_form",
        "surface_form_type",
        "preferred_label",
        "entity",
        "manual_OK",
    ]
    checks = (
        checks_1[columns]
        .append(checks_2[columns], ignore_index=True)
        .drop_duplicates("surface_form", keep="last")
    )
    # Exclude specific ratings
    checks = checks[
        (checks.manual_OK.isnull() == False) & (checks.manual_OK.isin(exclude) == False)
    ]
    return checks


# %% [markdown]
# ## 1. Surface forms to definitely keep in the final list

# %%
# Results from the manual review
manual_review_results = get_manual_review_results()

# %%
# List of surface forms
manual_keep = [
    "agile",
    "banking",
    "c#",
    "carer",
    "carers",
    "communication",
    "computer",
    "crm",
    "dementia",
    "dignity",
    "finance",
    "flexible",
    "java",
    "kpis",
    "negotiation",
    "numeracy",
    "portfolio",
    "python",
    "safeguarding",
    "saas",
    "warehouse",
]
# Additional list from the manual review
manual_keep += manual_review_results[
    manual_review_results.manual_OK == 3
].surface_form.to_list()
# Order and deduplicate
manual_keep = np.unique(manual_keep).tolist()

# %% [markdown]
# ## 2. Surface forms to remove from the final list

# %%
manual_remove = [
    "access",
    "administering",
    "advancement",
    "advert",
    "advertiser",
    "allowance",
    "america",
    "applicant",
    "battle",
    "bell",
    "billion",
    "bottom",
    "career",
    "chart",
    "check",
    "chore",
    "commentary",
    "compatibility",
    "complete application",
    "confidence",
    "conformance",
    "consent",
    "consideration",
    "convenience",
    "core part",
    "correspondence",
    "credibility",
    "despatch",
    "detect",
    "employment agency",
    "energy",
    "establishment",
    "euro",
    "exit",
    "face",
    "fuel",
    "gender identity",
    "generator",
    "high quality",
    "high standard",
    "history",
    "hook",
    "identification",
    "inception",
    "income",
    "individual right",
    "job description",
    "key",
    "keywords",
    "light",
    "limit",
    "listener",
    "logging",
    "look",
    "lunch",
    "majority",
    "map",
    "meaning",
    "mindset",
    "net",
    "occupancy",
    "offering",
    "option",
    "passion",
    "person",
    "politic",
    "political",
    "pool",
    "prize",
    "procedure",
    "process",
    "professional development",
    "promote",
    "reduction",
    "responsible",
    "salary",
    "security",
    "sound",
    "spirit",
    "split",
    "stress",
    "surrounding area",
    "tailor",
    "technologist",
    "territory",
    "text",
    "third party",
    "upper",
    "voice",
    "work",
]

# Surface forms that I'm less sure of whether they are noise or not
manual_remove_unsure = [
    "amendment",
    "broker",
    "case file",
    "design project",
    "process control",
    "project coordination",
    "service team",
    "support service",
    "technical team",
]

# Additional list from the manual review
manual_remove_all = (
    manual_remove
    + manual_remove_unsure
    + manual_review_results[manual_review_results.manual_OK == 1].surface_form.to_list()
)
# Order and deduplicate
manual_remove_all = np.unique(manual_remove_all).tolist()

# %% [markdown]
# ## 3. Surface forms to manually assign to entities

# %%
# Fetch reassignments from the manual review
manual_review_reassignments = get_manual_reassignments()[["surface_form", "new_entity"]]

# %%
# Additional manual reassignments
manual_assign = [
    ("insurance", 9084),
    ("tax", 2521),
    ("sale", 3659),
    ("audit", 7791),
    ("planning", 11982),
    ("cleaner", 4186),
    ("smartphone", 697),
    ("smartphones", 697),
    ("census", 387),
    ("teaching", 4782),
    ("teacher", 4782),
    ("ms office", 12963),
    ("electrical installation", 10570),
    ("electrician", 10570),
    ("maths", 3571),
    ("delivery driver", 2279),
    ("driving license", 2279),
    ("van driver", 2279),
    ("power point", 1856),
    ("powerpoint", 1856),
    ("credit control", 11811),
    ("cooking", 12815),
]
# Combine all assignments
manual_assign += list(
    zip(
        manual_review_reassignments.surface_form.to_list(),
        manual_review_reassignments.new_entity.to_list(),
    )
)


# %%
# Check the reassigned surface forms
manual_assign_df = pd.DataFrame(manual_assign, columns=["surface_form", "id"]).merge(
    skills[["preferred_label", "id"]], how="left"
)
assert manual_assign_df.surface_form.duplicated().sum() == 0

# %% [markdown]
# ## Export

# %%
# Create the dictionary
manual_refinements = {
    "keep": sorted(np.unique(manual_keep + get_assigned_surface_forms(manual_assign))),
    "remove": list(np.unique(manual_remove_all)),
    "assign": create_assign_dict_list(manual_assign),
}

# Save the dictionary in a json file
json.dump(manual_refinements, open(OUTPUTS_PATH, "w"), indent=4)
