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
# # Postprocessing clustering results
#
# - Assigning skills entities to clusters
#
# To do:
# - Naming the clusters
# - Better versioning of the model
# - Add logger.info statements

# %%
from gensim.models import Word2Vec
import pandas as pd
import numpy as np
import pickle
import os

from ojd_daps.flows.enrich.labs.skills.skills_detection_utils import (
    load_model,
    save_model_locally,
    save_model_in_s3,
)
from ojd_daps.flows.enrich.labs.skills.helper_utils import (
    DATA_PATH,
    get_esco_skills,
    save_lookup,
    get_lookup,
)
from ojd_daps.flows.enrich.labs.skills.notebooks.PIPELINE_general_skills import (
    get_detected_surface_form_lists,
    get_surface_form_counts,
)
from ojd_daps.flows.enrich.labs.skills.notebooks.PIPELINE_assess_quality import (
    get_manual_review_results,
)

import ojd_daps.flows.enrich.labs.skills.cluster_utils as cluster_utils
from scipy.spatial.distance import pdist, squareform

# Clustering results folder
CLUSTERS_DIR = DATA_PATH / "processed/clusters"
CLUSTERS_DIR.mkdir(parents=True, exist_ok=True)

# %%
# Mock test for mapping Cluster 1 to higher level hierarchy
cluster_1_to_0 = {
    0: 6,  # Services, manual to services, manual, engineering
    1: 5,  # Business, admin and law
    2: 6,  # Engineering to services, manual, engineering
    3: 1,  # Health, care
    4: 4,  # ICT and project management
    5: 3,  # Marketing to Sales, logistics and marketing
    6: 3,  # Sales/Customer service to Sales, logistics and marketing
    7: 2,  # Education
    # 8: 1, # Biomedical to Health, care & biomedical sciences
    8: 0,  # languages to General work skills
    9: 0,  # transversal to General work skills
}


def assign_cluster_0(cluster_1):
    if cluster_1 in list(cluster_1_to_0.keys()):
        return cluster_1_to_0[cluster_1]
    else:
        return cluster_1


# %%
def get_entity_cluster_counts_lists(df, clusters="cluster_1"):
    """Get lists of cluster assignments and counts for each skill entity
    df = dataframe with counts and clusters
    """
    preferred_labels = df.preferred_label.unique()
    cluster_counts_lists = {}
    for lab in preferred_labels:
        df_skill = df[df.preferred_label == lab]
        cluster_lists = df_skill[clusters].to_list()
        count_lists = df_skill.counts.to_list()
        quality_lists = df_skill.predicted_q.to_list()
        cluster_counts_lists[lab] = (cluster_lists, count_lists, quality_lists)
    return cluster_counts_lists


def entropy(prop):
    return -np.sum(np.log(prop) * prop)


def norm_entropy(prop):
    """Entropy divided by the maximum possible entropy for the given number of probabilities"""
    if len(prop) == 1:
        return 0
    else:
        return entropy(prop) / entropy(np.array([1] * len(prop)) / len(prop))


def assign_skill_to_cluster(cluster_counts_list):
    df = pd.DataFrame(cluster_counts_list).T.rename(
        columns={0: "cluster", 1: "counts", 2: "predicted_q"}
    )
    df["weighted_counts"] = df["counts"] * df["predicted_q"]
    df_clust = (
        df.groupby("cluster")
        .sum()
        .reset_index()
        .sort_values("weighted_counts", ascending=False)
    )
    df_clust["prob"] = df_clust.weighted_counts / df_clust.weighted_counts.sum()
    if len(df_clust) or (df_clust.iloc[0] > df_clust.iloc[1]):
        assigned_clust = df_clust.iloc[0].cluster
    else:
        assigned_clust = df_clust[
            df_clust.weighted_counts == df_clust.weighted_counts.max()
        ].cluster.to_list()
    skill_entropy = norm_entropy(df_clust.prob.to_list())
    return assigned_clust, skill_entropy, df_clust[["cluster", "prob"]]


def assign_skills_to_clusters(cluster_counts_lists, level="cluster_1"):
    """Assign all skills to cluster"""
    skill_cluster_assignments = []
    skill_entropies = []
    for key in cluster_counts_lists:
        clust, skill_entropy, _ = assign_skill_to_cluster(cluster_counts_lists[key])
        skill_cluster_assignments.append(clust)
        skill_entropies.append(skill_entropy)

    skill_clusters = pd.DataFrame(
        data={
            "preferred_label": cluster_counts_lists.keys(),
            f"skill_entropy_{level}": skill_entropies,
            f"skill_{level}": skill_cluster_assignments,
        }
    )

    return skill_clusters


# %%
if __name__ == "__main__":
    # Most recent model
    skills_model = load_model("v01_1qc", from_local=True)
    sf_df = skills_model["surface_forms"].copy()

    # Import the job adverts
    filepath = (
        DATA_PATH
        / "processed/detected_skills/job_ads_DEC2020_MAY2021_detected_skills_v01_1.pickle"
    )
    job_skills = pickle.load(open(filepath, "rb"))

    # Get surface form counts
    detected_surface_forms = get_detected_surface_form_lists(job_skills)
    sf_counts = get_surface_form_counts(detected_surface_forms)

    print("Determining skills cluster assignments")
    # Assign first level labels
    partitions_counts = sf_df.merge(
        sf_counts[["surface_form", "counts"]], on=["surface_form"]
    ).dropna(axis=0, subset=["cluster_1"])
    cluster_counts_lists = get_entity_cluster_counts_lists(
        partitions_counts, "cluster_1"
    )
    skill_clusters_1 = assign_skills_to_clusters(cluster_counts_lists)

    # Assign second level labels
    partitions_counts_2 = partitions_counts.merge(
        skill_clusters_1, on="preferred_label", how="left"
    ).query("cluster_1==skill_cluster_1")
    cluster_counts_lists_2 = get_entity_cluster_counts_lists(
        partitions_counts_2, "cluster_2"
    )
    skill_clusters_2 = assign_skills_to_clusters(cluster_counts_lists_2, "cluster_2")

    # Final skills table
    skill_clusters = skill_clusters_1.merge(skill_clusters_2, on=["preferred_label"])

    print("Expanding skills cluster assignments")
    # Expand surface form assignment to all surface forms
    # that hav a skills entity with a skill cluster label
    sf_df_skill_clusters = sf_df.merge(skill_clusters)

    # Keep note of which surface forms have been originally clustered in their cluster,
    # vs. which have been 'assigned' or 'reassigned' their cluster label
    sf_df_skill_clusters["cluster_status"] = "clustered"
    sf_df_skill_clusters.loc[
        sf_df_skill_clusters.cluster_1.isnull(), "cluster_status"
    ] = "assigned"
    sf_df_skill_clusters.loc[
        (sf_df_skill_clusters.cluster_1.isnull() == False)
        & (sf_df_skill_clusters.cluster_1 != sf_df_skill_clusters.skill_cluster_1),
        "cluster_status",
    ] = "reassigned"

    # Apply skill cluster labels to all surface forms
    sf_df_skill_clusters["cluster_1"] = sf_df_skill_clusters["skill_cluster_1"]
    sf_df_skill_clusters["cluster_2"] = sf_df_skill_clusters["skill_cluster_2"]
    sf_df_skill_clusters["cluster_0"] = sf_df_skill_clusters.cluster_1.apply(
        lambda x: assign_cluster_0(x)
    )

    surface_forms_with_clusters = (
        skills_model["surface_forms"]
        .drop(["cluster_1", "cluster_2"], axis=1)
        .merge(
            sf_df_skill_clusters[
                [
                    "surface_form",
                    "cluster_0",
                    "cluster_1",
                    "cluster_2",
                    "cluster_status",
                ]
            ],
            on="surface_form",
            how="left",
        )
    )

    # Add manual review labels
    manual_review = get_manual_review_results().query("manual_OK==3")
    manual_review["manual_OK"] = 1
    new_sf_df = surface_forms_with_clusters.merge(
        manual_review[["surface_form", "manual_OK"]], how="left"
    )

    # Update model's surface forms
    new_model = skills_model.copy()
    new_model["surface_forms"] = new_sf_df
    new_model["name"] = "v02"
    save_model_locally(new_model)
    save_model_in_s3(new_model)


# %%
