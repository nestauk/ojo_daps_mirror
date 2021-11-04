# -*- coding: utf-8 -*-
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
# # Review clusters and give them names

# %%
from ojd_daps.flows.enrich.labs.skills.notebooks.PIPELINE_general_skills import (
    build_lookup_dicts,
    get_detected_surface_form_lists,
    get_all_surface_form_ids,
    get_unique_surface_form_ids,
    get_all_surface_forms,
    get_unique_surface_forms,
    get_surface_form_counts,
)

from ojd_daps.flows.enrich.labs.skills.notebooks.PIPELINE_surface_form_clustering import (
    surface_form_2_vec,
)

from ojd_daps.flows.enrich.labs.skills.helper_utils import (
    DATA_PATH,
    get_esco_skills,
)

from ojd_daps.flows.enrich.labs.skills.skills_detection_utils import (
    load_model,
    setup_spacy_model,
    detect_skills,
    clean_text,
)


# %%
import pickle
import numpy as np
import altair as alt
from ojd_daps.flows.enrich.labs.skills.altair_save_utils import (
    google_chrome_driver_setup,
    save_altair,
)
import umap
import pandas as pd

colour_pal = [
    "#000075",
    "#e6194b",
    "#3cb44b",
    "#f58231",
    "#f032e6",
    "#bcf60c",
    "#fabebe",
    "#008080",
    "#e6beff",
    "#DCDCDC",
    "#9a6324",
    "#800000",
    "#808000",
    "#ffe119",
    "#46f0f0",
    "#4363d8",
    "#911eb4",
    "#aaffc3",
    "#000000",
    "#ffd8b1",
    "#808000",
    "#000075",
    "#DCDCDC",
]

# %%
model = load_model("v02", from_local=True)
nlp = setup_spacy_model(model["nlp"])

# %%
model["surface_forms"].sample()

# %% [markdown]
# ## Characterise the clusters
#
# - Visualise the clusters
# - Show most popular surface forms for each cluster
# - Generate automatic labels
# - Review

# %%
cluster_partitions = (
    model["surface_forms"]
    .drop_duplicates(["cluster_0", "cluster_1", "cluster_2"])
    .sort_values(["cluster_2"])
    .reset_index(drop=True)
)[["cluster_0", "cluster_1", "cluster_2"]].to_csv(
    "../data/processed/clusters/clusters_v02.csv", index=False
)

# %%
cluster_partitions = (
    model["surface_forms"]
    .groupby(["cluster_0", "cluster_1", "cluster_2"])
    .agg(counts=("surface_form", "count"))
    .reset_index(drop=False)
    .sort_values(["cluster_2"])
)[
    ["cluster_0", "cluster_1", "cluster_2", "counts"]
]  # .to_csv('../data/processed/clusters/clusters_v02.csv', index=False)
cluster_partitions.to_csv("../data/processed/clusters/clusters_v02.csv", index=False)


# %%
def get_sf2vec_vectors(model, unique_sf):
    """extract sf2vec vectors from model"""
    skill2vec_emb = []
    sfs = []
    for sf in unique_sf:
        try:
            emb = model.wv[sf]
            skill2vec_emb.append(emb)
            sfs.append(sf)
        except KeyError:
            pass
    skill2vec_emb = np.array(skill2vec_emb)
    return skill2vec_emb, sfs


# %%
sf2vec_model = surface_form_2_vec(None, use_cached=True)

# %%
all_sf = model["surface_forms"].surface_form.to_list()

# %%
emb, emb_sf = get_sf2vec_vectors(sf2vec_model, all_sf)

# %%
emb.shape

# %%
# quality_model = pickle.load(open('../models/quality_rating_model_v01_1q.p', 'rb'))
# quality_model['best_threshold']

# %%
# Import the job adverts
filepath = (
    DATA_PATH
    / "processed/detected_skills/job_ads_DEC2020_MAY2021_detected_skills_v01_1.pickle"
)
job_skills = pickle.load(open(filepath, "rb"))

# Export table for checking
detected_surface_forms = get_detected_surface_form_lists(job_skills)
sf_counts = get_surface_form_counts(detected_surface_forms)

# %%
sf_counts.head(1)

# %%
sf_df = model["surface_forms"].merge(sf_counts, on="surface_form", how="left")


# %%
# For each cluster_2
def get_cluster_forms(sf_df, int_label=0, cluster="cluster_2"):
    clust_forms = (
        sf_df.query(f"{cluster}=={int_label}")
        .sort_values("counts", ascending=False)
        .query("(manual_OK==1) | (is_predicted_OK==1)")
    )
    return clust_forms[
        [
            "surface_form",
            "surface_form_type",
            "entity",
            "preferred_label",
            "counts",
            "predicted_q",
            "is_predicted_OK",
        ]
    ]


def print_frequent_forms(sf_df, int_label=0, cluster="cluster_2", n=10):
    df = get_cluster_forms(sf_df, int_label, cluster)
    return ", ".join(df.iloc[0:n].surface_form.to_list())


# %%
get_cluster_forms(sf_df, 0)

# %%
cluster_level = "cluster_2"
clust_label = 15
print(print_frequent_forms(sf_df, int_label=clust_label, cluster=cluster_level, n=10))
get_cluster_forms(sf_df, clust_label, cluster=cluster_level).head(20)

# %% [markdown]
# ## Visualise

# %%
reducer = umap.UMAP(
    random_state=111, n_neighbors=20, min_dist=0.01, n_components=2, metric="cosine"
)
embedding = reducer.fit_transform(emb)
embedding.shape

# %%
df_viz = pd.DataFrame(
    data={"surface_form": emb_sf, "x": embedding[:, 0], "y": embedding[:, 1]}
)
df_viz = df_viz.merge(sf_df, how="left")
df_viz = df_viz[-df_viz.cluster_0.isnull()]
df_viz["log_counts"] = np.power(df_viz["counts"], 1 / 2)
for i in range(3):
    df_viz[f"cluster_{i}"] = df_viz[f"cluster_{i}"].apply(lambda x: str(int(x)))
# df_viz = df_viz[df_viz.cluster_2.isin(['0'])]

df_viz = df_viz[df_viz.cluster_status == "clustered"]

alt.data_transformers.disable_max_rows()
fig = (
    alt.Chart(df_viz, width=1000, height=600)
    .mark_circle(size=25)
    .encode(
        x=alt.X("x", axis=alt.Axis(grid=False)),
        y=alt.Y("y", axis=alt.Axis(grid=False)),
        color=alt.Color(
            "cluster_1", scale=alt.Scale(domain=list(range(10)), range=colour_pal)
        ),
        size="log_counts",
        tooltip=[
            "surface_form",
            "preferred_label",
            "cluster_0",
            "cluster_1",
            "cluster_2",
            "counts",
        ],
    )
    .interactive()
)

# %%
fig

# %% [markdown]
# ### Export table for review

# %%
cols = [
    "surface_form",
    "counts",
    "fraction",
    "surface_form_type",
    "entity",
    "preferred_label",
    "cluster_0",
    "cluster_1",
    "cluster_2",
    "cluster_status",
    "predicted_q",
    "is_predicted_OK",
    "manual_OK",
]

df = (
    sf_df[cols]
    .sort_values("counts", ascending=False)
    .sort_values(["cluster_0", "cluster_1", "cluster_2"])
)
df = df[(df.is_predicted_OK == 1) | (df.manual_OK == 1)]
df.to_csv("v02_surface_forms.csv", index=False)

# %%
driver = google_chrome_driver_setup()

# %%
save_altair(fig, "skills_constellation", driver)

# %% [markdown]
# ## Process skills review results
#
# - Process the manual reassignments of surface forms' cluster_2 memberships
#     - Reassign the associated surface forms as well
#     - Remove the surface forms with rechecked_cluster==0
# - Reassign cluster_2 and cluster_1 memberships
#

# %%
# New cluster_2 memberships

new_cluster2_to_cluster2 = {
    28: 9,  # Project management merge with Business management,
}

new_cluster2_to_cluster1 = {
    3: 10,  # scientific research to its own cluster_1
    28: 1,  # proj management to business admin
    14: 0,  # civil eng to manufacturing and engineering
    2: 11,  # food and cleaning to a separate cluster
    7: 12,  # Finance & Law cluster
    8: 12,  # Finance & Law cluster
    11: 12,  # Finance & Law cluster
    12: 12,  # Finance & Law cluster
    35: 13,  # Procurement, Logistics & Trade
    36: 13,  # Procurement, Logistics & Trade
    37: 13,  # Procurement, Logistics & Trade
    21: 14,  # Healthcare
    22: 14,  # Healthcare
    23: 14,  # Healthcare
    24: 14,  # Healthcare
}

new_cluster2_to_cluster0 = {
    3: 1,  # scientific research to healthcare,
    28: 5,  # proj management to business admin,
    2: 7,  # food and cleaning to a separate cluster
}

# %%
model = load_model("v02", from_local=True)
sf_df = model["surface_forms"].copy()


# %%
# Adjust cluster_2 memberships w.r.t. other clusters
for key in new_cluster2_to_cluster0:
    sf_df.loc[sf_df.cluster_2 == key, "cluster_0"] = new_cluster2_to_cluster0[key]

for key in new_cluster2_to_cluster1:
    sf_df.loc[sf_df.cluster_2 == key, "cluster_1"] = new_cluster2_to_cluster1[key]

for key in new_cluster2_to_cluster2:
    sf_df.loc[sf_df.cluster_2 == key, "cluster_2"] = new_cluster2_to_cluster2[key]


# %%
clust_cols = ["cluster_0", "cluster_1", "cluster_2"]
new_partitions = (
    sf_df.sort_values(clust_cols)
    .drop_duplicates(clust_cols)[clust_cols]
    .reset_index(drop=True)
)
# new_partitions

# %%
# Load manual review results & reassignments
reviewed_cluster_assignments = pd.read_excel(
    "../data/aux/v02_surface_forms_rechecked.xlsx"
)

# Surface form cluster_2 membership reassignments
cl2_reassignments = reviewed_cluster_assignments[
    reviewed_cluster_assignments.manual_reassignment_cluster_2.isnull() == False
]


# %%
# Additional surface forms to remove
additional_to_remove = [
    "adjusts",
    "mattress",
    "pedigree",
    "shadow",
    "refusal",
    "prepares",
    "liases",
]
# All surface forms to remove
forms_to_remove = sorted(
    additional_to_remove
    + reviewed_cluster_assignments[
        reviewed_cluster_assignments.rechecked_cluster == 0
    ].surface_form.to_list()
)
len(forms_to_remove)


# %%
sf_df = sf_df[-sf_df.surface_form.isin(forms_to_remove)]

# %%
# Check reassignment conflicts
unique_entities = cl2_reassignments.entity.unique()
new_cluster = []
for i in unique_entities:
    df = cl2_reassignments[cl2_reassignments.entity == i]
    new_cluster.append(set(df.manual_reassignment_cluster_2))
is_conflict = [len(c) != 1 for c in new_cluster]
conflicted_entities = sorted(
    [entity for i, entity in enumerate(unique_entities) if is_conflict[i]]
)
len(conflicted_entities)


# %%
conflicted_entities

# %%
# Adjust surface forms' cluster_2 memberships
entity_new_clusters = cl2_reassignments.drop_duplicates("entity")[
    ["entity", "manual_reassignment_cluster_2"]
]
for i, row in entity_new_clusters.iterrows():
    new_cluster_2 = row.manual_reassignment_cluster_2
    if new_cluster_2 == 28:
        new_cluster_2 = 9  # Project management
    sf_df.loc[sf_df.entity == row.entity, "cluster_2"] = new_cluster_2
    sf_df.loc[sf_df.entity == row.entity, "cluster_1"] = (
        new_partitions[new_partitions.cluster_2 == new_cluster_2].iloc[0].cluster_1
    )
    sf_df.loc[sf_df.entity == row.entity, "cluster_0"] = (
        new_partitions[new_partitions.cluster_2 == new_cluster_2].iloc[0].cluster_0
    )
    sf_df.loc[sf_df.entity == row.entity, "cluster_status"] = "manually_reassigned"


# %%
sf_df.sort_values(clust_cols).drop_duplicates(clust_cols)[clust_cols].reset_index(
    drop=True
)

# %% [markdown]
# ## Review and subcluster hospitality cluster
# - Sub-cluster select clusters (17, maybe 29, and 43)
# - Review the sub-clusterings
# - Produce final cluster assignments, cluster names and most popular surface forms/skills for each cluster
#
#

# %%
from gensim.models import Word2Vec
from scipy.spatial.distance import pdist, squareform
import ojd_daps.flows.enrich.labs.skills.cluster_utils as cluster_utils
from ojd_daps.flows.enrich.labs.skills.skills_detection_utils import (
    load_model,
    save_model_locally,
    save_model_in_s3,
)

# %%
# Load the vectors
sf2vec_model = Word2Vec.load("../models/sf2vec_sf_clusters_v01_1qc.model")

# Keep only surface forms that have appeared at least 10 times
FREQ_THRESH = 10


# %%
# Nodes
def get_nodes(sf_df, cluster_to_subcluster):
    nodes = sf_df.copy()
    nodes = nodes.merge(sf_counts)
    nodes = nodes[nodes.counts >= FREQ_THRESH]
    nodes = nodes[nodes.cluster_2 == cluster_to_subcluster]
    clust_ids = list(range(len(nodes)))
    nodes["clust_id"] = clust_ids
    return nodes


def update_session_name_params(clustering_params, cluster_to_subcluster):
    # Name of this clustering session
    session_name = "sf_clusters_v02_subclustering_" + str(cluster_to_subcluster)

    # Path to save the clustering results
    fpath_ = DATA_PATH / f"interim/clustering/{session_name}/"
    fpath_.mkdir(parents=True, exist_ok=True)
    fpath = f"{fpath_}/"

    clustering_params["session_name"] = session_name
    clustering_params["fpath"] = fpath
    return clustering_params


def get_similarity_matrix(sf2vec_model, nodes):
    # Vectors and similiarity matrix
    vectors = np.array([sf2vec_model.wv[sf] for sf in nodes.surface_form.to_list()])
    sim_matrix = 1 - squareform(pdist(vectors, metric="cosine"))
    return sim_matrix, vectors


def prepare_initial_partition(nodes, clustering_params):
    # Prepare and save the Level-0 partition file (all in one cluster)
    partition_df = pd.DataFrame()
    partition_df["id"] = nodes.clust_id.to_list()
    partition_df["cluster"] = [0] * len(nodes)
    partition_df.to_csv(
        clustering_params["fpath"]
        + f'/{clustering_params["session_name"]}_clusters_Level0.csv',
        index=False,
    )


def subclustering_prep(cluster_to_subcluster, clustering_params, sf_df, sf2vec_model):
    nodes = get_nodes(sf_df, cluster_to_subcluster)
    clustering_params = update_session_name_params(
        clustering_params, cluster_to_subcluster
    )
    sim_matrix, vectors = get_similarity_matrix(sf2vec_model, nodes)
    prepare_initial_partition(nodes, clustering_params)
    return clustering_params, nodes, sim_matrix, vectors


# %%
# CLUSTERING PARAMS
# Name of this clustering session
session_name = "sf_clusters_v02_select_subclustering"
# Number of nearest neighbours used for the graph construction
nearest_neighbours = [10, 15, 20, 25]
# Ensemble size for the first step
N = 1000
# Ensemble size for the consensus step
N_consensus = 100
# Number of clustering trials for each nearest neighbour value
N_nn = N // len(nearest_neighbours)
# Which clusters to break down from the partition
clusters = "all"  # Either a list of integers, or 'all'
# Path to save the clustering results
fpath_ = DATA_PATH / f"interim/clustering/{session_name}/"
# fpath_.mkdir(parents=True, exist_ok=True)
fpath = f"{fpath_}/"

clustering_params = {
    "N": N,
    "N_consensus": N_consensus,
    "N_nn": N_nn,
    "clusters": clusters,
    "fpath": fpath,
    "session_name": session_name,
    "nearest_neighbours": nearest_neighbours,
}

# %%
cluster_to_subcluster = 17
clustering_params, nodes, sim_matrix, vectors = subclustering_prep(
    cluster_to_subcluster, clustering_params, sf_df, sf2vec_model
)


# %%
### Clustering
# Set the random_state variable for reproduciblity
clustering_params["random_state"] = 13131
# Perform the clustering
cluster_utils.subcluster_nodes(W=sim_matrix, l=0, **clustering_params)
# Collect subclusters into one partition
partition_1 = cluster_utils.collect_subclusters(
    0, clustering_params["fpath"], clustering_params["session_name"], n_total=len(nodes)
)
############
print(
    cluster_utils.ConsensusClustering.describe_partition(partition_1.cluster.to_list())
)
############

# %%
COOC = np.load(
    clustering_params["fpath"]
    + "/sf_clusters_v02_subclustering_17_COOC_Level0_Cluster0.npy"
)

# %%
cluster_utils.plot_sorted_matrix(COOC, partition_1.cluster.to_list())

# %%
# 0 = Horticulture
# 1 = Animal Husbandry
# 2 = Ecology
# 3 = Hospitality

# Final partitions for cluster_2:
# 17 = Horticulture, Husbandry & Ecology
# 2 = Hospitality

# %%
# nodes[list(partition_1.cluster==0)].sort_values('counts', ascending=False)

# %%
# nodes[nodes.entity.isin(manual_assignment_to_clust_2)]

# %%
manual_assignment_to_clust_2 = {3583, 5917, 6178, 1484, 1100, 3045, 4842}

# %%
nodes_clust = nodes.copy().reset_index(drop=True)
nodes_clust = nodes_clust[(nodes_clust.is_predicted_OK) | (nodes_clust.manual_OK == 1)]
nodes_clust["new_cluster"] = partition_1.cluster
keep_clust_17 = set(
    nodes_clust[nodes_clust.new_cluster.isin([0, 1, 2])].entity.to_list()
).difference(manual_assignment_to_clust_2)
new_clust_2 = set(
    nodes_clust[nodes_clust.new_cluster.isin([3])].entity.to_list()
).union(manual_assignment_to_clust_2)

# %%
conflicts = keep_clust_17.intersection(new_clust_2)
len(conflicts)

# %%
# Reassign to the new cluster #2
sf_df.loc[sf_df.entity.isin(new_clust_2), "cluster_2"] = 2
sf_df.loc[sf_df.entity.isin(new_clust_2), "cluster_1"] = 11
sf_df.loc[sf_df.entity.isin(new_clust_2), "cluster_0"] = 7
sf_df.loc[sf_df.entity.isin(new_clust_2), "cluster_status"] = "manually_reassigned"

# %% [markdown]
# ### Review and subcluster services cluster

# %%
cluster_to_subcluster = 2
clustering_params, nodes, sim_matrix, vectors = subclustering_prep(
    cluster_to_subcluster, clustering_params, sf_df, sf2vec_model
)

### Clustering
# Set the random_state variable for reproduciblity
clustering_params["random_state"] = 777
# Perform the clustering
cluster_utils.subcluster_nodes(W=sim_matrix, l=0, **clustering_params)
# Collect subclusters into one partition
partition_1 = cluster_utils.collect_subclusters(
    0, clustering_params["fpath"], clustering_params["session_name"], n_total=len(nodes)
)
############
print(
    cluster_utils.ConsensusClustering.describe_partition(partition_1.cluster.to_list())
)
############

# %%
COOC = np.load(
    clustering_params["fpath"]
    + f"/sf_clusters_v02_subclustering_{cluster_to_subcluster}_COOC_Level0_Cluster0.npy"
)
cluster_utils.plot_sorted_matrix(COOC, partition_1.cluster.to_list())

# %%
# Visualise general skills
reducer_subclusters = umap.UMAP(
    random_state=121, n_neighbors=20, min_dist=0.01, n_components=2, metric="cosine"
)
embedding_subclust = reducer_subclusters.fit_transform(vectors)
df_viz_subclust = nodes.copy().reset_index(drop=True)
df_viz_subclust["x"] = embedding_subclust[:, 0]
df_viz_subclust["y"] = embedding_subclust[:, 1]
df_viz_subclust["cluster"] = partition_1["cluster"].apply(lambda x: str(x))
df_viz_subclust["log_counts"] = np.power(df_viz_subclust["counts"], 1 / 3)
alt.data_transformers.disable_max_rows()
fig_ = (
    alt.Chart(df_viz_subclust, width=1000, height=600)
    .mark_circle(size=25)
    .encode(
        x=alt.X("x", axis=alt.Axis(grid=False)),
        y=alt.Y("y", axis=alt.Axis(grid=False)),
        color=alt.Color(
            "cluster", scale=alt.Scale(domain=list(range(3)), range=colour_pal)
        ),
        size="log_counts",
        tooltip=[
            "surface_form",
            "preferred_label",
            "cluster_0",
            "cluster_1",
            "cluster_2",
            "cluster",
            "counts",
        ],
    )
    .interactive()
)
fig_

# %%
# nodes[list(partition_1.cluster==2)].sort_values('counts', ascending=False).head(50)

# %%
manual_assignment_to_clust_2 = {8901, 3583, 2490}
new_clust_44 = set(nodes[list(partition_1.cluster == 2)].entity.unique()).difference(
    manual_assignment_to_clust_2
)
# Reassign to the new cluster #44
sf_df.loc[sf_df.entity.isin(new_clust_44), "cluster_2"] = 44
sf_df.loc[sf_df.entity.isin(new_clust_44), "cluster_1"] = 11
sf_df.loc[sf_df.entity.isin(new_clust_44), "cluster_0"] = 7
sf_df.loc[sf_df.entity.isin(new_clust_44), "cluster_status"] = "manually_reassigned"

# %%
# sf_df[sf_df.entity.isin(new_clust_44)]

# %% [markdown]
# ### Review and subcluster general skills

# %%
# cluster_to_subcluster = 43
# clustering_params, nodes, sim_matrix, vectors = subclustering_prep(cluster_to_subcluster, clustering_params, sf_df, sf2vec_model)

# %%
# ### Clustering
# # Set the random_state variable for reproduciblity
# clustering_params['random_state'] = 2323
# # Perform the clustering
# cluster_utils.subcluster_nodes(W=sim_matrix, l=0, **clustering_params)
# # Collect subclusters into one partition
# partition_1 = cluster_utils.collect_subclusters(0, clustering_params['fpath'], clustering_params['session_name'], n_total=len(nodes))
# ############
# print(cluster_utils.ConsensusClustering.describe_partition(partition_1.cluster.to_list()))
# ############

# %%
# COOC = np.load(clustering_params['fpath'] + f'/sf_clusters_v02_subclustering_{cluster_to_subcluster}_COOC_Level0_Cluster0.npy')
# cluster_utils.plot_sorted_matrix(COOC, partition_1.cluster.to_list());

# %%
# nodes[list(partition_1.cluster==3)].sort_values('counts', ascending=False).head(15)

# %%
# # Visualise general skills
# reducer_43 = umap.UMAP(random_state=121,  n_neighbors=20, min_dist=0.01, n_components=2, metric='cosine')
# embedding_43 = reducer_43.fit_transform(vectors)
# df_viz_43 = nodes.copy().reset_index(drop=True)
# df_viz_43['x'] = embedding_43[:, 0]
# df_viz_43['y'] = embedding_43[:, 1]
# df_viz_43['cluster'] = partition_1['cluster'].apply(lambda x: str(x))
# df_viz_43['log_counts'] = np.power(df_viz_43['counts'], 1/3)
# alt.data_transformers.disable_max_rows()
# fig_=alt.Chart(df_viz_43, width=1000, height=600).mark_circle(size=25).encode(
#     x=alt.X('x', axis=alt.Axis(grid=False)),
#     y=alt.Y('y', axis=alt.Axis(grid=False)),
#     color=alt.Color('cluster', scale=alt.Scale(domain=list(range(3)), range=colour_pal)),
#     size='log_counts',
#     tooltip=['surface_form', 'preferred_label', 'cluster_0', 'cluster_1', 'cluster_2', 'counts']
# ).interactive()
# fig_

# %% [markdown]
# ### Cluster names

# %%
final_partitions = (
    sf_df.sort_values(clust_cols)
    .drop_duplicates(clust_cols)[clust_cols]
    .reset_index(drop=True)
    .copy()
)
OJO_cluster_labels = pd.read_excel("../data/aux/OJO cluster labels.xlsx")
final_partitions_labels = final_partitions.merge(
    OJO_cluster_labels[
        ["cluster_2", "label_cluster_0", "label_cluster_1", "label_cluster_2"]
    ]
)


# %%
# for cl in ['cluster_0', 'cluster_1', 'cluster_2']:
#     final_partitions_labels[cl] = final_partitions_labels[cl].astype(int)

# %%
sf_df_labels = sf_df.merge(final_partitions_labels, how="left")
sf_df_labels_counts = sf_df_labels.merge(sf_counts)

# %%
cl = "cluster_2"
cl_counts = sf_df_labels_counts.groupby(cl).agg(counts=("counts", "sum")).reset_index()

# %%
new_order = [
    1,
    0,
    2,
    4,
    6,
    5,
    7,
    3,
    9,
    8,
    10,
    11,
    15,
    16,
    12,
    13,
    14,
    19,
    18,
    17,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    35,
    34,
    37,
    36,
    38,
    39,
    40,
]

# %%
final_partitions_labels = (
    final_partitions_labels.merge(cl_counts).sort_values(
        ["cluster_0", "cluster_1", "counts"]
    )
).loc[new_order]

# %%
# final_partitions_labels

# %% [markdown]
# ### New integers

# %%
sf_df_new_labels = sf_df_labels.copy()


# %%
def change_label(label, old_label_to_new_label):
    if label != np.nan:
        return int(old_label_to_new_label[label])
    else:
        return label


# %%
for cl in ["cluster_1", "cluster_2"]:
    old_labels = final_partitions_labels[cl].unique()
    old_label_to_new_label = dict(zip(old_labels, range(len(old_labels))))
    sf_df_new_labels.loc[
        sf_df_new_labels[cl].isnull() == False, cl
    ] = sf_df_new_labels.loc[sf_df_new_labels[cl].isnull() == False, cl].apply(
        lambda x: change_label(x, old_label_to_new_label)
    )

# %%
final_partitions_new_labels = (
    sf_df_new_labels.sort_values(clust_cols)
    .drop_duplicates(clust_cols)[clust_cols]
    .reset_index(drop=True)
    .copy()
)
final_partitions_new_labels = final_partitions_new_labels[
    -final_partitions_new_labels.cluster_0.isnull()
]
final_partitions_new_labels

# %% [markdown]
# ### Produce most popular surface forms

# %%
sf_df_labels_counts = sf_df_new_labels.merge(sf_counts)

# %%
sf_df_ = sf_df_labels_counts[
    (sf_df_labels_counts.is_predicted_OK == 1) | (sf_df_labels_counts.manual_OK == 1)
]


# %%
sf_df_[sf_df_.surface_form == "multidisciplinary"]

# %%
cluster_str = [
    str(int(i))
    + "_"
    + "_".join(
        print_frequent_forms(sf_df_, int_label=i, cluster="cluster_2", n=10).split(", ")
    )
    for i in final_partitions_new_labels.cluster_2.to_list()
]


# %%
for s in cluster_str:
    print(s)

# %% [markdown]
# ### Visualise the final partition

# %%
emb, emb_sf = get_sf2vec_vectors(
    sf2vec_model, sf_df_[sf_df_.counts >= 10].surface_form.to_list()
)

# %%
reducer = umap.UMAP(
    random_state=111, n_neighbors=20, min_dist=0.01, n_components=2, metric="cosine"
)
embedding = reducer.fit_transform(emb)
embedding.shape

# %%
df_viz = pd.DataFrame(
    data={"surface_form": emb_sf, "x": embedding[:, 0], "y": embedding[:, 1]}
)
df_viz = df_viz.merge(sf_df_, how="left")
df_viz = df_viz[-df_viz.cluster_0.isnull()]
df_viz["rescaled_counts"] = np.power(df_viz["counts"], 1 / 2)
for i in range(3):
    df_viz[f"cluster_{i}"] = df_viz[f"cluster_{i}"].apply(lambda x: str(int(x)))
# df_viz = df_viz[df_viz.cluster_2.isin(['0'])]

alt.data_transformers.disable_max_rows()
fig = (
    alt.Chart(df_viz, width=600, height=600)
    .mark_circle(size=25)
    .encode(
        x=alt.X("x", axis=alt.Axis(grid=False)),
        y=alt.Y("y", axis=alt.Axis(grid=False)),
        color=alt.Color("label_cluster_0", scale=alt.Scale(range=colour_pal)),
        size="rescaled_counts",
        tooltip=[
            "surface_form",
            "preferred_label",
            "label_cluster_0",
            "label_cluster_1",
            "label_cluster_2",
        ],
    )
    .interactive()
)

# %%
fig

# %%
driver = google_chrome_driver_setup()
save_altair(fig, "skills_constellation_v02_1", driver)

# %% [markdown]
# ### Save a new model

# %%
clust_label_cols = [
    "cluster_0",
    "cluster_1",
    "cluster_2",
    "label_cluster_0",
    "label_cluster_1",
    "label_cluster_2",
]
FINAL_LABELS = (
    sf_df_new_labels[clust_label_cols]
    .drop_duplicates()
    .sort_values(clust_label_cols)
    .reset_index(drop=True)
)
FINAL_LABELS = FINAL_LABELS[-FINAL_LABELS.cluster_0.isnull()]

# %%
FINAL_LABELS.to_csv("../data/processed/clusters/clusters_v02_1.csv", index=False)

# %%
FINAL_LABELS

# %%
FINAL_LABELS_FREQ_FORMS = FINAL_LABELS.copy()
freq_surface_forms = [
    print_frequent_forms(sf_df_, int_label=i, cluster="cluster_2", n=10)
    for i in FINAL_LABELS.cluster_2.to_list()
]
FINAL_LABELS_FREQ_FORMS["most_frequent_surface_forms"] = freq_surface_forms

# %%
FINAL_LABELS_FREQ_FORMS.to_csv(
    "../data/processed/clusters/clusters_v02_1_frequent_forms.csv", index=False
)

# %%
model_surface_forms = sf_df_new_labels[
    [
        "entity",
        "surface_form",
        "surface_form_type",
        "preferred_label",
        "predicted_q",
        "is_predicted_OK",
        "manual_OK",
        "cluster_status",
        "cluster_0",
        "cluster_1",
        "cluster_2",
        "label_cluster_0",
        "label_cluster_1",
        "label_cluster_2",
    ]
]

# %%
rechecked = reviewed_cluster_assignments[
    reviewed_cluster_assignments.rechecked_cluster == 1
].surface_form.to_list()
model_surface_forms.loc[
    model_surface_forms.surface_form.isin(rechecked), "manual_OK"
] = 1

# %%
additional_OKs = [
    "delivery driver",
]
model_surface_forms.loc[
    model_surface_forms.surface_form.isin(additional_OKs), "manual_OK"
] = 1
model_surface_forms.loc[model_surface_forms.surface_form.isin(additional_OKs)]

# %%
# model_surface_forms[model_surface_forms.is_predicted_OK==1]

# %%
# Update model's surface forms
new_model = model.copy()
new_model["surface_forms"] = model_surface_forms
new_model["name"] = "v02_1"
save_model_locally(new_model)
save_model_in_s3(new_model)

# %% [markdown]
# ## Final fine tuning

# %%
# x_model = skills_utils.load_model("v02_1", from_local=False)
# x_model['matcher'].add("new_manual", [nlp('learning disability')])

# %%
# x_model['matcher'](nlp('3d body scan, oversee carrer operation'))

# %%
# x_model['matcher'](nlp('3d body scan and learning disability'))

# %%
# new_manual_add = [
#     {
#         'surface_form':
#         'entity':
#         'cluster_0':
#         'cluster_1':
#         'cluster_2':
#     }

# ]
# for d in new_manual_add:
#     d['manual_OK'] = 1
#     d['cluster_status'] = 'manual'
#     d['label_cluster_0'] = FINAL_LABELS[FINAL_LABELS.cluster_0==d['cluster_0']].entity
#     d['label_cluster_1'] = FINAL_LABELS[FINAL_LABELS.cluster_1==d['cluster_1']].entity
#     d['label_cluster_2'] = FINAL_LABELS[FINAL_LABELS.cluster_2==d['cluster_2']].entity

# %% [markdown]
# # Test

# %%
import ojd_daps.flows.enrich.labs.skills.skills_detection_utils as skills_utils
import importlib

importlib.reload(skills_utils)
model = skills_utils.load_model("v02_1", from_local=True)
nlp = skills_utils.setup_spacy_model(model["nlp"])

# %%
job_ads = pd.read_csv("../data/raw/job_ads/sample_reed_extract.csv")

# %%
job_description_text = job_ads.iloc[9123].description
print(job_description_text)
skills_utils.detect_skills(clean_text(job_description_text), model, nlp)

# %%
