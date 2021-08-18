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
# # Clustering of surface forms
# To do:
# - Better naming of variables
# - Factoring out clustering parameters into a config file
# - Adding logging.info messages
# - Improve docstrings

# %%
from gensim.models import Word2Vec
import pandas as pd
import numpy as np
import pickle 
import os

from ojd_daps.flows.enrich.labs.skills.skills_detection_utils import load_model, save_model_locally, save_model_in_s3
from ojd_daps.flows.enrich.labs.skills.helper_utils import (
    DATA_PATH, get_esco_skills, save_lookup, get_lookup
)
from ojd_daps.flows.enrich.labs.skills.notebooks.PIPELINE_general_skills import ( 
    build_lookup_dicts,
    get_detected_surface_form_lists,
    get_all_surface_form_ids,
    get_unique_surface_form_ids,
    get_all_surface_forms,
    get_unique_surface_forms,
    get_surface_form_counts
)
from ojd_daps.flows.enrich.labs.skills.notebooks.PIPELINE_assess_quality import (
    get_manual_review_results
)

import ojd_daps.flows.enrich.labs.skills.cluster_utils as cluster_utils
# import ojd_daps.flows.enrich.labs.skills.cluster_profiling_utils as cluster_profiling_utils
from scipy.spatial.distance import pdist, squareform

        
# Clustering results folder
CLUSTERS_DIR = (DATA_PATH / 'processed/clusters')
CLUSTERS_DIR.mkdir(parents=True, exist_ok=True) 

# %%
# Keep only surface forms that have appeared at least 10 times
FREQ_THRESH = 10

# CLUSTERING PARAMS
# Name of this clustering session
session_name = 'sf_clusters_v01_1qc'
# Number of nearest neighbours used for the graph construction
nearest_neighbours = [10, 15, 20, 25]
# Ensemble size for the first step
N = 1000
# Ensemble size for the consensus step
N_consensus = 100
# Number of clustering trials for each nearest neighbour value
N_nn = N // len(nearest_neighbours)
# Which clusters to break down from the partition
clusters = 'all' # Either a list of integers, or 'all'
# Path to save the clustering results
fpath_ = DATA_PATH / f'interim/clustering/{session_name}/'
fpath_.mkdir(parents=True, exist_ok=True) 
fpath = f'{fpath_}/'

clustering_params = {
    'N': N,
    'N_consensus': N_consensus,
    'N_nn': N_nn,
    'clusters': clusters,
    'fpath': fpath,
    'session_name': session_name,
    'nearest_neighbours': nearest_neighbours}


# %%
def create_clustering_dataset(sf_df, manual_review, detected_surface_forms):
    """ Select the data points for clustering
    Note: sf_df should contain quality rating predictions
    manual_review should contain data about manual labels (1, 3)
    
    ### Cluster surface form vectors

    - Remove surface forms that are very rare
    - Remove surface forms that are in general or language skills cluster
    - Remove surface forms that have been predicted to be 'low quality'
        - But make sure that manually verified surface forms are kept    
    """

    # Surface form counts in the job ads
    sf_counts = get_surface_form_counts(detected_surface_forms)

    # Set of unique surface forms, detected in the job ads
    unique_sf = get_unique_surface_forms(detected_surface_forms)

    # Low frequency surface forms
    low_freq_forms = sf_counts.query(f"counts<{FREQ_THRESH}").surface_form.to_list()

    # Preselected clusters
    lang_skills = get_lookup(CLUSTERS_DIR / 'language_cluster')
    gen_skills = get_lookup(CLUSTERS_DIR / 'general_skills_cluster')

    # General skills surface forms
    lang_skill_forms = sf_counts[sf_counts.surface_form.isin(lang_skills["language_surface_forms"])].surface_form.to_list()
    gen_skills_forms = sf_counts[sf_counts.surface_form.isin(gen_skills["general_skills_forms"])].surface_form.to_list()

    # Surface forms that are predicted to not be OK
    predicted_is_OK_forms = sf_df.query('is_predicted_OK==1').surface_form.to_list()
    manual_is_OK_forms = manual_review.query('manual_OK==3').surface_form.to_list()
    manual_not_OK_forms = manual_review.query('manual_OK==1').surface_form.to_list()

    # Surface forms that are not included in the most recent model
    excluded_forms = sf_counts[sf_counts.surface_form.isin(sf_df.surface_form.to_list())==False].surface_form.to_list()

    # Select forms to cluster
    list_of_forms_to_remove = [
        low_freq_forms,
        lang_skill_forms,
        gen_skills_forms,
        manual_not_OK_forms,
        excluded_forms
    ]

    sf_to_cluster = sf_counts.copy()
    for forms_to_remove in list_of_forms_to_remove:
        sf_to_cluster = remove_surface_forms(sf_to_cluster, forms_to_remove)

    sf_to_cluster = sf_to_cluster[
        sf_to_cluster.surface_form.isin(predicted_is_OK_forms)
        | sf_to_cluster.surface_form.isin(manual_is_OK_forms)
    ]
    
    sf_to_cluster = (sf_to_cluster
                     .sort_values('surface_form')
                     .merge(sf_df[['surface_form', 'surface_form_type', 'entity', 'preferred_label']], how='left')
                    )
    
    return sf_to_cluster


def surface_form_2_vec(detected_surface_forms, use_cached=True):
    """ Surface form to vector model (sf2vec) """
    
    filepath = f"models/sf2vec_{clustering_params['session_name']}.model"
    fpath = f'{DATA_PATH.parent / filepath}'
    
    if os.path.exists(fpath) and use_cached:
        model = Word2Vec.load(fpath)
    else:
        # Get skills per job, and determine the max window size
        n_skills_per_job = [len(x) for x in detected_surface_forms]
        max_window = max(n_skills_per_job)

        # Build the model
        model = Word2Vec(sentences=detected_surface_forms, 
                         size=200, 
                         window=max_window, 
                         min_count=1, 
                         workers=4, 
                         sg=1, 
                         seed=123, 
                         iter=30)

        model.save(fpath)
    return model

                     
def get_sf2vec_vectors(model, unique_sf):
    """ extract sf2vec vectors from model """
    skill2vec_emb = []
    for sf in unique_sf:
        emb = model.wv[sf] 
        skill2vec_emb.append(emb)
    skill2vec_emb = np.array(skill2vec_emb)
    return skill2vec_emb
     
                     
# Create the word2vec model
def get_sf2vec_embeddings(detected_surface_forms, sf_to_cluster):
    cluster_forms = sf_to_cluster.surface_form.to_list()
    sf2vec_model = surface_form_2_vec(detected_surface_forms)
    sf2vec_embeddings = get_sf2vec_vectors(sf2vec_model, cluster_forms)
    return sf2vec_embeddings
                     

def remove_surface_forms(df, forms_to_remove):
    df = df[df.surface_form.isin(forms_to_remove)==False]
    return df



# %%
def add_preselected_clusters(partitions):
    n_clust_lev1 = partitions.level_1.max()
    n_clust_lev2 = partitions.level_2.max()

    # Preselected clusters
    lang_skills = get_lookup(CLUSTERS_DIR / 'language_cluster')
    gen_skills = get_lookup(CLUSTERS_DIR / 'general_skills_cluster')   
    
    LANG_CLUST_1 = n_clust_lev1 + 1
    LANG_CLUST_2 = n_clust_lev2 + 1
    GENERAL_CLUST_1 = n_clust_lev1 + 2
    GENERAL_CLUST_2 = n_clust_lev2 + 2 
    
    partitions_full = pd.concat([
        partitions[['level_1', 'level_2', 'surface_form']],
        assign_manual_clusters(lang_skills["language_surface_forms"],LANG_CLUST_1,LANG_CLUST_2),
        assign_manual_clusters(gen_skills["general_skills_forms"],GENERAL_CLUST_1,GENERAL_CLUST_2)
    ])
    return partitions_full

def assign_manual_clusters(sf, CLUST_1, CLUST_2):
    n = len(sf)
    return pd.DataFrame(
        (list(zip([CLUST_1]*n, [CLUST_2]*n, sf))),
        columns = ['level_1', 'level_2', 'surface_form']
    )

def update_skills_model(skills_model, surface_form_partitions_full):
    new_model = skills_model.copy()
    new_model_sf_df = (new_model['surface_forms']
                    .merge(
                        surface_form_partitions_full[['entity', 'surface_form','level_1', 'level_2']], 
                        how='left', 
                        on=['surface_form', 'entity'])
                    .rename(columns={
                        'level_1': 'cluster_1',
                        'level_2': 'cluster_2'
                    }))
    new_model['surface_forms'] = new_model_sf_df
    new_model['name'] = f"{skills_model['name']}c"
    save_model_locally(new_model)
    save_model_in_s3(new_model)
    return new_model


def clustering_pipeline(sf_to_cluster, sim_matrix, clustering_params):

    nodes = sf_to_cluster.copy()
    clust_ids = list(range(len(sf_to_cluster)))
    nodes['clust_id'] = clust_ids
    ### Clustering: Level 1

    # Prepare and save the Level-0 partition file (all in one cluster)
    partition_df = pd.DataFrame()
    partition_df['id'] = clust_ids
    partition_df['cluster'] = [0] * len(nodes)
    partition_df.to_csv(fpath_ / f'{session_name}_clusters_Level0.csv', index=False)

    # Set the random_state variable for reproduciblity
    clustering_params['random_state'] = 14523

    # Perform the clustering
    cluster_utils.subcluster_nodes(W=sim_matrix, l=0, **clustering_params)
    # Collect subclusters into one partition
    partition_1 = cluster_utils.collect_subclusters(0, fpath, session_name, n_total=len(clust_ids))
    
    ############
    print(cluster_utils.ConsensusClustering.describe_partition(partition_1.cluster.to_list()))
    ############
    
    ### Clustering: Level 2

    # Load the partition that we wish to further split apart
    partition = pd.read_csv(fpath + session_name + '_clusters_Level1.csv')
    # Set the random_state variable for reproduciblity
    clustering_params['random_state'] = 1
    # Perform the clustering
    cluster_utils.subcluster_nodes(W=sim_matrix, l=1, **clustering_params)
    # Adjust the clustering labels and save
    partition_2 = cluster_utils.collect_subclusters(1, fpath, session_name, n_total=len(clust_ids))

    ### Clustering: Combine all partitions
    partition_1 = pd.read_csv(fpath + session_name + '_clusters_Level1.csv')
    partition_2 = pd.read_csv(fpath + session_name + '_clusters_Level2.csv')

    # Create a dataframe with all three partitions
    partitions = partition_1.merge(partition_2, on='id')
    partitions = partitions.rename(columns={'cluster_x': 'level_1', 'cluster_y': 'level_2'})

    # Relabel Level 2 clusters to match the ordering of Level 1 clusters
    partitions = partitions.sort_values(['level_1','level_2'])
    level_2_labels = partitions.drop_duplicates('level_2').level_2.to_list()
    level_2_new_labels = list(range(len(level_2_labels)))
    relabel_dict = dict(zip(level_2_labels, level_2_new_labels))
    partitions.level_2 = partitions.level_2.apply(lambda x: relabel_dict[x])

    #######
    print(len(partitions))
    #######
    
    partitions = (partitions
                  .merge(nodes, 
                         left_on='id', 
                         right_on='clust_id', 
                         #validate="1:1", 
                         how='left')
                  .drop(['id', 'fraction'], axis=1)
                  .drop_duplicates('clust_id', keep='first') # just in case (in principle shouldn't happen)
                  .sort_values(['level_1','level_2'])
                 )
    partitions.to_csv(DATA_PATH / f'processed/clusters/{session_name}.csv')

    return partitions



# %%
if __name__ == "__main__":
    
    # Import the job adverts
    filepath = DATA_PATH / 'processed/detected_skills/job_ads_DEC2020_MAY2021_detected_skills_v01_1.pickle'
    job_skills = pickle.load(open(filepath,'rb'))

    # Surface forms that have been manually reviewed
    manual_review = get_manual_review_results()

    # Detected surface forms
    detected_surface_forms = get_detected_surface_form_lists(job_skills)
    
    # Most recent model
    skills_model = load_model('v01_1q', from_local=True)
    sf_df = skills_model['surface_forms']   
        
    # Select forms to cluster
    sf_to_cluster = create_clustering_dataset(sf_df, manual_review, detected_surface_forms)
    
    # Get vector representations of the forms for clustering
    print('Creating surface form embeddings')
    sf2vec_embeddings = get_sf2vec_embeddings(detected_surface_forms, sf_to_cluster)
    
    # Clustering
    print('Clustering')
    sim_matrix = 1-squareform(pdist(sf2vec_embeddings, metric='cosine'))
    partitions = clustering_pipeline(sf_to_cluster, sim_matrix, clustering_params)

    # Adding manual clusters, additional data and saving
    partitions_full = (add_preselected_clusters(partitions)
                       .merge(skills_model['surface_forms'], how='left')
                      )    
    partitions_full.to_csv(DATA_PATH / f'processed/clusters/{session_name}_full.csv')

    # Updating and saving the model
    new_model = update_skills_model(skills_model, partitions_full)
    




# %%
