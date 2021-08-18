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
# # Defining language and general skills clusters

# %% [markdown]
# ## Import dependencies and data

# %%
from collections import Counter, defaultdict
from itertools import combinations
from gensim.models import Word2Vec
import igraph
from igraph import Graph
import pandas as pd
import numpy as np
import pickle 
import os

from ojd_daps.flows.enrich.labs.skills.skills_detection_utils import load_model
from ojd_daps.flows.enrich.labs.skills.helper_utils import DATA_PATH, get_esco_skills, save_lookup, get_lookup

# Percentile of core skills to manually check
PRCTILE = 99


# %%
def build_lookup_dicts(sf_df):
    """ Conversion between surface forms and an integer number (for building graphs) """
    # Surface form to an integer ID number
    sf_to_id = dict(zip(sf_df.surface_form.to_list(), list(range(len(sf_df)))))
    # Integer ID number to a surface form
    id_to_sf = dict(zip(list(range(len(sf_df))), sf_df.surface_form.to_list()))
    return sf_to_id, id_to_sf


def get_detected_surface_form_lists(job_skills):
    """ Get a list of lists of surface forms, each list corresponding to a job advert """
    # Lists of detected surface forms
    detected_surface_forms = []
    for j, detected_skills in enumerate(job_skills):
        if len(detected_skills) != 0:
            detected_surface_forms.append([s['surface_form'] for s in detected_skills])
    return detected_surface_forms

def get_all_surface_form_ids(detected_surface_forms, sf_to_id):
    detected_surface_form_ids = [[sf_to_id[sf] for sf in sf_list] for sf_list in detected_surface_forms]
    all_sf_ids = [sf for sf_list in detected_surface_form_ids for sf in sf_list]
    return all_sf_ids


def get_unique_surface_form_ids(detected_surface_forms, sf_to_id):
    all_sf_ids = get_all_surface_form_ids(detected_surface_forms, sf_to_id)
    unique_sf_ids = np.sort(np.unique(all_sf_ids))
    return unique_sf_ids


def get_all_surface_forms(detected_surface_forms):
    all_sf = [sf for sf_list in detected_surface_forms for sf in sf_list]
    return all_sf


def get_unique_surface_forms(detected_surface_forms):
    all_sf = get_all_surface_forms(detected_surface_forms)
    unique_sf = np.sort(np.unique(all_sf))
    return unique_sf.tolist()


def get_surface_form_counts(detected_surface_forms):
    """ Get total surface form counts, and the fraction of job adverts they have appeared in """
    all_sf = get_all_surface_forms(detected_surface_forms)
    unique_sf = get_unique_surface_forms(detected_surface_forms)
    
    sf_counts = Counter(all_sf)
    
    counts = np.array([sf_counts[sf] for sf in unique_sf])

    sf_counts_df = pd.DataFrame(data={
        'surface_form': sf_counts.keys(),
        'counts': sf_counts.values(),
        'fraction': np.array(list(sf_counts.values())) / len(detected_surface_forms)
    })
    
    return sf_counts_df


# %% [markdown]
# ## Preselect general transversal skills clusters
#
# - Collect language skills in a separate cluster
# - Collect "general skills" in another separate cluster
#   - Select these by using ESCO "transversal" skills category (manually review)
#   - In addition, check the most frequent skills across analysed job data
#   - In addition, check the most central/core skills across the jobs data

# %%
def export_ESCO_transversal_skills(skills):
    """ Export non-language transversal ESCO skills for manual review """
    skills[
        (skills.reuse_level=='transversal')
        & (skills.skill_category.isin(['L', 'L_use'])==False)
    ].to_csv(DATA_PATH / 'aux/general_skills_qual_check.csv')
    
def get_language_skills(skills):
    """ Get ESCO language skills entity ids """
    language_skills = skills[skills.skill_category.isin(['L', 'L_use'])]
    language_skills_ids = language_skills.id.to_list() 
    return language_skills_ids
    
def get_reviewed_ESCO_transversal_skills():
    """ Get manually reviewed transversal ESCO skills entity ids """
    return (pd.read_csv(DATA_PATH / 'aux/general_skills_qual_checked.csv')
            .query('manual_is_general_skill==1')
           ).id.to_list()

def build_cooc_graph(detected_surface_forms, unique_sf_ids, id_to_sf):
    """  Build a co-occurrence graph """
    
    # Co-occurrences from job data
    edge_counts = defaultdict(int)
    for id_list in detected_surface_forms:
        for combo in combinations(id_list, 2):
            edge_counts[tuple(sorted(combo))]+=1

    # Build the co-occurrence graph
    G = Graph(directed=False)
    G.add_vertices(
        n=len(unique_sf_ids),
        attributes = {
            'sf_id': list(unique_sf_ids),
            'name': [id_to_sf[i] for i in unique_sf_ids]
        }
    )

    G.add_edges(
        es = list(edge_counts.keys()),
        attributes = {
            'weight': [np.log(x) for x in list(edge_counts.values())]
        }
    )
    
    return G

def calculate_coreness(graph, weights=None, ignore_betweenness=False):
    """ Calculate 'coreness' (might take a few minutes for graphs with ~10k nodes) """
    # Eigenvector centrality, e
    centrality= graph.eigenvector_centrality(directed=False, scale=True, weights=weights)
    # Clustering coefficient, c
    clustering_coeff = graph.transitivity_local_undirected(weights=weights)
        
    if ignore_betweenness == False:
        # Betweenness centrality, b
        betweenness = graph.betweenness(directed=False, weights=weights)
        # Normalise betweenness centrality between 0 and 1, b_norm
        betweenness_norm = np.array(betweenness) / np.max(betweenness)
        # Average b_norm and e
        mean_centrality = 0.5*np.array(betweenness_norm) + 0.5*np.array(centrality)
    else:
        mean_centrality = centrality
        betweenness_norm = []
    
    # Multiply averaged centrality with (1-c) to select skills that are high on both measures
    m = mean_centrality*(1-np.array(clustering_coeff))
    # Deal with nulls
    m[np.where(np.isnan(m))[0]] = 0
    # Normalise between 0 and 1
    m = m / np.max(m)
    
    other_measures = {
        'ev_centrality': centrality,
        'betweenness_centrality': betweenness_norm,
        'clustering_coeff': clustering_coeff}
    return m, other_measures

def surface_form_measures(sf_df, detected_surface_forms):
    """ Calculate various surface form centrality and significance measures """
    # Lookups
    sf_to_id, id_to_sf = build_lookup_dicts(sf_df)
    # List of list of detected surface forms
    detected_surface_forms = get_detected_surface_form_lists(job_skills)
    # Unique forms IDs
    unique_sf_ids = get_unique_surface_form_ids(detected_surface_forms, sf_to_id)  
    
    print('Calculate surface form coreness')
    # Build a co-occurrence graph
    G = build_cooc_graph(detected_surface_forms, unique_sf_ids, id_to_sf)
    # Coreness and graph measures
    coreness, graph_measures = calculate_coreness(G, weights=None, ignore_betweenness=False)

    print('Calculate surface form counts')    
    # Calculate surface form counts 
    sf_counts_df = get_surface_form_counts(detected_surface_forms)
   
    # Get 'qualitative' general skills
    skills = get_esco_skills()
    language_skills_ids = get_language_skills(skills)    
    general_qual_skills_ids = get_reviewed_ESCO_transversal_skills()

    # Combine all measures
    sf_measures = sf_df.loc[unique_sf_ids].copy()
    sf_measures['coreness_data'] = coreness
    for key in graph_measures.keys():
        sf_measures[key] = graph_measures[key]

    sf_measures['coreness_data'] = coreness
    sf_measures=(sf_measures
                 .merge(sf_counts_df, how='left')
                )

    sf_measures['skill_class'] = 'other'
    sf_measures.loc[sf_measures.entity.isin(general_qual_skills_ids), 'skill_class'] = 'general_qual'
    sf_measures.loc[sf_measures.entity.isin(language_skills_ids), 'skill_class'] = 'language'

    sf_measures.sort_values('coreness_data', ascending=False)
    
    return sf_measures

def export_core_surface_forms_to_check(sf_measures, prct_to_check = 99, filename='general_skills_quant'):
    """ sf_measures = table to check, prct_to_check = percentile to check """
    # Export surface forms that have both high coreness and high frequency in the data
    general_sf_to_check = sf_measures[
        (sf_measures.coreness_data >= np.percentile(sf_measures.coreness_data, prct_to_check))
        & (sf_measures.fraction >= np.percentile(sf_measures.fraction, prct_to_check))
        & (sf_measures.skill_class=='other')
    ]
    fpath = DATA_PATH / f'aux/{filename}_check.csv'
    general_sf_to_check.to_csv(fpath)
    print(f'Surface form coreness measures saved in {fpath}')

    
def find_and_review_core_skills(sf_df, detected_surface_forms, prct_to_check=99, filename='general_skills_quant'):
    sf_measures = surface_form_measures(sf_df, detected_surface_forms)
    export_core_surface_forms_to_check(sf_measures, prct_to_check, filename)
    
def get_language_skills_surface_forms(sf_df):
    skills = get_esco_skills()
    language_skills_ids = get_language_skills(skills)
    language_skills_forms = sf_df[sf_df.entity.isin(language_skills_ids)].surface_form.to_list()
    return language_skills_ids, language_skills_forms


def get_general_skills_surface_forms():
    # General skills IDs
    general_qual_skills_ids = get_reviewed_ESCO_transversal_skills()
    
    general_sf_checked = pd.read_csv(DATA_PATH / 'aux/general_skills_quant_checked.csv')
    general_quant_skills_ids = general_sf_checked.query("is_manual_general_skill==1").entity.to_list()
    
    general_skills_ids = list(set(general_quant_skills_ids + general_qual_skills_ids))
    general_skills_forms = sf_df[sf_df.entity.isin(general_skills_ids)].surface_form.to_list()
    return general_skills_ids, general_skills_forms



# %%
if __name__ == "__main__":
    # Model that was used to analyse 1M skills
    model_v01_r1 = load_model('v01_r1', from_local=True)
    sf_df = model_v01_r1['surface_forms']
    # Most recent surface forms model
    skills_model = load_model('v01_1', from_local=True)
    
    # If file for manual review doesn't exist
    if not os.path.exists(DATA_PATH / 'aux/general_skills_quant_checked.csv'):
        
        # Import the job adverts
        filepath = DATA_PATH / 'processed/detected_skills/job_ads_DEC2020_MAY2021_detected_skills_v01_r1.pickle'
        job_skills = pickle.load(open(filepath,'rb'))

        # Export table for checking
        detected_surface_forms = get_detected_surface_form_lists(job_skills)
        find_and_review_core_skills(sf_df, detected_surface_forms, prct_to_check=PRCTILE, filename=FILENAME)  
    
    else:
        # Collect all checks
        language_skills_ids, language_skills_forms = get_language_skills_surface_forms(skills_model['surface_forms'])
        general_skills_ids, general_skills_forms = get_general_skills_surface_forms()

        # Save the manually reviewed clusters
        CLUSTERS_DIR = (DATA_PATH / 'processed/clusters')
        CLUSTERS_DIR.mkdir(parents=True, exist_ok=True) 

        language_cluster = {
            'language_surface_forms': language_skills_forms,
            'language_skills_entities': language_skills_ids
        }

        general_skills_cluster = {
            'general_skills_forms': general_skills_forms,
            'general_skills_ids': general_skills_ids
        }
        print(f"Saved language skills cluster data in {CLUSTERS_DIR}")
        save_lookup(language_cluster, CLUSTERS_DIR / 'language_cluster')
        print(f"Saved general skills cluster data in {CLUSTERS_DIR}")
        save_lookup(general_skills_cluster, CLUSTERS_DIR / 'general_skills_cluster')    
