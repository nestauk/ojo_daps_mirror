"""
deduplication.faiss_utils
-------------------------

    Find similar vectors using a FAISS index. The methodology
assumes very high-dimensionality vectors which, almost by
definition, tend to occupy very sparse spaces. For this reason, 
it is the L1 (Manhattan distance) metric which is used by 
default, although this can be overwritten. The main use cases 
are for finding near duplicates, and otherwise "contextually"
similar vectors.
A score is returned, which is defined relative to the `k_large`
nearest neighbours. To understand this, you should consider
that we make no assumptions about:
a) the lumpiness (i.e. density) of the vector space, in that 
   we assume that the space may be arbitarily lumpy; and
b) a consistent definition of the "physical" interpretation 
   of the density of any particular region of the vector space.
   This is to say that the space is not assumed to be flat.
Sampling only the `k_large` nearest neighbours therefore
means that each region local to each vector is treated
independently from one another. This then allows for a consistent
definition of similarity, in terms of the mean distance
of the `k_large` nearest neighbours to a "query" vector,
such that vectors close to the mean distance have a score of zero
and the vectors close to the query vector have a score of one
(noting, of course, that negative scores are possible but
uninteresting by definition).
Clearly if `k_large` is too large then the assumption a) falls apart
and if too small then assumption b) falls apart. `k_large` should
ideally be tuned to be roughly the lower limit of how the
neighbourhood of "contextually" similar vectors is. One might expect,
in the case of a million text documents, that such "soft clusters" 
would have at least 1000 members and so the default value of 
`k_large` has been set at 1000.
    """

import numpy as np
import logging
import boto3
import json
import faiss
import importlib
from functools import lru_cache

############
# Following lines are needed until issue is fixed in daps_utils
from daps_utils import db
import ojd_daps
db.CALLER_PKG = ojd_daps
db_session = db.db_session
############

FLOAT_TYPE = np.float32
STR_TYPE = np.dtype('U40')
S3_PATH = 'labs/deduplication/faiss/{}'
BUCKET_NAME = 'open-jobs-lake'

# functions to save to/load from S3
def save_to_s3(filename, contents):
    """Saves the contents to the filename in {BUCKET_NAME}/{S3_PATH}"""
    s3 = boto3.resource('s3')
    obj = s3.Object(BUCKET_NAME, S3_PATH.format(filename))
    obj.put(Body=contents)

def load_from_s3(filename):
    """Loads the file contents from the filename at {BUCKET_NAME}/{S3_PATH}"""
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=S3_PATH.format(filename))
    return obj['Body'].read().decode()


# functions to save/load model parameters

def save_model(k='20', k_large='1000',
                n_clusters='250', score_threshold='0.8', metric='METRIC_L1'):
    """Save the model config to s3.

    Args:
        orm (sqlalchemy.Base): A SqlAlchemy ORM, representing the table of vectors
        id_field (str): The name of the id field in the ORM.
        k (int): The maximum number of duplicates that can be found. (default=20)
        k_large (int): The sample size of "background" neighbour documents, for
                    quantifying similarity. The larger this number is, the
                    looser the definition is of "near" duplicates, and
                    so more results will be returned; although it will
                    have no impact on the number of exact duplicates.
                    (default=1000)
        metric (faiss.METRIC*): The distance metric for faiss to use.
                                (default=faiss.METRIC_L1)
        score_threshold (float): See above for definition. (default=0.8)
    """
    save_to_s3('k.txt', k)
    save_to_s3('k_large.txt', k_large)
    save_to_s3('n_clusters.txt', n_clusters)
    save_to_s3('score_threshold.txt', score_threshold)
    save_to_s3('metric.txt', metric)

def load_model(data, ids):
    """Loads the model"""
    k = int(load_from_s3('k.txt'))
    k_large = int(load_from_s3('k_large.txt'))
    n_clusters = int(load_from_s3('n_clusters.txt'))
    score_threshold = float(load_from_s3('score_threshold.txt'))
    metric = load_from_s3('metric.txt')
    metric = class_for_name('faiss', metric)
    return find_similar_vectors(data, ids, k, k_large, n_clusters, metric, score_threshold)

def class_for_name(module_name, class_name):
    # load the module, will raise ImportError if module cannot be loaded
    m = importlib.import_module(module_name)
    # get the class, will raise AttributeError if class cannot be found
    c = getattr(m, class_name)
    return c

# function to find similar vectors, and the appy_model function

def find_similar_vectors(data, ids, k, k_large,
                        n_clusters,
                        metric, score_threshold):
    """Returns a lookup of similar vectors, by ID.
    Similarity is determined by the given metric parameter. For high-dim
    vectors, such as those generated by BERT transformers
    this should be faiss.METRIC_L1. Explicitly, documents with
        (mean(D_large) - D) / mean(D_large) > duplicate_threshold
    are counted as "duplicates" of each other, where D_large is a vector of
    distances of the k_large nearest neighbours, and D is a vector of
    distances of the k nearest neighbours.
    Args:
        data (np.array): An array of vectors.
        ids (np.array): An array of id fields.
        k (int): The maximum number of duplicates that can be found.
        k_large (int): The sample size of "background" neighbour documents, for
                    quantifying similarity. The larger this number is, the
                    looser the definition is of "near" duplicates, and
                    so more results will be returned; although it will
                    have no impact on the number of exact duplicates.
        metric (faiss.METRIC*): The distance metric for faiss to use.
        score_threshold (float): See above for definition.
    """
    n, d = data.shape
    k = n if k > n else k
    k_large = n if k_large > n else k_large
    n_clusters = n if n < n_clusters else n_clusters
    quantizer = faiss.IndexFlat(d, metric)
    index = faiss.IndexIVFFlat(quantizer, d, n_clusters)
    index.train(data)
    index.add(data)

    # Make an expansive search to determine the base level of
    # similarity in this space as the mean similarity of documents
    # in the close vicinity
    index.nprobe = 100
    D, I = index.search(data, k_large)
    base_similarity = D.mean(axis=1)  # Calculate the mean distance

    # Now subset only the top k results
    D = D[:,:k]  # Distances
    I = I[:,:k]  # Indexes of the k results

    # Extract similar vectors
    similar_vectors = {}
    for _id, all_ids, sims, base in zip(ids, ids[I], D, base_similarity):
        _id = str(_id)  # FAISS returns ids as strings
        scores = (base - sims) / base
        over_threshold = scores > score_threshold
        # If no similar results, noting that the query vector is always
        # found so there will always be one result
        if over_threshold.sum() <= 1:
            continue
        results = {i: float(s) for i, s in zip(all_ids, scores)
                if s > score_threshold  # Ignore low scores
                and _id != i  # Ignore the query vector itself
                and i not in similar_vectors}  # Don't duplicate results
        # Possible that there are no similar vectors,
        # depending on the score_threshold
        if len(results) == 0:
            continue
        similar_vectors[_id] = results
    return similar_vectors

def apply_model(data, ids):
    """Loads and applies the model, returning links"""
    similar_vectors = load_model(data, ids)
    return similar_vectors
