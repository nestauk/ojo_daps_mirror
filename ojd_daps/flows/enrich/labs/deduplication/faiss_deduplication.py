"""
deduplication.faiss_deduplication
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

from ojd_daps.orms.raw_jobs import JobAdDescriptionVector

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

# functions to download vectors

def prefill_inputs(orm):
    """For performance, preallocate numpy arrays to be filled later.
    Numpy array size to be determined dynamically, based on vector dimensions
    from the DB and count of vectors in the DB.
    """
    # Determine the "height" and "width" of the array
    # by asking the database
    count = session.query(orm).count()  # "Height" of array
    a_vector, = session.query(orm.vector).limit(1).one()
    dim = len(a_vector) # "Width" of array
    # Preallocate space
    data = np.empty((count, 768), dtype=FLOAT_TYPE)
    ids = np.empty((count, ), dtype=STR_TYPE)
    return data, ids

def query_and_bundle(session, fields, offset, limit, filter_):
    """Query the database for a list of SqlAlchemy fields,
    and apply limits, offsets and filters as required. The results
    are bundled into numpy arrays."""
    q = session.query(*fields)  # raw query
    q = q.offset(offset) if filter_ is None else q.filter(filter_)  # filter / offset
    ids, vectors = zip(*q.limit(limit))  # unravel results
    # bundle into arrays
    _ids = np.array(ids, dtype=STR_TYPE)
    _str_vectors = [json.loads(vector) for vector in vectors]
    _vectors = np.array(_str_vectors, dtype=FLOAT_TYPE)
    return _ids, _vectors

def read_data(data, ids, orm, id_field,
            chunksize=10000, max_chunks=None):
    """Read data into the data and id arrays,
    always starting from the last read chunk (e.g. if connection fails).
    Data is read using the last available id,
    since filtering is much faster than offsetting, for large datasets.
    """
    id_field = getattr(orm, id_field)
    fields = (id_field, orm.vector)
    count, _ = data.shape
    empty_ids = (ids != '')
    offset = sum(empty_ids)  # resume if already started
    # Calculate a filter statement, since these are faster than OFFSET
    filter_ = None if offset == 0 else id_field > ids[offset-1]
    # Set default values of {n,max}_chunks
    n_chunks = -1 if max_chunks is None else 0
    max_chunks = 0 if max_chunks is None else max_chunks
    # Start or continue collecting filling the data and id arrays
    while offset < count:
        if n_chunks >= max_chunks:
            # Note: this never happens if max_chunks is set to default
            break
        if offset % 10*chunksize == 0:
            logging.info(f"Collecting row {offset+1} of {count}")
        # Query the database and bundle the results into intermediate arrays
        limit = chunksize if offset + chunksize < count else None
        _ids, _data = query_and_bundle(session, fields, offset, limit, filter_)
        # Update the preallocated arrays
        ids[offset:offset+_ids.shape[0]] = _ids
        data[offset:offset+_data.shape[0]] = _data
        # Update the filter/offset criteria
        filter_ = id_field > _ids[-1]
        offset += chunksize
        # Increment the number of chunks we've processed thus far
        if max_chunks > 0:
            # Note: this never happens if max_chunks is set to default
            n_chunks += 1

def download_vectors(orm, id_field,
                    chunksize=10000, max_chunks=None):
    """Download vectors from the DB"""
    data, ids = prefill_inputs(orm)  # Empty numpy arrays
    while "reading data":
        try:
            # Start or continue reading
            read_data(data=data, ids=ids, orm=orm,
                    id_field=id_field,
                    chunksize=chunksize,
                    max_chunks=max_chunks)
        # The following has only been found to happen if your
        # connection drops slightly, which corrupts the JSON
        except json.JSONDecodeError:
            n = sum(ids != '')  # Total docs so far
            continue  # Retry
        else:
            break  # Done
    # Truncate the results, to remove unallocated entries
    # (this happens when max_chunks is not None)
    done_rows = (ids != '')
    ids = ids[done_rows]
    data = data[done_rows]
    # Return
    return data, ids

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

# model function, and application function

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

# example of application of model

with db_session('dev') as session:
    save_model()
    data, ids = download_vectors(JobAdDescriptionVector,'id',chunksize=10, max_chunks=1)
    links = apply_model(data, ids)

print(links)