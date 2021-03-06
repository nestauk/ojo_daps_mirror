"""
deduplication_flow
-------------
A Flow for identifying duplicate job advert descriptions.
"""
import json
import numpy as np
from functools import partial
from datetime import timedelta

from ojd_daps.orms.raw_jobs import JobAdDescriptionVector as Vector
from ojd_daps.orms.raw_jobs import RawJobAd

FLOAT_TYPE = np.float32
STR_TYPE = np.dtype("U40")
CHUNKSIZE = 10000
VECTOR_DIM = 768


def get_sliding_windows(start_date, end_date, interval):
    """
    Create a set of sliding window intervals between the given dates,
    with the successive windows sliding by half an interval. The interval
    is defined in days.
    """
    one_interval = timedelta(days=interval)
    half_interval = timedelta(days=interval / 2)
    windows = []
    while start_date < end_date:
        # Cast dates to str to enable `literal_binds` to work later
        windows.append([str(start_date), str(start_date + one_interval)])
        start_date += half_interval
    return windows


def query_and_bundle(session, query):
    """
    Query the database for the job ad IDs and vectors, and then bundle the
    results into numpy arrays.
    """
    ids, vectors = zip(*session.execute(query).fetchall())  # Unravel results
    vectors = list(map(json.loads, vectors))  # Cast str-json to json
    # Bundle into arrays
    _ids = []
    _vectors = []
    for i, element in enumerate(vectors):
        try:
            np.array(element, dtype=FLOAT_TYPE)
        except ValueError:
            _ids.append("")
            _vectors.append(np.zeros(VECTOR_DIM))
        else:
            _ids.append(ids[i])
            _vectors.append(element)
    _ids = np.array(_ids, dtype=STR_TYPE)
    _vectors = np.array(_vectors, dtype=FLOAT_TYPE)
    return _ids, _vectors


def prefill_inputs(count):
    """
    For performance, preallocate numpy arrays to be filled later.
    Numpy array size to be determined dynamically
    """
    # Preallocate space
    data = np.empty((count, VECTOR_DIM), dtype=FLOAT_TYPE)
    ids = np.empty((count,), dtype=STR_TYPE)
    return data, ids


def create_base_query(session, fields, start_date, end_date):
    """
    Generate a standard query for vectors in a sliding window.
    """
    query = session.query(*fields)
    query = query.join(RawJobAd, Vector.id == RawJobAd.id)
    query = query.filter(RawJobAd.created.between(start_date, end_date))
    return query


def read_data(data, ids, session, queries):
    """
    Iterate over prewritten SQL queries to read data into the preallocated
    `data` and `ids` arrays.
    """
    offset = 0
    for query in queries:
        _ids, _data = query_and_bundle(session, query)
        # Update the preallocated arrays
        ids[offset : offset + _ids.shape[0]] = _ids  # pylint: disable=E1136
        data[offset : offset + _data.shape[0]] = _data  # pylint: disable=E1136
        # Update the filter/offset criteria
        offset += CHUNKSIZE


def query_to_str(query):
    """
    Convert a SqlAlchemy query to a raw SQL query, with all parameters
    parsed into the raw query ("literal_binds").
    """
    return str(query.statement.compile(compile_kwargs={"literal_binds": True}))


def download_vectors(session, queries, count, max_errors=100):
    """
    Download vectors from the DB, taking into account the possibility
    of mysterious json.JSONDecodeErrors
    """
    data, ids = prefill_inputs(count)  # Empty numpy arrays
    n_errors = 0
    while "reading data":
        try:
            # Start or continue reading
            read_data(data=data, ids=ids, session=session, queries=queries)
        # The following transient error can happen, unknown reason
        except json.JSONDecodeError:
            n_errors += 1
            if n_errors == max_errors:
                raise
            continue  # Retry
        else:
            break  # Done
    return data, ids


def generate_window_query_chunks(session, start_date, end_date):
    """
    Generate every JobAdDescriptionVector query between two dates.

    The method here is to get the PKs at each interval, in order to
    construct queries for job ads in a PK range.
    """
    _create_query = partial(  # create_base_query is called twice, so use partial
        create_base_query,
        session=session,
        start_date=start_date,
        end_date=end_date,
    )
    # Get the query for this window
    vector_query = _create_query(fields=(Vector.id, Vector.vector))
    # Get the counts, to be used later for prefilling inputs
    ids_query = _create_query(fields=(Vector.id,))
    count = ids_query.count()  # Total number of job ad vectors
    # Generate every query in chunks
    raw_queries = []
    total = 1  # For while-loop book keeping
    (lower_pk,) = ids_query.limit(1).one()  # Get the first PK, for the first window
    while total < count:
        is_final_query = count - total < CHUNKSIZE
        # Find the next PK for this chunk
        above = Vector.id >= lower_pk
        offset = None if is_final_query else CHUNKSIZE
        (upper_pk,) = ids_query.filter(above).offset(offset).limit(1).one()
        # Generate a query for vectors is `above` the lower PK and `below` the upper PK
        below = Vector.id < upper_pk
        query = vector_query.filter(above)
        if not is_final_query:  # Don't need `below` filter for the final chunk
            query = query.filter(below)
        # Convert the SqlAlchemy query to a raw SQL query
        raw_queries.append(query_to_str(query))
        # Swap in the new lower_pk
        lower_pk = upper_pk
        total += CHUNKSIZE
    return raw_queries, count
