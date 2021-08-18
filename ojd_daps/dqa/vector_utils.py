"""
vectors_utils
=============

Read vector data from MySQL, assuming the given
schema has an ID field and a field called "vector".
For large number of vectors, this can be slow and
also very memory inefficient. There are two main
bottlenecks which are overcome in this package:

1) LIMIT / OFFSET is slower than filtering by sequential ids
2) Creating lists and then arrays is slower and more memory intensive than preallocating arrays with thoughtful types.
"""

import numpy as np
import logging
import json

# The types used for preallocating space
FLOAT_TYPE = np.float32
STR_TYPE = np.dtype("U40")


def query_and_bundle(session, fields, offset, limit, filter_):
    """Query the database for a list of SqlAlchemy fields,
    and apply limits, offsets and filters as required. The results
    are bundled into numpy arrays."""
    q = session.query(*fields)  # raw query
    q = q.offset(offset) if filter_ is None else q.filter(filter_)  # filter / offset
    ids, vectors = zip(*q.limit(limit))  # unravel results
    vectors = tuple(map(json.loads, vectors))  # cast str-json to json
    # bundle into arrays
    _ids = np.array(ids, dtype=STR_TYPE)
    _vectors = np.array(vectors, dtype=FLOAT_TYPE)
    return _ids, _vectors


def prefill_inputs(orm, session):
    """For performance, preallocate numpy arrays to be filled later.
    Numpy array size to be determined dynamically, based on vector dimensions
    from the DB and count of vectors in the DB.
    """
    count = session.query(orm).count()  # "Height" of array
    (a_vector,) = session.query(orm.vector).limit(1).one()
    a_vector = json.loads(a_vector)
    dim = len(a_vector)  # "Width" of array
    # Preallocate space
    data = np.empty((count, dim), dtype=FLOAT_TYPE)
    ids = np.empty((count,), dtype=STR_TYPE)
    return data, ids


def read_data(
    data,
    ids,
    orm,
    session,
    id_field,
    chunksize=10000,
    max_chunks=None,
):
    """Read data into the data and id arrays,
    always starting from the last read chunk (e.g. if connection fails).
    Data is read using the last available id,
    since filtering is much faster than offsetting, for large datasets.
    """
    id_field = getattr(orm, id_field)
    fields = (id_field, orm.vector)
    count, _ = data.shape
    empty_ids = ids != ""
    offset = sum(empty_ids)  # resume if already started
    # Calculate a filter statement, since these are faster than OFFSET
    filter_ = None if offset == 0 else id_field > ids[offset - 1]
    # Set default values of {n,max}_chunks
    n_chunks = -1 if max_chunks is None else 0
    max_chunks = 0 if max_chunks is None else max_chunks
    # Start or continue collecting filling the data and id arrays
    while offset < count:
        if n_chunks >= max_chunks:
            # Note: this never happens if max_chunks is set to default
            break
        if offset % 10 * chunksize == 0:
            logging.info(f"Collecting row {offset+1} of {count}")
        # Query the database and bundle the results into intermediate arrays
        limit = chunksize if offset + chunksize < count else None
        _ids, _data = query_and_bundle(session, fields, offset, limit, filter_)
        # Update the preallocated arrays
        ids[offset : offset + _ids.shape[0]] = _ids
        data[offset : offset + _data.shape[0]] = _data
        # Update the filter/offset criteria
        filter_ = id_field > _ids[-1]
        offset += chunksize
        # Increment the number of chunks we've processed thus far
        if max_chunks > 0:
            # Note: this never happens if max_chunks is set to default
            n_chunks += 1


def download_vectors(orm, id_field, session, chunksize=10000, max_chunks=None):
    """Download vectors from the DB"""
    data, ids = prefill_inputs(orm, session)  # Empty numpy arrays
    while "reading data":
        try:
            # Start or continue reading
            read_data(
                data=data,
                ids=ids,
                orm=orm,
                session=session,
                id_field=id_field,
                chunksize=chunksize,
                max_chunks=max_chunks,
            )
        # The following has only been found to happen if your
        # connection drops slightly, which corrupts the JSON
        except json.JSONDecodeError:
            continue  # Retry
        else:
            break  # Done
    # Truncate the results, to remove unallocated entries
    # (this happens when max_chunks is not None)
    done_rows = ids != ""
    ids = ids[done_rows]
    data = data[done_rows]
    # Return
    return data, ids
