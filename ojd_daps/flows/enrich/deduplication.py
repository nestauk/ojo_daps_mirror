"""
deduplication_flow
-------------
A Flow for identifying duplicate job advert descriptions.
"""
# Required for batch
import os

os.system(
    f'pip install -r {os.path.dirname(os.path.realpath(__file__))}/requirements.txt 1> /dev/null'
    )
import json
import re
import numpy as np
import logging

from metaflow import FlowSpec, step, S3, Parameter, batch, resources
from datetime import datetime, timedelta

from ojd_daps.orms.raw_jobs import JobAdDescriptionVector, RawJobAd

from daps_utils import talk_to_luigi, db
from daps_utils.flow import DapsFlowMixin
from daps_utils.db import db_session, object_as_dict

import ojd_daps
from daps_utils import db
db.CALLER_PKG = ojd_daps
db_session = db.db_session

FLOAT_TYPE = np.float32
STR_TYPE = np.dtype('U40')
CHUNKSIZE = 20000
VECTOR_DIM = 768
INTERVAL = 56 # 8 weeks

def sliding_window_getter(job_ads, interval):
    windows = []
    max_date = max(job_ads, key=lambda x:x['created'])
    min_date = min(job_ads, key=lambda x:x['created'])
    window_start_date = min_date['created']
    while window_start_date < max_date['created']:
        windows.append([window_start_date, window_start_date+timedelta(days=interval)])
        window_start_date = window_start_date + timedelta(days=interval/2)
    return windows


def query_and_bundle(session, fields, offset, limit, filter_):
    """Query the database for a list of SqlAlchemy fields,
    and apply limits, offsets and filters as required. The results
    are bundled into numpy arrays."""
    q = session.query(*fields)  # raw query
    q = q.offset(offset) if filter_ is None else q.filter(filter_)  # filter / offset
    ids, vectors = zip(*q.limit(limit))  # unravel results
    vectors = list(map(json.loads, vectors))  # cast str-json to json
    # bundle into arrays
    _ids = []
    _vectors = []
    for i, element in enumerate(vectors):
        try:
            np.array(element, dtype=FLOAT_TYPE)
            _ids.append(ids[i])
            _vectors.append(element)
        except ValueError:
            _ids.append('')
            _vectors.append(np.zeros(VECTOR_DIM))
    _ids = np.array(_ids, dtype=STR_TYPE)
    _vectors = np.array(_vectors, dtype=STR_TYPE)
    return _ids, _vectors


def prefill_inputs(orm, session):
    """For performance, preallocate numpy arrays to be filled later.
    Numpy array size to be determined dynamically, based on vector dimensions
    from the DB and count of vectors in the DB.
    """
    # Determine the "height" and "width" of the array
    # by asking the database
    count = session.query(orm).count()  # "Height" of array
    # Preallocate space
    data = np.empty((count, VECTOR_DIM), dtype=FLOAT_TYPE)
    ids = np.empty((count, ), dtype=STR_TYPE)
    return data, ids


def read_data(
    data,
    ids,
    orm,
    id_field,
    session,
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
        ids[offset : offset + _ids.shape[0]] = _ids  # pylint: disable=E1136
        data[offset : offset + _data.shape[0]] = _data  # pylint: disable=E1136
        # Update the filter/offset criteria
        filter_ = id_field > _ids[-1]
        offset += chunksize
        # Increment the number of chunks we've processed thus far
        if max_chunks > 0:
            # Note: this never happens if max_chunks is set to default
            n_chunks += 1


def download_vectors(orm, id_field, database, session,
                    chunksize=10000, max_chunks=None):
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

@talk_to_luigi
class DeduplicationFlow(FlowSpec, DapsFlowMixin):

    @step
    def start(self):
        """
        Starts the flow.
        """
        self.next(self.get_vectors)

    @step
    def get_vectors(self):
        """
        Downloads the vectors
        """
        with db_session(database=self.db_name) as session:
            self.data, self.ids = download_vectors(orm=JobAdDescriptionVector,\
                id_field='id', database=self.db_name, session=session)
        self.next(self.get_windows)

    @step
    def get_windows(self):
        """Gets ads and generates 8 weekly overlapping windows
        """
        with db_session(database='production') as session:
            jobad_query = session.query(RawJobAd.id, RawJobAd.created)
            self.job_ads = [
                {"id": _id, "created": created}
                for _id, created in jobad_query
            ]
        self.ad_ids = [ad['id'] for ad in self.job_ads]
        self.ad_created = [ad['created'] for ad in self.job_ads]
        self.windows = sliding_window_getter(job_ads=self.job_ads, interval=INTERVAL)
        self.next(self.get_vector_ids, foreach='windows')

    @step
    def get_vector_ids(self):
        """For each window, finds indexes of vectors corresponding
        to the 8 week chunk
        """
        chunk = [i for i,x in enumerate(self.ad_created) if self.input[0] <= x <= self.input[1]] # find index of created dates in the window
        id_chunk = [self.ad_ids[i] for i in chunk] # get ids of job adverts in the chunk
        self.vector_id_chunk = [i for i,x in enumerate(self.ids.tolist()) if x in id_chunk] # get index of vectors corresponding to job ad ids
        self.next(self.join_vector_ids)

    @step
    def join_vector_ids(self, inputs):
        """Joins foreach outputs, persists ids and data
        """
        self.vector_id_chunks = [input.vector_id_chunk for input in inputs if len(input.vector_id_chunk) > 0]
        self.ids = inputs[0].ids
        self.data = inputs[0].data
        self.next(self.pre_find_similar_vectors)

    @step
    def pre_find_similar_vectors(self):
        """Dummy step to separate previous join from next for each
        """
        self.next(self.find_similar_vectors, foreach='vector_id_chunks')

    @step
    def find_similar_vectors(self):
        """
        Finds similar vectors
        """
        from labs.deduplication.faiss_utils import apply_model
        ids = np.array([self.ids[i] for i in self.input], dtype=STR_TYPE)
        data = np.array([self.data[i] for i in self.input], dtype=FLOAT_TYPE)
        self.similar_vectors = apply_model(data, ids)
        first_id = self.ids[0]
        last_id = self.ids[-1]
        data = [{"first_id": _id1, "second_id": _id2, "weight": weight}
            for _id1, sims in self.similar_vectors.items()
            for _id2, weight in sims.items()]
        filename = f'deduplication_{first_id}-{last_id}_test-{self.test}.json'
        with S3(run=self) as s3:
            data = json.dumps(data)
            url = s3.put(filename, data)
        self.next(self.dummy_join_step)

    @step
    def dummy_join_step(self, inputs):
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == '__main__':
    DeduplicationFlow()
