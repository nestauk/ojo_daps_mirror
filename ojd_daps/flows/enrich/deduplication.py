"""
deduplication_flow
-------------
A Flow for identifying duplicate job advert descriptions.
"""
import json

from daps_utils import DapsFlowMixin

from metaflow import FlowSpec, S3, batch, pip, step

from ojd_daps.orms.raw_jobs import RawJobAd

from sqlalchemy import func


CHUNKSIZE = 10000
INTERVAL = 56  # 8 weeks
LIMIT = 2  # number of windows in testing


class DeduplicationFlow(FlowSpec, DapsFlowMixin):
    @step
    def start(self):
        self.next(self.get_windows)

    @step
    def get_windows(self):
        """Gets ads and generates 8 weekly overlapping windows"""
        from ojd_daps.flows.enrich.deduplication_utils import get_sliding_windows

        # Read from "production" to guarantee a meaningful sample in test mode
        with self.db_session(database="production") as session:
            # Get the earliest start and latest end date
            query = session.query(
                func.min(RawJobAd.created), func.max(RawJobAd.created)
            )
            ((start_date, end_date),) = query.all()
        # Generate sliding windows over the start and end date
        interval = INTERVAL / 2 if self.test else INTERVAL
        self.windows = get_sliding_windows(
            start_date=start_date, end_date=end_date, interval=interval
        )
        # Limit number of windows in test mode
        if self.test:
            self.windows = self.windows[:LIMIT]
        self.next(self.generate_window_queries)

    @step
    def generate_window_queries(self):
        """
        For each time window, generate a set of filter/offset queries
        which will be used in the foreach/batch step (i.e. find_similar_vectors)
        to read the data for that window in chunks. Note that for each window,
        the output contains the set of queries `raw_queries` and also `count`,
        which is the number of job ads in window, and is required for preallocating
        the data and ID arrays.
        """
        from ojd_daps.flows.enrich.deduplication_utils import (
            generate_window_query_chunks,
        )

        self.window_queries = []
        with self.db_session(database="production") as session:
            for start_date, end_date in self.windows:
                raw_queries, count = generate_window_query_chunks(
                    session, start_date, end_date
                )
                self.window_queries.append((raw_queries, count))
        self.next(self.find_similar_vectors, foreach="window_queries")

    @batch(cpu=8, memory=48000)
    @pip(path="requirements_dedup.txt")
    @step
    def find_similar_vectors(self):
        """
        Reads vectors from the DB and then finds similar vectors
        """
        from labs.deduplication.faiss_utils import apply_model
        from ojd_daps.flows.enrich.deduplication_utils import download_vectors

        # Read vectors from the database
        queries, count = self.input
        print("Downloading", count, "vectors...")
        with self.db_session(database="production") as session:
            data, ids = download_vectors(session, queries, count)
        assert len(list(ids.flatten())) == count  # Sanity check (internal consistency)

        # Find the similar vectors
        print("Applying model to find similar vectors")
        similar_vectors = apply_model(data, ids)
        # Reformat the data for saving
        print("Reformatting data for saving")
        data = [
            {"first_id": _id1, "second_id": _id2, "weight": weight}
            for _id1, sims in similar_vectors.items()
            for _id2, weight in sims.items()
        ]
        # Save the output chunk in subchunks, which saves both IO and memory on curate.
        # Use idx rather than common.get_chunks to save on memory
        for idx in range(0, len(data), CHUNKSIZE):
            chunk = data[idx : idx + CHUNKSIZE]
            first_id = chunk[0]["first_id"]
            last_id = chunk[-1]["first_id"]
            filename = f"deduplication_{first_id}-{last_id}_test-{self.test}.json"
            with S3(run=self) as s3:
                chunk = json.dumps(chunk)
                s3.put(filename, chunk)
        self.next(self.dummy_join_step)

    @step
    def dummy_join_step(self, inputs):
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    DeduplicationFlow()
