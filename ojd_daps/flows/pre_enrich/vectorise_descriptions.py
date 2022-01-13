"""
vectorise_descriptions_flow
---------------------------

A Flow for vectorising descriptions from job adverts.
"""
import json

from daps_utils import DapsFlowMixin
from daps_utils.db import object_as_dict

from metaflow import FlowSpec, S3, batch, pip, retry, step

import ojd_daps
from ojd_daps.orms.raw_jobs import RawJobAd


MODEL_NAME = "distilbert-base-nli-stsb-mean-tokens"
CHUNKSIZE = 5000


def load_model(model_name):
    """Load the sentence transformer"""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def truncate_and_jsonify(vector, decimal_places):
    """
    Truncate each value in a vector to a fixed number
    of decimal places and convert to json-string
    """
    truncated_format = f"%.{decimal_places}f"
    string_vector = [truncated_format % v for v in vector.tolist()]
    return json.dumps(string_vector)


def encode_job_ads(job_ads, decimal_places=5):
    """
    Encode job ad descriptions using a sentence transformer, and truncate
    the vectors to save space.
    """
    model = load_model(MODEL_NAME)
    # Pop to reduce memory overhead
    texts = list(job_ad.pop("description") for job_ad in job_ads)
    ids = (job_ad["id"] for job_ad in job_ads)
    del job_ads  # Reduce memory overhead

    embeddings = model.encode(texts)
    del texts  # Reduce memory overhead

    vectors = [
        {
            "id": _id,
            "vector": truncate_and_jsonify(vector, decimal_places=decimal_places),
        }
        for _id, vector in zip(ids, embeddings)
    ]
    return vectors


class VectoriseDescriptionsFlow(FlowSpec, DapsFlowMixin):
    @step
    def start(self):
        self.next(self.get_descriptions)

    @step
    def get_descriptions(self):
        """
        Gets locations, breaks up into chunks of CHUNKSIZE.
        """
        from sqlalchemy.sql.expression import func

        non_empty_text = func.length(RawJobAd.description) > 5
        self.chunks = []
        with self.db_session(database="production") as session:
            # Create a base query, over which we will filter/iterate in chunks
            base_query = session.query(RawJobAd.id, RawJobAd.description)
            base_query = base_query.filter(non_empty_text).order_by(RawJobAd.id)
            # Read chunks from the database, ordered by the PK
            # (this is the fastest way to read in chunks)
            while "reading db":
                query = base_query  # For the first chunk
                if len(self.chunks) > 0:  # For subsequent chunks
                    max_id = self.chunks[-1][-1]["id"]  # last id of most recent chunk
                    query = base_query.filter(RawJobAd.id > max_id)
                new_ads = [object_as_dict(obj) for obj in query.limit(CHUNKSIZE)]
                self.chunks.append(new_ads)
                # The final chunk will have less than CHUNKSIZE, including
                # if it is empty
                if (len(new_ads) < CHUNKSIZE) or self.test:
                    break
        self.next(self.vectorise_descriptions, foreach="chunks")

    @retry
    @pip(path="requirements_transform.txt")
    @batch(cpu=8, memory=32000)
    @step
    def vectorise_descriptions(self):
        """
        Vectorises descriptions
        """
        vectors = encode_job_ads(job_ads=self.input)

        # Write the data to a uniquely named file
        first_id = vectors[0]["id"]
        last_id = vectors[-1]["id"]
        filename = f"vectorise_descriptions_test-{self.test}_{first_id}_{last_id}.json"
        with S3(run=self) as s3:
            data = json.dumps(vectors)
            s3.put(filename, data)
        self.next(self.join_vectors)

    @step
    def join_vectors(self, inputs):
        """
        Joins inputs
        """
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    VectoriseDescriptionsFlow()
