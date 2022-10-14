"""
location lookup
---------------

A Flow for generating a lookup table of UK place names to GSS and NUTS locations
"""
import csv
import json
from io import StringIO

import boto3

from daps_utils import DapsFlowMixin

from metaflow import FlowSpec, S3, step, retry, pip

from ojd_daps.flows.common import flatten, get_chunks

CHUNKSIZE = 1000
# Location of the raw data, originally from tinyurl.com/4nushphc
BUCKET = "open-jobs-lake"
KEYS = [
    "jobs-metadata/locs_index_v2.csv",
]


def stream_metadata(bucket, key):
    """Read the data from S3 as a streaming object"""
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket, Key=key)
    raw_data = obj["Body"].read()
    raw_stream = StringIO(raw_data.decode("latin"))
    return raw_stream


def read_csv(bucket, key):
    """Read CSV data from S3"""
    raw_stream = stream_metadata(bucket, key)
    header = next(csv.reader(raw_stream))
    return csv.DictReader(raw_stream, fieldnames=header)


def dedupe(data, primary_key):
    """Deduplicate the data based on the primary key,
    taking the first instance in each case

    Args:
        data (iterable of dict): Data to be deduplicated
        primary_key (str): Name of the primary key
    Yields:
        row (dict): First row with it's value of the primary key
    """
    pks = set()
    for row in data:
        if row[primary_key] in pks:
            continue
        pks.add(row[primary_key])
        yield row


def transform_metadata(row):
    """Standardise fields and impute NUTS metadata"""
    row = row.copy()
    row = {k: v for k, v in row.items()}
    return row


class LocationMetadataFlow(FlowSpec, DapsFlowMixin):
    @step
    def start(self):
        """
        Starts the flow.
        """
        metadata = []
        for key in KEYS:
            key_metadata = read_csv(BUCKET, key)
            key_metadata = list(dedupe(key_metadata, "ipn_18_code"))
            metadata.append(key_metadata)
        metadata = flatten(metadata)
        chunksize = round(CHUNKSIZE / 10) if self.test else CHUNKSIZE
        limit = (2 * chunksize) if self.test else None
        self.metadata = get_chunks(metadata[:limit], chunksize)
        self.next(self.transform_metadata, foreach="metadata")

    @retry
    @step
    def transform_metadata(self):
        """Pulls out the required fields and identifies NUTS region"""
        metadata = list(map(transform_metadata, self.input))
        first_id = metadata[0]["ipn_18_code"]
        last_id = metadata[-1]["ipn_18_code"]
        filename = f"location_lookup_test-{self.test}_{first_id}_{last_id}.json"
        with S3(run=self) as s3:
            data = json.dumps(metadata)
            s3.put(filename, data)
        self.next(self.dummy_join)

    @step
    def dummy_join(self, inputs):
        self.next(self.end)

    @step
    def end(self):
        """Done"""
        pass


if __name__ == "__main__":
    LocationMetadataFlow()
