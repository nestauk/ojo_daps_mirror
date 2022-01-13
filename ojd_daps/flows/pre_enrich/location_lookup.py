"""
location lookup
---------------

A Flow for generating a lookup table of UK place names to GSS and NUTS locations
"""
import csv
import json
import re
from functools import lru_cache
from io import StringIO

import boto3

from daps_utils import DapsFlowMixin

from metaflow import FlowSpec, S3, batch, pip, step

from ojd_daps.flows.common import flatten, get_chunks

CHUNKSIZE = 1000
# Lookups for converting ONS names to human-readable
NAME_LOOKUP = {
    "place": "ipn_",
    "ctry": "country_",
    "lad": "lad_",
    "hlth": "health_",
    "rgn": "region_",
}
CODE_LOOKUP = {"nm": "_name", "cd": "_code"}
# Location of the raw data, originally from tinyurl.com/4nushphc
BUCKET = "open-jobs-lake"
KEYS = [
    "jobs-metadata/IPN_GB_2019.csv",
    "jobs-metadata/manual_places.csv",
    "jobs-metadata/manual_ni_places.csv",
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


def extract_fields(row, ignore_fields=["lat", "long"]):
    """
    Extract required fields from the row, and change make
    field names human-readable as required.

    Args:
        row (dict): A row of raw ONS PlaceName metadata
        ignore_fields (list): Fields to be unaltered and kept in place
    Returns:
        new_row (dict): Altered copy of the original row
    """
    new_row = row.copy()
    for key in row:
        # Don't pop fields or alter the field name
        # if in `ignore_fields`
        if key in ignore_fields:
            continue
        # Pop so that we can discard if no match is found
        value = new_row.pop(key)
        # Scan for whether this key matches one of the prefix/suffix
        # combinations specified in `NAME_LOOKUP` and `CODE_LOOKUP`
        # meaning that a) we want it for our ORM and b) we need
        # to rename it to something human-readable
        is_orm_field = False
        for old_pfix, old_sfix, new_pfix, new_sfix in suffix_lookup():
            regex = rf"^{old_pfix}(\d+){old_sfix}$"
            is_orm_field = re.match(regex, key) is not None
            if is_orm_field:  # Break out once found
                break
        if not is_orm_field:
            continue
        # Convert the name to something human-readable
        key = key.replace(old_pfix, new_pfix)
        key = key.replace(old_sfix, new_sfix)
        new_row[key] = value
    return new_row


@lru_cache()
def suffix_lookup():
    """
    Flat lookup of 'old' (ONS) prefixes and suffixes to
    'new' (human-readable) prefixes and suffixes
    """
    return [
        (old_pfix, old_sfix, new_pfix, new_sfix)
        for old_pfix, new_pfix in NAME_LOOKUP.items()
        for old_sfix, new_sfix in CODE_LOOKUP.items()
    ]


@lru_cache()
def NutsFinder():
    """
    Cache the NutsFinder, instantiated from the most finely grained
    scale available. Note the the lookup time for scale=1 are super slow
    but using less granular scales will result in missing places near
    bodies of water
    """
    from nuts_finder import NutsFinder as _NutsFinder

    return _NutsFinder(scale=1)


def find_nuts(lat, lon):
    """
    Yield every NUTS field for this lat, lon in the form:

        nuts_1_code: ACODE123
        nuts_1_name: Liverpool

    for all NUTS levels. If no level is found, nothing is returned.
    """
    nf = NutsFinder()
    for nuts in nf.find(lat=float(lat), lon=float(lon)):
        lvl = f'nuts_{nuts["LEVL_CODE"]}'
        yield f"{lvl}_code", nuts["NUTS_ID"]
        yield f"{lvl}_name", nuts["NUTS_NAME"].lower().title()


def impute_nuts(row):
    """Append the NUTS fields and values for this row"""
    row = row.copy()
    for k, v in find_nuts(lat=row.pop("lat"), lon=row.pop("long")):
        row[k] = v
    return row


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
    row = {k: v for k, v in row.items() if v is not None}
    row = extract_fields(row)
    row = impute_nuts(row)
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
            key_metadata = list(dedupe(key_metadata, "placeid"))
            metadata.append(key_metadata)
        metadata = flatten(metadata)
        chunksize = round(CHUNKSIZE / 10) if self.test else CHUNKSIZE
        limit = (2 * chunksize) if self.test else None
        self.metadata = get_chunks(metadata[:limit], chunksize)
        self.next(self.transform_metadata, foreach="metadata")

    @batch(cpu=4)
    @pip(path="requirements_nuts.txt")
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
