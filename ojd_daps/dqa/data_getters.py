"""
data getters
------------

Utils for retrieving data from either S3 or the database
"""

import boto3
from random import uniform
from functools import lru_cache
from collections import defaultdict
from decimal import Decimal
from daps_utils import db
from ojd_daps.flows.collect.common import get_metaflow_bucket
from ojd_daps.orms.raw_jobs import RawJobAd, JobAdDescriptionVector
from ojd_daps.orms.std_features import Location, Salary, SOC
from ojd_daps.orms.link_tables import JobAdLocationLink, JobAdSOCLink
from ojd_daps.dqa.vector_utils import download_vectors

# <<<<
# Workaround for the hackweek
import ojd_daps

db.CALLER_PKG = ojd_daps
# Uncomment and fill out the following if you want to specify a backup database
db.CALLER_PKG.config["mysqldb"]["mysqldb"][
    "host"
] = "ojo-backup-2021-07-16-04-40.ci9272ypsbyf.eu-west-2.rds.amazonaws.com"
# >>>>

CENTRAL_BUCKET = "most_recent_jobs"


def get_s3_job_ads(job_board, read_body=True, sample_ratio=1):
    """Retrieve 'raw' job advert data from S3

    Args:
        job_board (str): Assumed to be 'reed'
        read_body (bool): If you really don't need to look at the text, set this to False to speed things up.
        sample_ratio (float): If you need to reduce the sample size randomly, scale this down appropriately
                             (i.e. 0.02 means randomly reject 98% of the data).
    Yields:
         A job advert "object" in dict form
    """
    bucket = get_metaflow_bucket()
    prefix = f"{CENTRAL_BUCKET}/production/{job_board}"
    s3 = boto3.resource("s3")
    bucket = s3.Bucket(bucket)
    for obj in bucket.objects.filter(Prefix=prefix):
        if uniform(0.0, 1.0) + sample_ratio < 1:
            continue
        body = obj.get()["Body"].read() if read_body else None
        yield {
            "timestamp": obj.last_modified,
            "filename": obj.key,
            "job_board": job_board,
            "body": body,
            "filesize": obj.size,
        }


def get_db_job_ads(limit=None, chunksize=1000, return_features=False):
    """Retrieve 'curated' job advert data from the database

    Args:
        limit (int): Maximum number of objects returned
        chunksize (int): Basically making this bigger or smaller will change
                         the data retrieval speed. It's hard to predict
                         what the optimum number will be: it depends on the data,
                         your laptop and your internet connection speed.
                         1000 is a good minimum though!
        return_features (bool): Return predicted features for each job ad
                                under an additional field (features).
    Yields:
         A job advert "object" in dict form
    """
    features = get_features() if return_features else None
    max_id, total_rows = None, 0  # To keep a track of progress
    with db.db_session("production") as session:
        session.bind.execution_options(
            stream_results=True
        )  # Better performance for large rows
        # Setup a base query
        base_query = session.query(RawJobAd).order_by(RawJobAd.id)
        # Limit / offset until done
        while (
            total_rows % chunksize == 0
        ):  # i.e. expect chunksize until the final chunk
            ids = set()  # for calculating the offset
            query = base_query
            if total_rows > 0:  # after the first chunk do an offset
                query = query.filter(RawJobAd.id > max_id)
            # Yield rows in this chunk
            for obj in query.limit(chunksize):
                ids.add(obj.id)
                total_rows += 1
                row = db.object_as_dict(obj)
                if features:
                    row["features"] = features[row["id"]]
                yield row
                # Break early if limit reached
                if total_rows == limit:
                    break
            max_id = max(ids)  # Recalculate the offset
            # Break if limit reached or there are no new results
            if total_rows == limit or len(ids) == 0:
                break


def get_locations(level, do_lookup=False):
    """
    Retrieve locations which we have assigned to each job advert.

    Args:
        level (str): A geographic level, choose from: lad_18, health_12, region_18, nuts_0, nuts_1, nuts_2, nuts_3
    Yields:
        row (dict): Job advert IDs matched to a geographic code.
    """

    lookup = get_location_lookup() if do_lookup else None
    fields = (JobAdLocationLink.job_id, getattr(Location, f"{level}_code"))
    join_on_id = Location.ipn_18_code == JobAdLocationLink.location_id
    with db.db_session("production") as session:
        # Setup a base query
        query = session.query(*fields).distinct(*fields).outerjoin(Location, join_on_id)
        for id, location in query.all():
            if location is None:
                continue
            row = {"job_id": id, f"{level}_code": location}
            if lookup:
                row[f"{level}_name"] = lookup[location]
            yield row


@lru_cache()
def get_location_lookup():
    """
    Retrieve name lookups for every geography code.

    Returns:
        lookup (dict): Lookup of code --> name
    """
    with db.db_session("production") as session:
        metadata = list(map(db.object_as_dict, session.query(Location).all()))
    # Match codes to their names from the database
    code_lookup = defaultdict(set)
    for row in metadata:
        for key in row:
            if not key.endswith("code"):
                continue
            if key.startswith("ipn"):
                continue
            code = row[key]
            name = row[f"{key[:-5]}_name"]
            if not code:  # empty, None, etc
                continue
            code_lookup[code].add(name)
    # Some small DQA to guarantee consistency
    for code, names in code_lookup.copy().items():
        names = list(filter(len, names))  # Get non-empty names
        if len(names) > 1:
            print(f"Multiple names ({names}) found for {code}, taking the shortest")
        try:
            code_lookup[code] = min(names, key=len)  # Shortest
        except ValueError:
            raise ValueError(f"Zero non-empty names found for {code}")
    return dict(code_lookup)


def get_salaries():
    """
    Retrieve the salary we have assigned to each job advert.

    Returns:
        salaries (list of dict): One row per job advert, containing salary, rate and
                                 normalised annual salary for rate != 'per annum'
    """
    with db.db_session("production") as session:
        for row in map(db.object_as_dict, session.query(Salary).all()):
            # Salary isn't a link table so id means job_id
            row["job_id"] = str(row.pop("id"))
            # Don't need the __version__ field, this is implied in the job_ads data
            row.pop("__version__")
            # Salaries are "Decimal" objects, which are a little faffy
            # so convert to float
            yield {
                column: (float(value) if type(value) is Decimal else value)
                for column, value in row.items()
            }


def get_soc():
    """
    Retrieve SOCS which we have assigned to each job advert.

    Yields:
        row (dict): Job advert IDs matched to a geographic code.
    """
    fields = (JobAdSOCLink.job_id, SOC.soc_code, SOC.soc_title)
    join_on_id = SOC.soc_id == JobAdSOCLink.soc_id
    with db.db_session("production") as session:
        # Setup a base query
        query = session.query(*fields).join(SOC, join_on_id)
        for row in map(db.object_as_dict, query.all()):
            yield row


@lru_cache()
def get_features(location_level="nuts_2"):
    """
    Retrieve the predicted feature collection for all job adverts.

    Args:
        location_level (str): The level parameter, as described in `get_locations`.
    Returns:
        features (dict): A lookup of job_id to features
    """
    # To future developers: add new features here.
    # They should return a dict, at least containing the key 'job_id'
    feature_getters = [
        ("salary", get_salaries),
        ("location", lambda: get_locations(location_level, do_lookup=True)),
        ("soc", get_soc),
    ]
    # Generate the feature collection for all job adverts
    features = defaultdict(dict)
    for feature_name, getter in feature_getters:
        for row in getter():  # one per predicted job ad
            features[row.pop("job_id")][feature_name] = row
    return dict(features)  # Undefault the defaultdict


def get_vectors(chunksize=10000, max_chunks=None):
    """
    Get text vectors from the database, populated into numpy arrays.

    Args:
        chunksize (int): Chunksize to stream from the DB. Probably don't change this.
        max_chunks (int): Number of chunks to retrieve from the database (None = all).
    """
    with db.db_session("production") as session:
        return download_vectors(
            orm=JobAdDescriptionVector,
            id_field="id",
            session=session,
            chunksize=chunksize,
            max_chunks=max_chunks,
        )
