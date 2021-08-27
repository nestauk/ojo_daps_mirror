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
from itertools import groupby
from datetime import datetime as dt
import networkx
from sqlalchemy import func as sql_func
import logging
from tqdm import tqdm

from daps_utils import db
from ojd_daps.flows.collect.common import get_metaflow_bucket
from ojd_daps.orms.raw_jobs import RawJobAd, JobAdDescriptionVector
from ojd_daps.orms.std_features import Location, Salary, SOC, RequiresDegree
from ojd_daps.orms.link_tables import (
    JobAdLocationLink,
    JobAdSOCLink,
    JobAdSkillLink,
    JobAdDuplicateLink,
)
from ojd_daps.dqa.vector_utils import download_vectors

CENTRAL_BUCKET = "most_recent_jobs"


def get_s3_job_ads(job_board, read_body=True, sample_ratio=1):
    """Retrieve 'raw' job advert data from S3

    Args:
        job_board (str): Assumed to be 'reed'
        read_body (bool): If you really don't need to look at the text, set this to
                          False to speed things up.
        sample_ratio (float): If you need to reduce the sample size randomly, scale
                              this down appropriately
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


def get_db_job_ads(
    limit=None,
    chunksize=1000,
    return_features=False,
    deduplicate=False,
    min_dupe_weight=0.95,
    max_dupe_weight=1,
    split_dupes_by_location=False,
    from_date=None,
    to_date=None,
):
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
        deduplicate (bool): Whether or not to include a deduplication flag in the
                            returned data.
        {min, max}_dupe_weight (int): Min/max deduplication weight to consider (0 to 1)
                                      where 0 indicates that there is no similarity
                                      between job ad descriptions, and 1 indicates that
                                      the descriptions are identical.
        split_dupes_by_location (bool): Whether or not to discount pairs of job adverts
                                        (with duplicate descriptions) as duplicates if
                                        they do not share a common location.
        {to, from}_date (str): Strings formatted as %d-%m-%Y, indicating the date
                               bounds of the query.
    Yields:
         A job advert "object" in dict form
    """
    logging.basicConfig(level=logging.INFO)
    # Grab the features up front
    features = get_features() if return_features else None

    # Create a date filter
    date_filter = True
    if from_date:
        date_filter = RawJobAd.created >= dt.strptime(from_date, "%d-%m-%Y")
    if to_date:
        date_filter = date_filter & (
            RawJobAd.created <= dt.strptime(to_date, "%d-%m-%Y")
        )

    # Retrieve duplicate IDs if required
    dupe_ids = (
        get_duplicate_ids(
            min_weight=min_dupe_weight,
            max_weight=max_dupe_weight,
            split_by_location=split_dupes_by_location,
            date_filter=date_filter,
        )
        if deduplicate
        else set()
    )

    max_id, total_rows = None, 0  # To keep a track of progress
    with db.db_session("production") as session:
        session.bind.execution_options(
            stream_results=True
        )  # Better performance for large rows

        # Setup a base query
        base_query = session.query(RawJobAd).filter(date_filter).order_by(RawJobAd.id)
        # Limit / offset until done
        logging.info("Iterating over DB rows...")
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
                if deduplicate:
                    row["features"]["is_duplicate"] = row["id"] in dupe_ids
                yield row
                # Break early if limit reached
                if total_rows == limit:
                    break
            max_id = max(ids)  # Recalculate the offset
            # Break if limit reached or there are no new results
            if total_rows == limit or len(ids) == 0:
                break


def get_duplicate_ids(min_weight, max_weight, split_by_location, date_filter):
    """
    Retrieve ids of job ads, marked as duplicate in this date filter.
    In practice this is a shallow wrapper around identify_duplicates, but
    it is useful to have identify_duplicates seperate for debugging purposes.
    """
    with db.db_session("production") as session:
        all_ids = set(
            item for item, in session.query(RawJobAd.id).filter(date_filter).all()
        )
    dupe_ids = identify_duplicates(
        ids=all_ids,
        min_weight=min_weight,
        max_weight=max_weight,
        split_by_location=split_by_location,
    )
    logging.debug("Identified", len(dupe_ids), "dupes from", len(all_ids))
    return dupe_ids


@lru_cache()
def get_duplicate_subgraphs(min_weight=1, max_weight=1):
    """Generate every group of duplicate job ads, for all time"""
    logging.info("Retrieving duplicate subgraphs")
    with db.db_session("production") as session:
        query = session.query(JobAdDuplicateLink.first_id, JobAdDuplicateLink.second_id)
        query = query.filter(JobAdDuplicateLink.weight.between(min_weight, max_weight))
        edge_list = list(query.all())
    graph = networkx.Graph(edge_list)
    return list(networkx.connected_components(graph))


@lru_cache()
def get_subgraphs_by_location(min_weight=1, max_weight=1):
    """
    A typical use-case is that we want to consider job ads as distinct
    if they're advertised in different locations. This method will
    split the subgraphs into smaller subgraphs such to reflect unique
    (duplicate group number, location) combinations.
    """
    # Create an inverse lookup of {job_ad_id --> duplicate_group_idx}
    subgraphs = get_duplicate_subgraphs(min_weight, max_weight)
    group_lookup = {
        id: idx for idx, subgraph in enumerate(subgraphs) for id in subgraph
    }

    # Retrieve job ad location and description
    ids_by_group = []
    with db.db_session("production") as session:
        query = session.query(
            RawJobAd.id,
            RawJobAd.job_location_raw,
            sql_func.length(RawJobAd.description),
        )
        for id, location, len_description in query.all():
            idx = group_lookup.get(id)  # duplicate group idx
            # If this isn't a duplicate at all (idx = None)
            # or the description is too short (this shouldn't happen)
            # then don't include this in the new subgraph
            if idx is None or len_description < 5:
                continue
            ids_by_group.append((id, (idx, location)))

    # Convert back into subgraphs, grouping by (idx, location)
    ids_by_group = sorted(ids_by_group, key=lambda item: item[1])  # Sort before groupby
    _subgraphs = list(
        set(id for id, _ in ids)  # _ = (idx, loc)
        for _, ids in groupby(ids_by_group, key=lambda item: item[1])  # _ = (idx, loc)
    )

    # By definition, groups with length 1 aren't dupes
    return [ids for ids in _subgraphs if len(ids) > 1]


def identify_duplicates(ids, min_weight, max_weight, split_by_location):
    """Identify duplicates among provided IDs. Where duplicates
    are discovered (at least 2 IDs required by definition) then
    one of the IDs will be excluded so that it can be interpretted as the
    "exemplar" of the duplicate group."""
    subgraphs = (
        get_subgraphs_by_location(min_weight, max_weight)
        if split_by_location
        else get_duplicate_subgraphs(min_weight, max_weight)
    )

    dupes = set()
    logging.info("Iterating over duplicate subgraphs...")
    for subgraph in tqdm(subgraphs):
        # Most subgraphs won't contain any of the IDs,
        # and so doing isdisjoint is a speed-up on that
        # assumption
        if subgraph.isdisjoint(ids):
            continue
        # Get the set of duplicates and then remove a single exemplar
        _dupes = subgraph.intersection(ids)
        # Deterministically select (i.e. "min") and then remove the exemplar
        _dupes.remove(min(_dupes))
        # Append to the set of duplicates
        dupes = dupes.union(_dupes)
    return dupes


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
            logging.warning(
                f"Multiple names ({names}) found for {code}, taking the shortest"
            )
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


def get_requires_degree():
    """
    Retrieve requires_degree flag which we have assigned to each job advert.

    Yields:
        row (dict): Job advert IDs matched to a boolean flag.
    """
    with db.db_session("production") as session:
        for row in map(db.object_as_dict, session.query(RequiresDegree).all()):
            # RequiresDegree isn't a link table so id means job_id
            row["job_id"] = str(row.pop("id"))
            # Don't need the __version__ field, this is implied in the job_ads data
            row.pop("__version__")
            yield row


def get_skills():
    """
    Retrieve skills group data which we have assigned to each job advert.

    Yields:
        row (dict): Job advert IDs matched to a skills group.
    """
    with db.db_session("production") as session:
        for row in map(db.object_as_dict, session.query(JobAdSkillLink).all()):
            # Don't need the __version__ field, this is implied in the job_ads data
            row.pop("job_data_source")
            row.pop("__version__")
            row["job_id"] = str(row.pop("job_id"))
            yield {
                column: (float(value) if type(value) is Decimal else value)
                for column, value in row.items()
            }


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
        ("skills", get_skills),
        ("requires_degree", get_requires_degree),
    ]
    # Generate the feature collection for all job adverts
    features = defaultdict(dict)
    for feature_name, getter in feature_getters:
        logging.info(f"Retrieving feature '{feature_name}'")
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
