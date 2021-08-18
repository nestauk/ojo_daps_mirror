"""
locations_flow
------------

A Flow for extracting a standardised location from raw locations.
"""
# Required for batch
import os

# >>> Workaround for batch
os.system(
    f"pip install -r {os.path.dirname(os.path.realpath(__file__))}/requirements.txt 1> /dev/null"
)
# <<<

import json
import re
from collections import defaultdict
from metaflow import FlowSpec, step, S3, Parameter, batch, resources

from daps_utils import talk_to_luigi, db
from daps_utils.flow import DapsFlowMixin
from daps_utils.db import db_session, object_as_dict

from ojd_daps.flows.enrich.common import get_chunks
from ojd_daps.orms.std_features import Location
from ojd_daps.orms.raw_jobs import RawJobAd as JobAd  # abbreviate

# >>> Workaround for batch
import ojd_daps
from daps_utils import db

db.CALLER_PKG = ojd_daps
db_session = db.db_session
# <<<


CHUNKSIZE = 300000  # Leads to output filesizes of ~20MB


def location_lookup():
    """
    Retrieves all of the location lookups from the production database,
    i.e. regardless of whether we are in "test" mode or not.
    The lookup is retrieved in the form:

        processed name --> list of ids

    Therefore on location name can have multiple IDs, for the case where
    a placename is not unique.
    """
    lookup = defaultdict(list)
    with db_session(database="production") as session:
        query = session.query(Location.ipn_18_code, Location.ipn_18_name)
        for code, name in query.all():
            lookup[process_location(name)].append(code)
    return lookup


def process_location(raw_location):
    """Function that does a small amount of preprocessing of the raw
    location to get it ready for matching.

    Parameters
    ----------
    raw_location : str
        Unaltered raw location

    Returns
    -------
    processed_location : str
        Processed location for matching
    """
    processed_location = (
        raw_location.split(",")[0].replace(r"[^\w\s]", "").lower().replace(" ", "_")
    )
    return processed_location


def define_processed_location(raw_location):
    """Produces processed location for matching

    Parameters
    ----------
    outcode : regex
        Regular expression to identify postcode area.
    raw_location : str
        Unaltered location from raw job ad.

    Returns
    -------
    processed_location : str
        Location now ready for matching
    """
    result = re.compile(r"[A-Z]{1,2}[0-9][0-9A-Z]?\s?").findall(raw_location)
    return process_location(raw_location) if len(result) == 0 else result[0]


@talk_to_luigi
class LocationsFlow(FlowSpec, DapsFlowMixin):
    @step
    def start(self):
        """
        Starts the flow.
        """
        self.next(self.get_locations)

    @step
    def get_locations(self):
        """
        Gets locations.
        """
        with db_session(database=self.db_name) as session:
            query = session.query(JobAd.id, JobAd.job_location_raw, JobAd.data_source)
            query = query.filter(JobAd.job_location_raw is not None)
            self.job_locations = {
                _id: (define_processed_location(loc), src)
                for _id, loc, src in query.all()
            }
        self.next(self.match_locations)

    @step
    def match_locations(self):
        """
        Matches each available job ad location to >> at least zero << standardised
        locations - i.e. multiple matches, or zero matches are allowed.
        """
        lookup = location_lookup()
        self.link_table = [
            {"job_id": _id, "job_data_source": src, "location_id": location_id}
            for _id, (location, src) in self.job_locations.items()
            for location_id in lookup[location]
        ]
        self.next(self.end)

    @step
    def end(self):
        """Write the data out in chunks to limit the file size as the dataset grows"""
        with S3(run=self) as s3:
            for chunk in get_chunks(self.link_table, CHUNKSIZE):
                first_id = chunk[0]["job_id"]
                last_id = chunk[-1]["job_id"]
                filename = f"locations_{first_id}-{last_id}-test-{self.test}.json"
                data = json.dumps(chunk)
                s3.put(filename, data)


if __name__ == "__main__":
    LocationsFlow()
