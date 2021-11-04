"""
jobs_by_locations_flow
----------------------

A Flow for aggregating jobs by locations.
"""
# Required for batch
import os

os.system(
    f"pip install -r {os.path.dirname(os.path.realpath(__file__))}/requirements.txt 1> /dev/null"
)
import json
import re

from sqlalchemy import func

from metaflow import FlowSpec, step, S3, Parameter, batch, resources

from ojd_daps.orms.std_features import Location
from ojd_daps.orms.raw_jobs import RawJobAd
from ojd_daps.orms.link_tables import JobAdLocationLink

from daps_utils import talk_to_luigi, db
from daps_utils.flow import DapsFlowMixin
from daps_utils.db import db_session, object_as_dict

import ojd_daps
from daps_utils import db

db.CALLER_PKG = ojd_daps
db_session = db.db_session


@talk_to_luigi
class LocationsFlow(FlowSpec, DapsFlowMixin):
    @step
    def start(self):
        """
        Starts the flow.
        """
        self.next(self.get_data)

    @step
    def get_data(self):
        """
        Queries the aggregation from the database.
        """
        limit = 1000 if self.test else None
        with db_session(database=self.db_name) as session:
            loc_id = JobAdLocationLink.location_id
            count = func.count(loc_id).label("count")
            query = session.query(loc_id, count).group_by(loc_id)
            self.data = [obj._asdict() for obj in query.all()]
        self.next(self.end)

    @step
    def end(self):
        """
        Ends the flow, saving json data to S3
        """
        filename = f"jobs_by_locs_test-{self.test}.json"
        with S3(run=self) as s3:
            data = json.dumps(self.data)
            url = s3.put(filename, data)


if __name__ == "__main__":
    LocationsFlow()
