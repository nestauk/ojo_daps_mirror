"""
jobs_by_locations_flow
----------------------

A Flow for aggregating jobs by locations.
"""
import json
from sqlalchemy import func
from metaflow import FlowSpec, step, S3

import ojd_daps
from ojd_daps.orms.link_tables import JobAdLocationLink
from daps_utils import talk_to_luigi, DapsFlowMixin


@talk_to_luigi
class LocationsFlow(FlowSpec, DapsFlowMixin):
    @step
    def start(self):
        """
        Starts the flow.
        """
        # >>> Workaround for metaflow introspection
        self.set_caller_pkg(ojd_daps)
        # <<<
        self.next(self.get_data)

    @step
    def get_data(self):
        """
        Queries the aggregation from the database.
        """
        limit = 1000 if self.test else None
        with self.db_session(database="production") as session:
            loc_id = JobAdLocationLink.location_id
            count = func.count(loc_id).label("count")
            query = session.query(loc_id, count).group_by(loc_id)
            self.data = [obj._asdict() for obj in query.limit(limit)]
        self.next(self.end)

    @step
    def end(self):
        """
        Ends the flow, saving json data to S3
        """
        filename = f"jobs_by_locs_test-{self.test}.json"
        with S3(run=self) as s3:
            data = json.dumps(self.data)
            s3.put(filename, data)


if __name__ == "__main__":
    LocationsFlow()
