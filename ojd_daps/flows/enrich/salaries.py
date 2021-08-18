"""
salaries_flow
-------------

A Flow for extracting a standardised salary from raw job adverts.
"""
# Required for batch
import os

os.system(
    f"pip install -r {os.path.dirname(os.path.realpath(__file__))}/requirements.txt 1> /dev/null"
)
import json
import re

from metaflow import FlowSpec, step, S3, Parameter, batch, resources

from daps_utils import talk_to_luigi, db
from daps_utils.flow import DapsFlowMixin
from daps_utils.db import db_session, object_as_dict

# >>> Workaround for batch
try:
    from ojd_daps.orms.raw_jobs import RawJobAd
except ModuleNotFoundError:
    pass
# <<<

# >>> Workaround for batch
try:
    import ojd_daps
except ModuleNotFoundError:
    ojd_daps = None
# <<<

# >>> Workaround for metaflow introspection
from daps_utils import db

db.CALLER_PKG = ojd_daps
db_session = db.db_session
# <<<


@talk_to_luigi
class SalariesFlow(FlowSpec, DapsFlowMixin):
    @step
    def start(self):
        """
        Starts the flow.
        """
        self.next(self.get_adverts)

    @step
    def get_adverts(self):
        """
        Gets adverts, breaks up into chunks of 20,000.
        """
        from common import get_chunks

        limit = 1000 if self.test else None
        with db_session(database=self.db_name) as session:
            jobad_query = session.query(RawJobAd.id, RawJobAd.job_salary_raw)
            jobad_query = jobad_query.filter(RawJobAd.job_salary_raw is not None)
            self.job_ads = [
                {"id": _id, "job_salary_raw": salary}
                for _id, salary in jobad_query.limit(limit)
            ]
        self.chunks = get_chunks(self.job_ads, 20000)
        self.next(self.extract_salaries, foreach="chunks")

    @batch(cpu=4)  # cpu=4 to limit the number of containers per batch (diskspace issue)
    @step
    def extract_salaries(self):
        """
        Matches locations
        """
        from labs.salaries.regex.multi_regex_utils import apply_model

        salaries = []
        for job_ad in self.input:
            salary_dict = apply_model(job_ad)
            salary_dict['id'] = job_ad["id"]
            salaries.append(salary_dict)
        first_id = salaries[0]["id"]
        last_id = salaries[-1]["id"]
        filename = f"salaries_test-{self.test}_{first_id}_{last_id}.json"
        with S3(run=self) as s3:
            data = json.dumps(salaries)
            s3.put(filename, data)
        self.next(self.join_extracted_salaries)

    @step
    def join_extracted_salaries(self, inputs):
        """
        Dummy joins inputs step
        """
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    SalariesFlow()
