"""
requires_degree flow

A Metaflow Flow for extracting a degree requirement (true/false) from raw job adverts.
"""
import json

# Required for batch
import os

os.system(
    f"pip install -r {os.path.dirname(os.path.realpath(__file__))}/requirements.txt 1> /dev/null"
)
from daps_utils import talk_to_luigi, db
from metaflow import FlowSpec, step, S3, batch
from daps_utils.flow import DapsFlowMixin

### Boilerplate cribbed from salaries.py
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
# from daps_utils import db -- redundant?

db.CALLER_PKG = ojd_daps
db_session = db.db_session
# <<<
### /Boilerplate cribbed from salaries.py


@talk_to_luigi
class RequiresDegreeFlow(FlowSpec, DapsFlowMixin):
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
        from labs.requires_degree.model import io

        limit = 1000 if self.test else None
        ## object_as_dict doesn't seem to handle this table with column selection
        jobs = io.load_jobs(
            limit=limit,
            database=self.db_name,
            columns=(RawJobAd.id, RawJobAd.description),
        )
        self.job_ads = list(jobs)
        self.chunks = get_chunks(self.job_ads, 20000)
        self.next(self.extract_requires_degree, foreach="chunks")

    @batch(cpu=2, memory=8000)
    @step
    def extract_requires_degree(self):
        """
        Perform extraction - requires_degree True/False
        """
        from labs.requires_degree import model

        requires_degree = []
        for job_ad in self.input:
            requires_degree_dict = dict(requires_degree=model.apply_model(job_ad))
            requires_degree_dict["id"] = job_ad["id"]
            requires_degree.append(requires_degree_dict)
        first_id = requires_degree[0]["id"]
        last_id = requires_degree[-1]["id"]
        filename = f"requires_degree_test-{self.test}_{first_id}_{last_id}.json"
        with S3(run=self) as s3:
            data = json.dumps(requires_degree)
            s3.put(filename, data)
        self.next(self.join_requires_degree)

    @step
    def join_requires_degree(self, inputs):
        """
        Dummy joins inputs step
        """
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    RequiresDegreeFlow()
