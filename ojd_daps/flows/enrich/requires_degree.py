"""
requires_degree flow

A Metaflow Flow for extracting a degree requirement (true/false) from raw job adverts.
"""
import json

from common import generate_description_queries, retrieve_job_ads

from daps_utils import DapsFlowMixin

from metaflow import FlowSpec, S3, batch, pip, step

CHUNKSIZE = 5000


class RequiresDegreeFlow(FlowSpec, DapsFlowMixin):
    @step
    def start(self):
        self.next(self.get_adverts)

    @step
    def get_adverts(self):
        """
        Gets adverts, breaks up into chunks of 20,000.
        """
        self.queries = generate_description_queries(self, CHUNKSIZE)
        self.next(self.extract_requires_degree, foreach="queries")

    @batch(cpu=2, memory=8000)
    @pip(path="requirements.txt")
    @step
    def extract_requires_degree(self):
        """
        Perform extraction - requires_degree True/False
        """
        from labs.requires_degree import model

        job_ads = retrieve_job_ads(self)

        requires_degree = []
        for job_ad in job_ads:
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
