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
from metaflow import FlowSpec, step, S3, batch
from daps_utils import talk_to_luigi, db
from daps_utils.flow import DapsFlowMixin


@talk_to_luigi
class SalariesFlow(FlowSpec, DapsFlowMixin):
    @step
    def start(self):
        # >>> Workaround for metaflow introspection
        import ojd_daps

        self.set_caller_pkg(ojd_daps)
        # <<<
        self.next(self.get_adverts)

    @step
    def get_adverts(self):
        """
        Gets adverts, breaks up into chunks of 20,000.
        """
        from common import get_chunks
        from ojd_daps.orms.raw_jobs import RawJobAd

        limit = 1000 if self.test else None
        with self.db_session(database="production") as session:
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
            if salary_dict is None:  # no valid salary found
                continue
            salary_dict["id"] = job_ad["id"]
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
