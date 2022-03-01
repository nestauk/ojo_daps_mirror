"""
salaries_flow
-------------

A Flow for extracting a standardised salary from raw job adverts.
"""
import json

from daps_utils import DapsFlowMixin
from daps_utils.db import object_as_dict

from metaflow import FlowSpec, S3, batch, pip, step

from ojd_daps.flows.common import get_chunks
from ojd_daps.orms.raw_jobs import RawJobAd

CHUNKSIZE = 20000


class SalariesFlow(FlowSpec, DapsFlowMixin):
    @step
    def start(self):
        self.next(self.get_adverts)

    @batch(2, memory=16000)
    @step
    def get_adverts(self):
        """
        Gets adverts, breaks up into chunks of 20,000.
        """
        limit = 2 * CHUNKSIZE if self.test else None
        with self.db_session(database="production") as session:
            jobad_query = session.query(
                RawJobAd.id,
                RawJobAd.raw_salary_unit,
                RawJobAd.raw_salary_currency,
                RawJobAd.raw_salary,
                RawJobAd.raw_min_salary,
                RawJobAd.raw_max_salary,
            )
            jobad_query = jobad_query.filter(RawJobAd.raw_salary_unit is not None)
            job_ads = list(map(object_as_dict, jobad_query.limit(limit)))
        self.chunks = get_chunks(job_ads, CHUNKSIZE)
        self.next(self.extract_salaries, foreach="chunks")

    @batch(cpu=2, memory=16000)
    @pip(path="requirements.txt")
    @step
    def extract_salaries(self):
        """
        Matches locations
        """
        from labs.salaries.common import extract_salary

        salaries = []
        for job_ad in self.input:
            salary_dict = extract_salary(job_ad)
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

    @batch(cpu=2, memory=16000)
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
