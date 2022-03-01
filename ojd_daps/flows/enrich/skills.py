"""
skills_flow
-------------
A Flow for extracting a skills from raw job adverts.
"""
import json

from common import generate_description_queries, retrieve_job_ads

from daps_utils import DapsFlowMixin

from metaflow import FlowSpec, S3, batch, pip, step

CHUNKSIZE = 5000
MODEL_VERSION = "v02_1"


class SkillsFlow(FlowSpec, DapsFlowMixin):
    @step
    def start(self):
        self.next(self.get_adverts)

    @step
    def get_adverts(self):
        """
        Gets adverts, breaks up into chunks of CHUNKSIZE.
        """
        self.queries = generate_description_queries(self, CHUNKSIZE)
        self.next(self.extract_skills, foreach="queries")

    @batch(cpu=2, memory=16000)
    @pip(path="requirements.txt")
    @step
    def extract_skills(self):
        """
        Matches locations
        """
        from labs.skills.skills_detection_utils import (
            load_model,
            setup_spacy_model,
            detect_skills,
            clean_text,
        )

        job_ads = retrieve_job_ads(self)
        model = load_model(MODEL_VERSION)
        nlp = setup_spacy_model(model["nlp"])
        skills = []
        for job_ad in job_ads:
            ad_skills = detect_skills(
                clean_text(job_ad["description"]), model, nlp, return_dict=True
            )
            for ad_skill in ad_skills:
                ad_skill["job_id"] = job_ad["id"]
                ad_skill["job_data_source"] = job_ad["data_source"]
            skills += ad_skills
        first_id = skills[0]["job_id"]
        last_id = skills[-1]["job_id"]
        filename = f"skills_test-{self.test}_{first_id}_{last_id}.json"
        with S3(run=self) as s3:
            data = json.dumps(skills)
            s3.put(filename, data)
        self.next(self.join_extracted_skills)

    @batch(cpu=2, memory=16000)
    @step
    def join_extracted_skills(self, inputs):
        """
        Dummy joins inputs step
        """
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    SkillsFlow()
