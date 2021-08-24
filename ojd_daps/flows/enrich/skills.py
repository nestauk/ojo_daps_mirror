"""
skills_flow
-------------
A Flow for extracting a skills from raw job adverts.
"""
# Required for batch
import os

os.system(
    f"pip install -r {os.path.dirname(os.path.realpath(__file__))}/requirements.txt 1> /dev/null"
)
import json
from common import generate_description_queries, retrieve_job_ads
from metaflow import FlowSpec, step, S3, Parameter, batch
from daps_utils import talk_to_luigi, DapsFlowMixin

CHUNKSIZE = 5000
MODEL_VERSION = "v02_1"


@talk_to_luigi
class SkillsFlow(FlowSpec, DapsFlowMixin):
    @step
    def start(self):
        """
        Starts the flow.
        """
        # >>> Workaround for metaflow introspection
        import ojd_daps

        self.set_caller_pkg(ojd_daps)
        # <<<
        self.next(self.get_adverts)

    @step
    def get_adverts(self):
        """
        Gets adverts, breaks up into chunks of CHUNKSIZE.
        """
        self.queries = generate_description_queries(self, CHUNKSIZE)
        self.next(self.extract_skills, foreach="queries")

    @batch(cpu=4, memory=16000)
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
