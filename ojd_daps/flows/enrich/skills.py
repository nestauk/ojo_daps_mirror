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

CHUNKSIZE = 20000
MODEL_VERSION = "v02_1"


@talk_to_luigi
class SkillsFlow(FlowSpec, DapsFlowMixin):
    @step
    def start(self):
        """
        Starts the flow.
        """
        self.next(self.get_adverts)

    @step
    def get_adverts(self):
        """
        Gets adverts, breaks up into chunks of CHUNKSIZE.
        """
        from common import get_chunks

        limit = 1000 if self.test else None
        with db_session(database="production") as session:
            jobad_query = session.query(
                RawJobAd.id, RawJobAd.data_source, RawJobAd.description
            )
            jobad_query = jobad_query.filter(RawJobAd.id is not None)
            job_ads = [object_as_dict(obj) for obj in jobad_query.limit(limit)]
        self.chunks = get_chunks(job_ads, CHUNKSIZE)
        self.next(self.extract_skills, foreach="chunks")

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

        model = load_model(MODEL_VERSION)
        nlp = setup_spacy_model(model["nlp"])
        skills = []
        for job_ad in self.input:
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
