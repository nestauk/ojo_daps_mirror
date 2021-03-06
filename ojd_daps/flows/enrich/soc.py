"""
soc flow
--------

A Flow for extracting a standardised SOC code from raw job titles.
"""
import json

from daps_utils import DapsFlowMixin
from daps_utils.db import object_as_dict

from labs.soc.common import load_json_from_s3
from labs.soc.substring_utils import apply_model

from metaflow import FlowSpec, S3, batch, pip, step

from ojd_daps.flows.common import get_chunks
from ojd_daps.flows.pre_enrich.soc_lookup import short_hash
from ojd_daps.orms.raw_jobs import RawJobAd

CHUNKSIZE = 30000


def generate_soc_ids(soc_codes, std_titles):
    """
    Get the unique SOC code - title combination for this match by validating
    against to one-to-one lookup table "soc_code_to_std_title"
    """
    soc_to_title = load_json_from_s3("soc_code_to_std_title")
    std_titles = set(std_titles)
    for soc in map(str, soc_codes):  # Cast all to string
        # Get the union of sets to remove titles not relevant to this match
        titles = set(soc_to_title[soc]) & std_titles
        for title in titles:
            # Geneate the unique hash for this soc title
            yield short_hash(soc + title)


class SocMatchFlow(FlowSpec, DapsFlowMixin):
    @step
    def start(self):
        """Gets job titles, breaks up into chunks of CHUNKSIZE"""
        # Read all job titles
        limit = 2 * CHUNKSIZE if self.test else None  # 2 chunks for testing
        with self.db_session(database="production") as session:
            query = session.query(RawJobAd.id, RawJobAd.job_title_raw)
            job_ads = [object_as_dict(obj) for obj in query.limit(limit)]

        # Post filter is quicker, and the memory requirement is anyway small
        job_ads = [
            job
            for job in job_ads
            if job["job_title_raw"] is not None and len(job["job_title_raw"]) > 1
        ]

        # Split into chunks ready for batching
        self.chunks = get_chunks(job_ads, CHUNKSIZE)
        self.next(self.match_title_to_soc, foreach="chunks")

    @batch
    @pip(path="requirements.txt")
    @step
    def match_title_to_soc(self):
        """
        Matches job title to SOC code
        """
        self.data = []
        for row in self.input:
            soc_codes, std_titles = apply_model(row)
            # If not prediction, continue
            if (soc_codes is None) or (std_titles is None):
                continue
            self.data.append(
                {
                    "soc_id_components": (soc_codes, std_titles),
                    "job_id": row["id"],
                }
            )
        self.next(self.join_step)

    @step
    def join_step(self, inputs):
        for chunk in inputs:
            # Generate the SOC ID from the [soc_codes, std_titles]
            jobs = [
                {
                    "soc_id": soc_id,
                    "job_id": row["job_id"],
                    "job_data_source": "reed",
                }
                for row in chunk.data
                for soc_id in generate_soc_ids(*row.pop("soc_id_components"))
            ]
            # Write this chunk to disk
            with S3(run=self) as s3:
                first_id = jobs[0]["job_id"]
                last_id = jobs[-1]["job_id"]
                filename = f"soc_{first_id}-{last_id}-test-{self.test}.json"
                data = json.dumps(jobs)
                s3.put(filename, data)
        self.next(self.end)

    @step
    def end(self):
        """Write the data out in chunks to limit the file size as the dataset grows"""
        pass


if __name__ == "__main__":
    SocMatchFlow()
