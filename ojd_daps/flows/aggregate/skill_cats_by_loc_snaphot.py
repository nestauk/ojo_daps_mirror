"""
skills_cat_by_loc_snapshot_flow
----------------------

A Flow for aggregating extracted skills by loc for the last full month.
"""
from metaflow import FlowSpec, step
from ojd_daps.flows.aggregate.common import (
    get_snapshot_ads,
    extract_features,
    aggregate_skills,
    standardise_location,
    save_data,
)
from daps_utils.flow import DapsFlowMixin


class SkillsCatbyLocFlow(FlowSpec, DapsFlowMixin):
    @step
    def start(self):
        job_ads = get_snapshot_ads()
        job_ads = extract_features(job_ads, "location", "skills")
        self.job_ads = list(map(standardise_location, job_ads))
        self.next(self.aggregate_data)

    @step
    def aggregate_data(self):
        self.data = aggregate_skills(
            self.job_ads, "nuts_2_code", "nuts_2_name", "Location"
        )
        self.next(self.end)

    @step
    def end(self):
        save_data(self, "skills_cats_by_loc_snapshot")


if __name__ == "__main__":
    SkillsCatbyLocFlow()
