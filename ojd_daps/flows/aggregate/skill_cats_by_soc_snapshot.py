"""
skills_cat_by_loc_snapshot_flow
----------------------

A Flow for aggregating extracted skills by SOC for the last full month.

"""
from metaflow import FlowSpec, step
from ojd_daps.flows.aggregate.common import (
    get_snapshot_ads,
    aggregate_skills,
    extract_features,
    most_common,
    save_data,
)
from daps_utils.flow import DapsFlowMixin


class SkillsCatbySOCFlow(FlowSpec, DapsFlowMixin):
    @step
    def start(self):
        job_ads = get_snapshot_ads()
        self.job_ads = extract_features(job_ads, "soc", "skills")
        self.next(self.filter_most_common_soc)

    @step
    def filter_most_common_soc(self):
        """
        Filter job ads only with the most common SOC codes
        """
        soc_codes = map(lambda ad: ad.get("soc_code"), self.job_ads)
        soc_codes = filter(None, soc_codes)  # Ignore jobs not matched to SOC
        common_socs = most_common(soc_codes, 15)
        self.job_ads = list(
            filter(lambda ad: ad.get("soc_code") in common_socs, self.job_ads)
        )
        self.next(self.aggregate_data)

    @step
    def aggregate_data(self):
        self.data = aggregate_skills(self.job_ads, "soc_code", "soc_title", "SOC")
        self.next(self.end)

    @step
    def end(self):
        save_data(self, "skills_cats_by_soc_snapshot")


if __name__ == "__main__":
    SkillsCatbySOCFlow()
