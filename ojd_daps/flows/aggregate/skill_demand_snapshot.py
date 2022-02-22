"""
skills_demand_snapshot_flow
----------------------

A Flow for aggregating extracted skills for the last full month.
"""
from metaflow import FlowSpec, step
from daps_utils.flow import DapsFlowMixin
from ojd_daps.flows.aggregate.common import (
    get_snapshot_ads,
    extract_features,
    sort_and_groupby,
    save_data,
)


class SkillsDemandFlow(FlowSpec, DapsFlowMixin):
    @step
    def start(self):
        job_ads = get_snapshot_ads()
        self.job_ads = extract_features(job_ads, "skills")
        self.next(self.aggregate)

    @step
    def aggregate(self):
        job_ads = list(
            filter(lambda ad: ad.get("label_cluster_0") is not None, self.job_ads)
        )
        self.data = [
            {
                "Level 0 skill groups": cluster_0,
                "Level 1 skill groups": cluster_1,
                "Level 2 skill groups": cluster_2,
                "Percentage": 100 * (len(list(chunk)) / len(job_ads)),
            }  # number of skills in cluster
            for (cluster_0, cluster_1, cluster_2), chunk in sort_and_groupby(
                job_ads, "label_cluster_0", "label_cluster_1", "label_cluster_2"
            )
        ]
        self.next(self.end)

    @step
    def end(self):
        save_data(self, "skills_demand_snapshot")


if __name__ == "__main__":
    SkillsDemandFlow()
