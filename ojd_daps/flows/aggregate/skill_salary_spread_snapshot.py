"""
skill_salary_spread_snapshot_flow
---------------------------------

A Flow for aggregating extracted skills for the last full month.

The output is a json with row format:

{
    'skill': Skill name,
    'min_lower_quartile': lower quartile of minimum salary,
    'min_median': median of minimum salary,
    'min_upper_quartile': upper quartile of minimum salary,
    'max_lower_quartile': lower quartile of maximum salary,
    'max_median': median of maximum salary,
    'max_upper_quartile': upper quartile of maximum salary
}
"""
from metaflow import FlowSpec, step
from ojd_daps.flows.aggregate.common import (
    get_snapshot_ads,
    extract_features,
    most_common,
    sort_and_groupby,
    iterquantiles,
    save_data,
)
from daps_utils.flow import DapsFlowMixin

NARROW_LABEL = "label_cluster_2"
BROAD_LABEL = "label_cluster_0"


class SkillsSalarySpreadFlow(FlowSpec, DapsFlowMixin):
    @step
    def start(self):
        job_ads = get_snapshot_ads()
        self.job_ads = extract_features(job_ads, "salary", "skills")
        self.next(self.get_jobs_with_common_skills)

    @step
    def get_jobs_with_common_skills(self):
        """
        Gets the top 50 most common skills and restricts links
        to those in the top 50.
        """
        skills = list(filter(None, (ad.get(NARROW_LABEL) for ad in self.job_ads)))
        most_common_skills = most_common(skills, 50)
        self.job_ads = [
            ad for ad in self.job_ads if ad.get(NARROW_LABEL) in most_common_skills
        ]
        self.next(self.aggregate)

    @step
    def aggregate(self):
        """Sorts links by skill and groups, produces aggregations"""
        self.data = []
        for (narrow_skill, broad_label), job_ads in sort_and_groupby(
            self.job_ads, NARROW_LABEL, BROAD_LABEL
        ):
            job_ads = list(job_ads)
            weekly_agg = {
                "Narrow skill group": narrow_skill,
                "Broad skill group": broad_label,
            }
            for label, quantile in iterquantiles(job_ads):
                weekly_agg[label] = quantile
            self.data.append(weekly_agg)
        self.next(self.end)

    @step
    def end(self):
        save_data(self, "skills_salary_spread_snapshot")


if __name__ == "__main__":
    SkillsSalarySpreadFlow()
