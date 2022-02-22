"""
weekly_salary_spread_flow
-------------------------

A Flow for aggregating weekly minimum and maximum salary spread.

The output is a json with row format:

{
    'isoweek': (Year, Month) value for each week of aggregated week,
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
    get_weekly_ads,
    extract_features,
    sort_and_groupby,
    iterquantiles,
    save_data,
)
from daps_utils.flow import DapsFlowMixin


class WeeklySalarySpreadFlow(FlowSpec, DapsFlowMixin):
    @step
    def start(self):
        job_ads = get_weekly_ads()
        self.job_ads = extract_features(job_ads, "salary")
        self.next(self.aggregate)

    @step
    def aggregate(self):
        """Aggregation step

        Sorts the dates, then groups all into each isoweek.
        For each isoweek, outputs a dict of the isoweek
        and median, 25th, and 75th percentile for the minimum
        and maximum salaries in that week.

        Saves the outputted list of dictionaries to S3.
        """
        grouped_dates = sort_and_groupby(self.job_ads, "week_date")
        self.data = []
        for isoweek, job_ads in grouped_dates:
            job_ads = list(job_ads)
            weekly_agg = {"Date": str(isoweek.strftime("%Y-%m-%d"))}
            for label, quantile in iterquantiles(job_ads):
                weekly_agg[label] = quantile
            self.data.append(weekly_agg)
        self.next(self.end)

    @step
    def end(self):
        save_data(self, "weekly-salary-spread")


if __name__ == "__main__":
    WeeklySalarySpreadFlow()
