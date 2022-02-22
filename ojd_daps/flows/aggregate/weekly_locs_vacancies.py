"""
weekly_locs_vacancies_flow
-------------------------
A Flow for aggregating weekly location counts of ads.

The weekly stocks are indexed against the
average stocks of the 4 weeks of April 2021.

The output is a json with row format:

{
    'isoweek': (Year, Month) value for each week of aggregated week,
    'nuts_2_name': Name of the location,
    'nuts_2_code': Code for the location
    'indexed_value': Weekly location stock, divided by the index value
}
"""
from metaflow import FlowSpec, step
from ojd_daps.flows.aggregate.common import (
    VOLUME_LABEL,
    _extract_features,
    get_index_stock_lookup,
    get_weekly_ads,
    save_data,
    sort_and_groupby,
    standardise_location,
    volume_calc,
)
from daps_utils.flow import DapsFlowMixin


class WeeklyLocsFlow(FlowSpec, DapsFlowMixin):
    @step
    def start(self):
        """
        Starts the flow.
        """
        job_ads = get_weekly_ads()
        self.job_ads = []
        for ad in job_ads:
            # Standardise location
            for location in _extract_features(ad, "location"):
                if ad.get.get("nuts_2_code") is None:
                    continue
                location = standardise_location(location)
                ad.update(location)
                self.job_ads.append(ad)
        self.next(self.get_index)

    @step
    def get_index(self):
        """Step that calculates the index value"""
        self.loc_index = get_index_stock_lookup(
            feature_name="location", code="nuts_2_code"
        )
        self.next(self.aggregate)

    @step
    def aggregate(self):
        """Aggregation step

        Sorts the dates, then groups all into each isoweek and location code.
        For each isoweek, outputs a dict of the isoweek
        and length of the chunk, divided by the index value.

        Saves the outputted list of dictionaries to S3.
        """
        self.data = [
            {
                "Date": str(week_date.strftime("%Y-%m-%d")),
                "Location Name": nuts_2_name,
                "Location Code": nuts_2_code,
                VOLUME_LABEL: volume_calc(job_ads, self.loc_index[nuts_2_code]),
            }
            for (week_date, nuts_2_name, nuts_2_code), job_ads in sort_and_groupby(
                self.job_ads, "week_date", "nuts_2_name", "nuts_2_code"
            )
        ]
        self.next(self.end)

    @step
    def end(self):
        save_data(self, "weekly-loc_vacancies")


if __name__ == "__main__":
    WeeklyLocsFlow()
