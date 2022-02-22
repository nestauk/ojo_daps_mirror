"""
weekly_soc_vacancies_flow
-------------------------
A Flow for aggregating weekly SOC counts of ads.

The weekly stocks are indexed against the
average stocks of the 4 weeks of February 2021.

The output is a json with row format:

{
    'isoweek': (Year, Month) value for each week of aggregated week,
    'soc': Name of the SOC,
    'indexed_value': Weekly SOC stock, divided by the index value
    }
"""
from metaflow import FlowSpec, step
from ojd_daps.flows.aggregate.common import (
    VOLUME_LABEL,
    _extract_features,
    get_index_stock_lookup,
    get_weekly_ads,
    sort_and_groupby,
    volume_calc,
    save_data,
)
from ojd_daps.flows.enrich.labs.soc.metadata_utils import (
    load_json_from_s3,
)

from daps_utils.flow import DapsFlowMixin


class WeeklySOCFlow(FlowSpec, DapsFlowMixin):
    @step
    def start(self):
        level_name_lookup = {
            str(soc["soc_group_code"]): soc["soc_group_title"]
            for soc in load_json_from_s3("soc_code_title_lookup")
        }

        self.job_ads = []
        for ad in get_weekly_ads():
            for soc in _extract_features(ad, "soc"):
                # Truncate SOC to level 1
                soc_1_code = soc["soc_code"][0]
                soc["soc_code"] = soc_1_code
                soc["soc_name"] = level_name_lookup[soc_1_code]
                ad.update(soc)
                self.job_ads.append(ad)
        self.next(self.get_index)

    @step
    def get_index(self):
        """Step that calculates the index value"""
        self.soc_index = get_index_stock_lookup("soc", "soc_code")
        self.next(self.aggregate)

    @step
    def aggregate(self):
        """Aggregation step

        Sorts the dates, then groups all into each isoweek and SOC code.
        For each isoweek, outputs a dict of the isoweek
        and length of the chunk, divided by the index value.

        Saves the outputted list of dictionaries to S3.
        """
        self.data = [
            {
                "Date": str(week_date.strftime("%Y-%m-%d")),
                "SOC Name": soc_name,
                VOLUME_LABEL: volume_calc(job_ads, self.soc_index[soc_code]),
            }
            for (week_date, soc_name, soc_code), job_ads in sort_and_groupby(
                self.job_ads, "week_date", "soc_name", "soc_code"
            )
        ]
        self.next(self.end)

    @step
    def end(self):
        save_data(self, "weekly-soc_vacancies")


if __name__ == "__main__":
    WeeklySOCFlow()
