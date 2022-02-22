"""
weekly_stock_flow
-----------------
A Flow for aggregating weekly stock of ads.

The weekly stocks are indexed against the
average stocks of the 4 weeks of April 2021.

The output is a json with row format:

{
    'isoweek': (Year, Month) value for each week of aggregated week,
    'indexed_value': Weekly stock, divided by the index value
}
"""
from metaflow import FlowSpec, step
from itertools import chain
from copy import deepcopy
from cardinality import count
from ojd_daps.flows.aggregate.common import (
    VOLUME_LABEL,
    STOCK_WEEKS,
    STOCK_IDX_START,
    STOCK_IDX_END,
    get_weekly_ads,
    iterdates,
    get_snapshot_ads,
    DEFAULT_START_DATE,
    DEFAULT_END_DATE,
    iterquantiles,
    extract_features,
    _extract_features,
    standardise_location,
    sort_and_groupby,
    volume_calc,
    save_data,
)
from daps_utils.flow import DapsFlowMixin


def flatten_locations(job_ads):
    _job_ads = []
    for ad in map(deepcopy, job_ads):
        # Flatten if multiple locations (note this never happens with the current alg)
        for location in _extract_features(ad, "location"):
            # Standardise location
            if location.get("nuts_2_code") is None:
                continue
            location = standardise_location(location)
            # Copy, update and append the ad
            _ad = deepcopy(ad)
            _ad.update(location)
            _job_ads.append(_ad)
    return _job_ads


class WeeklyStockFlow(FlowSpec, DapsFlowMixin):
    @step
    def start(self):
        """Step that calculates the index value."""
        self.job_ads = get_weekly_ads(
            start_date=STOCK_IDX_START, end_date=STOCK_IDX_END
        )
        self.index_value = len(self.job_ads) / STOCK_WEEKS
        self.next(self.prepare_stock_lookup)

    @step
    def prepare_stock_lookup(self):
        job_ads = flatten_locations(self.job_ads)
        self.loc_index = {
            _code: count(chunk) / STOCK_WEEKS
            for (_code,), chunk in sort_and_groupby(job_ads, "nuts_2_code")
        }
        self.next(self.prepare_week_ranges)

    @step
    def prepare_week_ranges(self):
        self.week_ranges = list(
            iterdates(start_date=DEFAULT_START_DATE, end_date=DEFAULT_END_DATE)
        )
        self.next(self.aggregate, foreach="week_ranges")

    @step
    def aggregate(self):
        """Aggregation step"""
        from_date, to_date = self.input  # unpack input
        print("Retrieving snapshot for", from_date, to_date)
        date = to_date.strftime("%Y-%m-%d")
        job_ads = get_snapshot_ads(from_date=from_date, to_date=to_date)

        # Weekly stock
        self.weekly_stock = {
            "Date": date,
            VOLUME_LABEL: volume_calc(job_ads, self.index_value),
        }

        # Weekly stock by location
        _job_ads = flatten_locations(job_ads)
        self.weekly_loc_vacancies = [
            {
                "Date": date,
                "Location Name": nuts_2_name,
                "Location Code": nuts_2_code,
                VOLUME_LABEL: volume_calc(ads, self.loc_index[nuts_2_code]),
            }
            for (nuts_2_name, nuts_2_code), ads in sort_and_groupby(
                _job_ads, "nuts_2_name", "nuts_2_code"
            )
        ]

        # Weekly salary spread
        _job_ads = extract_features(deepcopy(job_ads), "salary")
        self.weekly_salary_spread = dict(Date=date, **dict(iterquantiles(_job_ads)))
        self.next(self.join)

    @step
    def join(self, inputs):
        """Join the aggregation step"""
        # Weekly stock
        self.data = [input.weekly_stock for input in inputs]
        save_data(self, "weekly_stock")

        # Weekly stock by location
        self.data = list(
            chain.from_iterable(input.weekly_loc_vacancies for input in inputs)
        )
        save_data(self, "weekly_loc_vacancies")

        # Weekly salary spread
        self.data = [input.weekly_salary_spread for input in inputs]
        save_data(self, "weekly_salary_spread")
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    WeeklyStockFlow()
