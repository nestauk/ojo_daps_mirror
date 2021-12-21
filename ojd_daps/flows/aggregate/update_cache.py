"""Updates the data getters cache for the default date ranges."""
from cardinality import count

from daps_utils import DapsFlowMixin

from metaflow import FlowSpec, retry, step

from ojd_daps.dqa.data_getters import (
    DATE_FORMAT,
    cache,
    get_duplicate_ids,
    get_duplicate_subgraphs,
    get_features,
    get_snapshot_ads,
    get_subgraphs_by_location,
    iterdates,
)

TEST_CAP = 2


class CacheFlow(FlowSpec, DapsFlowMixin):
    @step
    def start(self):
        """Clear the cache and fanout over dates."""
        cache.clear()
        dates = list(iterdates())
        self.dates = dates[: TEST_CAP if self.test else None]
        print("Got", len(self.dates), "of", len(dates), "dates")
        self.next(self.get_duplicate_subgraphs)

    @retry
    @step
    def get_duplicate_subgraphs(self):
        """Generate all duplicate subgraphs"""
        total = count(get_duplicate_subgraphs(min_weight=0.95, max_weight=1))
        print("Got", total, "duplicate subgraphs")
        if total < 100_000:  # Should be 100ks
            raise ValueError("Too few duplicate subgraphs found")
        self.next(self.get_subgraphs_by_location)

    @retry
    @step
    def get_subgraphs_by_location(self):
        """
        Building on top of the previous step, split subgraphs
        by geographic location so that two jobs in different locations
        don't count as duplicates.
        """
        total = count(get_subgraphs_by_location(min_weight=0.95, max_weight=1))
        print("Got", total, "subgraphs by location")
        if total < 100_000:  # Should be 100ks
            raise ValueError("Too few subgraphs by location found")
        self.next(self.get_duplicate_ids, foreach="dates")

    @retry
    @step
    def get_duplicate_ids(self):
        """Convert the subgraphs into lists of lists duplicates for this date range."""
        from_date, to_date = self.input
        total = count(
            get_duplicate_ids(
                min_weight=0.95,
                max_weight=1,
                split_by_location=True,
                from_date=from_date.strftime(DATE_FORMAT),
                to_date=from_date.strftime(DATE_FORMAT),
            )
        )
        print("Got", total, "duplicate ids for", from_date, to_date)
        self.total = total
        self.next(self.join)

    @step
    def join(self, inputs):
        """Merge the results and sense check the numbers."""
        self.merge_artifacts(inputs, include=["dates"])  # Persist the 'dates' artefact
        total = sum(input.total for input in inputs)
        print("Got total", total, "duplicate ids")
        if total < 100:  # Expect 1ks
            raise ValueError("Too few duplicate ids")
        self.next(self.get_features)

    @retry
    @step
    def get_features(self):
        """Download and wrangle features by job ad."""
        feats = get_features()
        print("Got", len(feats), "ad features")
        if len(feats) < 1_000_000:
            raise ValueError("Too few features extracted")
        self.next(self.get_snapshot_ads, foreach="dates")

    @retry
    @step
    def get_snapshot_ads(self):
        """For each date range, generate the snapshot ads."""
        from_date, to_date = self.input
        ads = get_snapshot_ads(from_date=from_date, to_date=to_date)
        print("Got", len(ads), "for", from_date, to_date)
        if len(ads) < 10_000:  # Min per window
            raise ValueError("Too few ads found in this window")
        self.next(self.upload_to_s3)

    @retry
    @step
    def upload_to_s3(self, inputs):
        """Upload the updated cache to S3."""
        cache.read_only = False
        cache.upload_to_s3()
        self.next(self.end)

    @step
    def end(self):
        """Implement a dummy end to flow."""
        pass


if __name__ == "__main__":
    CacheFlow()
