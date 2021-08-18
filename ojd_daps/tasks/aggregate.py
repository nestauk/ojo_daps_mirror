"""
aggregate_task
--------------

A task that picks up aggregations flows defined in the config,
and curates their output to a specified ORM.

"""
import luigi
from copy import deepcopy
from daps_utils import CurateTask, DapsRootTask
from ojd_daps import config
from ojd_daps.orms.indicators import Jobs_by_Location

CONFIG = config["aggregate"]
orm_dict = {
    "jobs_by_locs": Jobs_by_Location
}  # for matching enrichment to ORM in Root task


class AggregateCurateTask(CurateTask):
    aggregate = luigi.Parameter()
    flow_tag = luigi.Parameter()
    exclude_fields = luigi.ListParameter(default=[])

    def curate_data(self, s3path):
        data = [
            {k: v for k, v in row.items() if k not in self.exclude_fields}
            for chunk in self.retrieve_data(s3path, self.aggregate)
            for row in chunk
        ]
        return data


class RootTask(DapsRootTask):
    def requires(self):
        """
        Iterate over YAML files in config/aggregate, and yield
        flows which have been specified.
        """
        config_copy = deepcopy(CONFIG)
        for aggregation, conf in config_copy.items():
            yield AggregateCurateTask(
                orm=orm_dict[aggregation],
                flow_path=f"aggregate/{aggregation}.py",
                aggregate=aggregation,
                test=self.test,
                flow_tag=self.db_name,
                force=True,  # Always force rerun
                force_upstream=True,
                **conf[self.db_name],
            )
