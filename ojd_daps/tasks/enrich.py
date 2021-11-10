"""Enrichment task for ojd_daps. Typical call syntax for local tests:

PYTHONPATH=. luigi --local-scheduler --module enrich RootTask [--specific-task-name locations]

--specific-task-name is an optional parameter that enables a reduced pipeline with only
the named task.
"""

import luigi
from copy import deepcopy
from daps_utils import (
    CurateTask,
    DapsRootTask,
    DapsTaskMixin,
)
from daps_utils.db import db_session
from ojd_daps import config
from ojd_daps.orms.link_tables import (
    JobAdLocationLink,
    JobAdSkillLink,
    JobAdSOCLink,
    JobAdDuplicateLink,
)
from ojd_daps.orms.raw_jobs import JobAdDescriptionVector
from ojd_daps.orms.std_features import Location, Salary, SOC, RequiresDegree
from sqlalchemy_utils.functions import get_declarative_base

CONFIG = config["enrich"]
PRECONFIG = config["pre_enrich"]
orm_dict = {
    # "Enrichment" Tasks
    "locations": JobAdLocationLink,
    "salaries": Salary,
    "skills": JobAdSkillLink,
    # "soc": JobAdSOCLink, # Broken, see https://github.com/nestauk/ojd_daps/issues/320
    "requires_degree": RequiresDegree,
    "deduplication": JobAdDuplicateLink,
    # "Pre-Enrichment" Tasks
    "vectorise_descriptions": JobAdDescriptionVector,
    "location_lookup": Location,
    # "soc_lookup": SOC, # Broken, see https://github.com/nestauk/ojd_daps/issues/320
}  # for matching each task to an output ORM


class EnrichmentCurateTask(CurateTask):
    file_prefix = luigi.Parameter()
    flow_tag = luigi.Parameter()
    exclude_fields = luigi.ListParameter(default=[])

    def curate_data(self, s3path):
        for chunk in self.retrieve_data(s3path, self.file_prefix):
            data = [
                {k: v for k, v in row.items() if k not in self.exclude_fields}
                for row in chunk
            ]
            yield data


class PreEnrichmentTasks(luigi.WrapperTask, DapsTaskMixin):
    """A wrapper for all tasks that need to take place before enrichment"""

    def requires(self):
        pre_config_copy = deepcopy(PRECONFIG)
        for flow_name, conf in pre_config_copy.items():
            # Don't repeat this pre-enrichment step if there is already data
            # in the database (useful for speeding up development!)
            if flow_name not in orm_dict:
                continue
            orm = orm_dict[flow_name]
            Base = get_declarative_base(orm)
            with db_session(self.db_name) as session:
                engine = session.get_bind()
                Base.metadata.create_all(engine)  # Guarantee the table exists
                n_objs = len(list(session.query(orm).limit(1)))
            if n_objs > 0:
                continue
            # Otherwise yield the pre-enrichment step
            yield EnrichmentCurateTask(
                orm=orm,
                flow_path=f"pre_enrich/{flow_name}.py",
                file_prefix=flow_name,
                test=self.test,
                flow_tag=self.db_name,
                force=True,  # Always force rerun
                force_upstream=True,
                **conf[self.db_name],
            )


class RootTask(DapsRootTask):

    specific_task_name = luigi.Parameter(default=None)

    def requires(self):
        """
        Iterate over YAML files in config/enrich, and yield
        flows which have been specified.
        """
        config_copy = deepcopy(CONFIG)
        requires_task_kwargs = {
            "test": self.test,
        }
        for flow_name, conf in config_copy.items():
            if flow_name not in orm_dict or (
                self.specific_task_name is not None
                and self.specific_task_name != flow_name
            ):
                continue
            yield EnrichmentCurateTask(
                orm=orm_dict[flow_name],
                flow_path=f"enrich/{flow_name}.py",
                file_prefix=flow_name,
                test=self.test,
                flow_tag=self.db_name,
                force=True,  # Always force rerun
                force_upstream=True,
                requires_task=PreEnrichmentTasks,
                requires_task_kwargs=requires_task_kwargs,
                **conf[self.db_name],
            )
