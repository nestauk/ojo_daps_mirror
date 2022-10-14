"""
tasks.extract
-------------

Task for extracting raw data from S3 and curating to the database.
The database is knocked down between curations.

The DAG for this task is slightly modified from the usual Curate->MFTask
structure:

  Root ─> Curate ┬─────────────┬─> Metaflow(extract) ─> Metaflow(combine)
                 └─> PrepareDB ┘

NB, this could be resolved as:

  Root ─> Curate ─> PrepareDB ─> Metaflow(extract) ─> Metaflow(combine)

but we choose the first DAG (with the loop) so that Metaflow(extract) remains
a direct child of Curate. In order to accommodate the loop in the DAG
there is a slight modification to the base CurateTask.requires
and CurateTask.input methods implemented in JobCurateTask
"""

import importlib.util
from copy import deepcopy
import json
from datetime import datetime as dt
from dateutil.parser import parse as parse_date
from pathlib import Path
import inspect

import luigi
from daps_utils import (
    CurateTask,
    DapsRootTask,
    DapsTaskMixin,
    ForceableTask,
    MetaflowTask,
)
from daps_utils.db import get_mysql_engine
from luigi.contrib.mysqldb import MySqlTarget
from ojd_daps import __basedir__, config
from ojd_daps.orms.raw_jobs import RawJobAd

CONFIG = config["extract"]


def is_task(obj):
    """Return true is obj is a luigi Task, otherwise False"""
    if isinstance(obj, type):
        return luigi.task.Task in inspect.getmro(obj)
    return False


class _DictParamEncoderPlus(luigi.parameter._DictParamEncoder):
    """
    JSON encoder for `DictParameterPlus`, which makes `Task`s into
    JSON serializable objects.
    """

    def default(self, obj):
        try:
            return super().default(obj)
        except TypeError:
            if is_task(obj):
                return obj.get_task_family()
        raise


class DictParameterPlus(luigi.DictParameter):
    """DictParameter with values that may include a luigi `Task`"""

    def __init__(self, encoder=_DictParamEncoderPlus, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = encoder

    def serialize(self, x):
        return json.dumps(x, cls=self.encoder)


def iterbase():
    """Yields every ORM Base, from every module in the orms directory"""
    # Alias the spec_from_file_location function, it's too long
    spec_from_loc = importlib.util.spec_from_file_location
    # Iterate over files in the orms directory
    orm_path = Path(__basedir__) / "orms"
    for path in orm_path.resolve().iterdir():
        if path.suffix != ".py":
            continue
        # Load the module for this python file
        spec = spec_from_loc("ojd_daps", str(path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # Yield Base, if it exists
        try:
            yield mod.Base
        except AttributeError:
            continue


def teardown(db_name):
    """Tear down every ORM table in the database"""
    engine = get_mysql_engine(database=db_name)
    for Base in iterbase():
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine)


class JobCurateTask(CurateTask):
    """A generic task for curation. Pass in a job board,
    and extract the data accordingly."""

    job_board = luigi.Parameter()
    exclude_fields = luigi.ListParameter(default=[])
    drop_db = luigi.BoolParameter()

    def curate_data(self, s3path):
        """Fairly standard implementation of the abstract curate_data method"""
        for chunk in self.retrieve_data(s3path, f"extract-{self.job_board}"):
            data = [
                {k: v for k, v in row.items()} for row in chunk if row["id"] is not None
            ]
            # Standardise any fields
            for row in data:
                row["created"] = parse_date(row["created"], dayfirst=True)
            yield data

    def requires(self):
        """
        Curate ┬──────────────> Metaflow(extract)
               └─> PrepareDB

        Override the base CurateTask.requires method so that
        we can yield PrepareDatabaseTask as well as MetaflowTask.
        Note that this causes a problem in CurateTask.run, which
        expects just a single Task dependency to be returned
        from CurateTask.input(), and so a custom input() method
        is also defined.
        """
        mftask_kwargs = dict(
            flow_path=self.flow_path,
            flow_tag=self.db_name,
            rebuild_base=self.rebuild_base,
            rebuild_flow=self.rebuild_flow,
            flow_kwargs=self.flow_kwargs,
            preflow_kwargs=self.preflow_kwargs,
            container_kwargs=self.container_kwargs,
            requires_task=self.requires_task,
            requires_task_kwargs=self.requires_task_kwargs,
            stale_breadcrumb=self.stale_breadcrumb,
        )
        yield MetaflowTask(**mftask_kwargs)
        yield PrepareDatabaseTask(
            drop_db=self.drop_db,
            mftask_kwargs=mftask_kwargs,
        )

    def input(self):
        """
        Custome input() method to return only the MetaflowTask
        so that CurateTask.run can operate as expected bearing in mind
        the modified CurateTask.requires method.
        """
        requires = filter(lambda task: isinstance(task, MetaflowTask), self.requires())
        mftask = list(requires)[0]
        return luigi.task.getpaths(mftask)


class PrepareDatabaseTask(ForceableTask, DapsTaskMixin):
    date = luigi.DateParameter(default=dt.now())  # Ensures once per day
    drop_db = luigi.BoolParameter()
    mftask_kwargs = DictParameterPlus()

    def output(self):
        conf = config["mysqldb"]["mysqldb"]
        conf["database"] = self.db_name
        return MySqlTarget(update_id=self.task_id, **conf)

    def run(self):
        """Teardown and refresh the database / models"""
        if self.drop_db:
            teardown(self.db_name)
        self.output().touch()

    def requires(self):
        """PrepareDB ──> Metaflow(extract)"""
        yield MetaflowTask(**self.mftask_kwargs)


class RootTask(DapsRootTask):
    """Yields all curation tasks from config"""

    drop_db = luigi.BoolParameter(
        default=True, parsing=luigi.BoolParameter.EXPLICIT_PARSING
    )

    def requires(self):
        """Root ─> Curate"""
        # Need to copy CONFIG as we need to pop, recalling that luigi
        # runs 'requires' several times (thus re-popping causes errors)
        config_copy = deepcopy(CONFIG)
        for job_board, config in config_copy.items():
            exclude_fields = config[self.db_name].pop("exclude_fields")

            # Metaflow(extract) ──> Metaflow(combine)
            combine_config = deepcopy(config[self.db_name])
            combine_config["flow_kwargs"]["job_board"] = job_board
            combine_task_kwargs = dict(
                flow_path="collect/combine_raw_data.py",
                force=True,
                flow_tag=self.db_name,
                **combine_config,
            )

            yield JobCurateTask(
                orm=RawJobAd,  # Where to write the data
                flow_path=f"extract/{job_board}.py",
                job_board=job_board,
                exclude_fields=exclude_fields,
                test=self.test,
                drop_db=self.drop_db,
                force=True,  # Always force rerun
                force_upstream=True,
                requires_task=MetaflowTask,
                requires_task_kwargs=combine_task_kwargs,
                **config[self.db_name],
            )
