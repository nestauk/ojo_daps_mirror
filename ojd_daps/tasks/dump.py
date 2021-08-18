"""
dump
----

A task for dumping a table for each ORM in ojd_daps.orms.indicators to S3.

In practice four files are generated for each ORM for a given {production, dev} label:

  1) Two tables for the ORM, saved to {__version__, latest}:
     BUCKET/{production,dev}/{__version__,latest}/{table_name}.json
  2) The version of the data, for bookkeeping purposes, saved to {__version__, latest}:
     BUCKET/{production,dev}/{__version__,latest}/__version__.json

Clearly BUCKET/{production,dev}/{__version__}/__version__.json is redundant,
but it simplifies the procedure and is otherwise harmless.

The intention is for the "versioned" S3 path to allow e.g. visualisations or
analysis to be pinned to specific version, while the "latest" S3 path is
a static URL corresponding to the "bleeding edge" of the data. There is also
enough information in the "latest" S3 path to determine the corresponding
"versioned" path.
"""

import boto3
import json
from functools import lru_cache
from daps_utils.db import object_as_dict, db_session
from daps_utils.tasks import DapsRootTask, ForceableTask, DapsTaskMixin
from daps_utils.parameters import SqlAlchemyParameter
from luigi.contrib.s3 import S3Target
from ojd_daps.orms import indicators
from ojd_daps import __version__


BASE = indicators.Base
BUCKET = "open-jobs-indicators"
VERSION_FMT = f"{{}}/{__version__}/{{}}.json"
LATEST_FMT = "{}/latest/{}.json"


@lru_cache()  # Cache because luigi evaluates the DAG several times
def get_indicator_orms():
    """Return every ORM in ojd_daps.orms.indicators"""
    objects = indicators.__dict__.values()  # All objects in ojd_daps.orms.indicators
    classes = filter(lambda obj: isinstance(obj, type), objects)  # All classes
    orms = filter(lambda cls: BASE in cls.mro()[1:], classes)  # All ORMs, excl. Base
    return list(orms)


def query_orm(orm, strip_fields=["__version__"]):
    """
    Return all rows of data from the table corresponding to this ORM.
    Exclude any fields from the output named in strip_fields
    """
    columns = filter(lambda col: col.name not in strip_fields, orm.__table__.columns)
    with db_session("production") as session:
        query = session.query(*columns)
        return list(map(object_as_dict, query.all()))


def save_to_s3(table_key, version_key, data):
    """
    Convert the data to JSON and save to S3 at the location:

        https://s3.console.aws.amazon.com/s3/object/{BUCKET}?prefix={db_name}

    whilst also saving the value of __version__ to the same directory for bookkeeping.
    """
    json_data = json.dumps(data)
    s3 = boto3.resource("s3")
    # Write the data
    s3.Object(BUCKET, table_key).put(Body=json_data)  # Save the table
    s3.Object(BUCKET, version_key).put(Body=__version__)  # Save the data version
    # Make the data public
    s3.ObjectAcl(BUCKET, table_key).put(ACL="public-read")
    s3.ObjectAcl(BUCKET, version_key).put(ACL="public-read")


def make_s3_keys(task, fmt):
    """
    Generates S3 paths to the outputs for this task.

    Args:
        task (DumpTask): A DumpTask instance for a given ORM
        fmt (str): A format string, expecting two parameters (table and db name)
    Returns:
        table_key (str): S3 Path to the dumped table output
        version_key (str): S3 Path to a file specifying the __version__ of this data
    """
    table_key = fmt.format(task.db_name, task.orm.__tablename__)
    version_key = fmt.format(task.db_name, "__version__")
    return table_key, version_key


class DumpTask(ForceableTask, DapsTaskMixin):
    """Task to dump the data in the specified ORM to S3"""

    orm = SqlAlchemyParameter()

    def make_s3_keys(self):
        """Convenience method for generating two sets of S3 keys,

        1) A "versioned" S3 path: will allow e.g. visualisations or
           analysis to be pinned to specific version.

        2) The "latest" S3 path: this is a fixed URL corresponding to
           the "bleeding edge" of the data.
        """
        # Write the data twice:
        for fmt in (VERSION_FMT, LATEST_FMT):
            yield make_s3_keys(self, fmt)

    def output(self):
        """Specify all of the output paths of this Task"""
        for table_key, version_key in self.make_s3_keys():
            return S3Target(f"s3://{BUCKET}/{table_key}")

    def run(self):
        """Write to all of the output paths of this Task"""
        data = query_orm(self.orm)
        # To avoid confusion downstream: don't write out data if there isn't any
        if not data:
            return
        for table_key, version_key in self.make_s3_keys():
            save_to_s3(table_key, version_key, data)


class RootTask(DapsRootTask):
    """Wrapper task to dump tables for each ORM to S3"""

    def requires(self):
        """Yield one Task per ORM"""
        for orm in get_indicator_orms():
            yield DumpTask(orm=orm, test=self.test, force=True)
