from unittest import mock
from ojd_daps.tasks.extract import iterbase, is_task, DictParameterPlus
from ojd_daps.tasks.extract import luigi, CurateTask, MetaflowTask
from ojd_daps.tasks.extract import (
    teardown as my_teardown,
)  # namespace clash with pytest

from sqlalchemy.ext.declarative.api import DeclarativeMeta


def test_iterbase():
    bases = list(iterbase())
    assert all(isinstance(b, DeclarativeMeta) for b in bases)
    assert len(bases) >= 5
    assert len(set(bases)) == len(bases)


@mock.patch("ojd_daps.tasks.extract.get_mysql_engine")
@mock.patch("ojd_daps.tasks.extract.iterbase")
def test_teardown(mocked_iterbase, mocked_get_engine):
    mocked_bases = [mock.Mock() for i in range(5)]
    mocked_iterbase.return_value = mocked_bases
    my_teardown("dummy")
    for Base in mocked_bases:
        assert Base.metadata.drop_all.call_count == 1
        assert Base.metadata.create_all.call_count == 1


def test_is_task():
    # Test it works for tasks
    for task in (
        luigi.Task,
        luigi.contrib.s3.S3PathTask,
        CurateTask,
        MetaflowTask,
    ):
        assert is_task(task)
    # Test it fails for things which aren't tasks
    for not_a_task in (True, False, "task", None, [luigi.Task]):
        assert not is_task(not_a_task)


def test_DictParameterPlus():
    # Check that Tasks are serialised
    task_parameter = DictParameterPlus(default=luigi.Task)
    data = task_parameter.serialize(luigi.Task)
    assert data == '"Task"'
