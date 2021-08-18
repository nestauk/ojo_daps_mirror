from unittest import mock
from sqlalchemy.ext.declarative.api import DeclarativeMeta

# functions to test
from ojd_daps.tasks.dump import get_indicator_orms, query_orm, save_to_s3, make_s3_keys

# globals
from ojd_daps.tasks.dump import BASE, BUCKET, __version__, VERSION_FMT, LATEST_FMT

PATH = "ojd_daps.tasks.dump.{}"


def test_get_indicator_orms():
    """Tests that ORMs are extracted from ojd_daps.orms.indicators"""
    assert get_indicator_orms() is get_indicator_orms()  # i.e cached
    orms = get_indicator_orms()
    assert len(orms) > 0
    assert BASE not in orms
    assert all(isinstance(orm, DeclarativeMeta) for orm in orms)


@mock.patch(PATH.format("db_session"))
@mock.patch(PATH.format("object_as_dict"), side_effect=lambda x: x)
def test_query_orm(mocked_obj2dict, mocked_session):
    # Mock up an ORM
    for _orm in get_indicator_orms():
        mocked_orm = mock.create_autospec(_orm)  # Spec the ORM on an actual ORM
        break
    mocked_orm.__table__.columns = []
    # Mock add column fields to the ORM
    for name in ["foo", "bar", "baz", "boom"]:  # a list of fields in the ORM
        col = mock.Mock(spec=["name"])
        col.name = name
        mocked_orm.__table__.columns.append(col)

    # Set up query mocking
    session = mocked_session().__enter__()
    session.query.return_value.all.return_value = [1, 2, 3]

    # Run the mocked query and check the fields have been stripped
    data = query_orm(mocked_orm, strip_fields=["foo", "boom"])
    cols, _ = session.query.call_args
    assert data == [1, 2, 3]
    assert list(col.name for col in cols) == ["bar", "baz"]


@mock.patch(PATH.format("boto3"))
def test_save_to_s3(mocked_boto):
    save_to_s3("table_key", "version_key", [1, 2, 3])
    s3 = mocked_boto.resource()
    arg_list = [args for args, kwargs in s3.Object.call_args_list]
    assert arg_list == [(BUCKET, "table_key"), (BUCKET, "version_key")]

    kwarg_list = [kwargs for args, kwargs in s3.Object().put.call_args_list]
    assert kwarg_list == [{"Body": "[1, 2, 3]"}, {"Body": __version__}]


def test_make_s3_keys():
    task = mock.Mock()
    task.orm.__tablename__ = "a_table_name"
    task.db_name = "a_db_name"
    for fmt in (VERSION_FMT, LATEST_FMT):
        table_key, version_key = make_s3_keys(task, fmt)
        assert table_key.endswith("a_table_name.json")
        assert version_key.endswith("__version__.json")
