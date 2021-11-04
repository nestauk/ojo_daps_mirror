from pathlib import Path

from daps_utils.db import db_session, get_orm_base

from ojd_daps import __version__


def _test_orm_builds(orm_name):
    """Tests that the ORM builds, and that the Base class has the __version__ parameter set"""
    with db_session() as session:
        engine = session.get_bind()
        try:
            Base = get_orm_base(orm_name)
        except AttributeError:  # Not every file is an ORM!
            return
        assert Base().__version__.default.arg == __version__
        Base.metadata.create_all(engine)
        Base.metadata.drop_all(engine)


def test_orms_build():
    orm_path = Path(__file__).parent.parent
    for file_path in orm_path.iterdir():
        if file_path.suffix != ".py":
            continue
        print("Testing", file_path)
        _test_orm_builds(orm_name=file_path.stem)
