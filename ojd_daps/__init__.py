################################################################
### Text automatically added by daps-utils metaflowtask-init ###
from .__initplus__ import load_current_version, __basedir__, load_config

try:
    config = load_config()
except ModuleNotFoundError as exc:  # For integration with setup.py
    print(exc)
    pass

__version__ = load_current_version()
################################################################


def declarative_base():
    from sqlalchemy.ext.declarative import declared_attr
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy import Column, TEXT

    class Base(object):
        @declared_attr
        def __tablename__(cls):
            return cls.__name__.lower()

        __version__ = Column(
            TEXT, nullable=False, default=__version__, onupdate=__version__
        )

    return declarative_base(cls=Base)
