from sqlalchemy import Column, Index
from sqlalchemy.types import INTEGER
from ojd_daps import declarative_base
from ojd_daps.orms.common import fixture

Base = declarative_base()

class Jobs_by_Location(Base):
    __tablename__ = "jobs_by_locations"
    location_id = Column(INTEGER, primary_key=True)
    count = fixture('integer')