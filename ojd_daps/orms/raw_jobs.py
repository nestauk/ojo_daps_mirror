"""
Raw Job Ads ORM.
"""


from sqlalchemy import Column
from sqlalchemy.types import VARCHAR, DATETIME
from ojd_daps import declarative_base
from ojd_daps.orms.common import fixture

Base = declarative_base()


class RawJobAd(Base):
    __tablename__ = "raw_job_adverts"
    id = Column(VARCHAR(50), primary_key=True)  # Based on the dataset
    data_source = Column(VARCHAR(10), primary_key=True)  # "Reed", "Indeed", etc
    created = Column(DATETIME, index=True)
    url = fixture("text")
    s3_location = fixture("text")
    job_title_raw = fixture("text")
    job_location_raw = fixture("text")
    job_salary_raw = fixture("text")
    company_raw = fixture("text")
    contract_type_raw = fixture("text")
    closing_date_raw = fixture("text")
    description = fixture("text")


class ReedAdDetail(Base):
    __tablename__ = "reed_ad_details"
    id = Column(VARCHAR(50), primary_key=True)
    type = fixture("text")
    sector = fixture("text")
    parent_sector = fixture("text")
    knowledge_domain = fixture("text")
    occupation = fixture("text")

class JobAdDescriptionVector(Base):
    __tablename__ = "job_ad_description_vectors"
    id = Column(VARCHAR(50), primary_key=True)
    vector = fixture("text")
