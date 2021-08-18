"""
Standard Features ORM.

Features link tables for the follwoing:
- Deduplication
- Cleaned job title 
- Occupation (SOC & ESCO)
- Industry (SIC)
- Geography (TTWA & NUTS3)
- Salary 
- Skills / competencies (Nesta’s skills taxonomy v2)
- Level of experience 
- Education level and qualifications 
"""


from sqlalchemy import Column, Index
from sqlalchemy.types import VARCHAR, DECIMAL
from ojd_daps import declarative_base
from ojd_daps.orms.common import fixture

Base = declarative_base()

JOB_AD_IDX = lambda: Index("raw_job_ad_idx", "job_id", "job_data_source")


class LinkBase(object):
    job_id = Column(VARCHAR(50), primary_key=True)  # Based on the dataset
    job_data_source = Column(VARCHAR(10), primary_key=True)


class JobAdDuplicateLink(Base):
    __tablename__ = "job_advert_duplicate_links"
    first_id = Column(VARCHAR(50), primary_key=True)  # Based on the dataset
    second_id = Column(VARCHAR(50), primary_key=True)  # Based on the dataset
    weight = Column(DECIMAL)

    __table_args__ = (
        Index("first_raw_job_ad_idx", "first_id"),
        Index("second_raw_job_ad_idx", "second_id"),
    )


class JobAdLocationLink(Base, LinkBase):
    __tablename__ = "job_ad_location_links"
    location_id = fixture("ipn_pk")
    __table_args__ = (
        JOB_AD_IDX(),
        Index("location_idx", "location_id"),
    )


class JobAdJobTitleLink(Base, LinkBase):
    __tablename__ = "job_ad_title_links"
    std_job_title_id = fixture("pk")
    __table_args__ = (
        JOB_AD_IDX(),
        Index("std_job_title_idx", "std_job_title_id"),
    )


class JobAdSOCLink(Base, LinkBase):
    __tablename__ = "job_ad_soc_links"
    soc_id = fixture("big_pk")
    __table_args__ = (
        JOB_AD_IDX(),
        Index("soc_idx", "soc_id"),
    )


class JobAdSICLink(Base, LinkBase):
    __tablename__ = "job_ad_sic_links"
    sic_id = fixture("pk")
    __table_args__ = (
        JOB_AD_IDX(),
        Index("sic_idx", "sic_id"),
    )


class JobAdESCOLink(Base, LinkBase):
    __tablename__ = "job_ad_esco_links"
    esco_id = fixture("pk")
    __table_args__ = (
        JOB_AD_IDX(),
        Index("esco_idx", "esco_id"),
    )


class JobAdSkillLink(Base, LinkBase):
    __tablename__ = "job_ad_skill_links"
    job_id = fixture("pk")
    surface_form = fixture("text")
    surface_form_type = fixture("text")
    preferred_label = fixture("text")
    entity = fixture("integer")
    predicted_q = fixture("float")
    cluster_0 = fixture("integer")
    cluster_1 = fixture("integer")
    cluster_2 = fixture("integer")
    label_cluster_0 = fixture("text")
    label_cluster_1 = fixture("text")
    label_cluster_2 = fixture("text")
    __table_args__ = (
        JOB_AD_IDX(),
        Index("entity_x", "entity"),
    )


class JobAdTTWALink(Base, LinkBase):
    __tablename__ = "job_ad_ttwa_links"
    ttwa_id = fixture("pk")
    __table_args__ = (
        JOB_AD_IDX(),
        Index("ttwa_idx", "ttwa_id"),
    )


class JobAdNUTS3Link(Base, LinkBase):
    __tablename__ = "job_ad_nuts3_links"
    nuts3_id = fixture("pk")
    __table_args__ = (
        JOB_AD_IDX(),
        Index("nuts3_idx", "nuts3_id"),
    )


class JobAdStdSalaryLink(Base, LinkBase):
    __tablename__ = "job_ad_std_salary_links"
    std_salary_id = fixture("pk")
    __table_args__ = (
        JOB_AD_IDX(),
        Index("std_salary_idx", "std_salary_id"),
    )


class JobAdExperienceLink(Base, LinkBase):
    __tablename__ = "job_ad_experience_links"
    experience_id = fixture("pk")
    __table_args__ = (
        JOB_AD_IDX(),
        Index("experience_idx", "experience_id"),
    )


class JobAdQualificationsLink(Base, LinkBase):
    __tablename__ = "job_ad_qualifications_links"
    qualification_id = fixture("pk")
    __table_args__ = (
        JOB_AD_IDX(),
        Index("qualification_idx", "qualification_id"),
    )


class JobAdReedAdDetailLink(Base, LinkBase):
    __tablename__ = "job_ad_reed_ad_detail_links"
    reed_ad_detail_id = fixture("pk")
    __table_args__ = (
        JOB_AD_IDX(),
        Index("reed_ad_detail_idx", "reed_ad_detail_id"),
    )
