"""
Indicators ORM.

Features all information aimed to be extracted from raw job adverts:
- Cleaned job title
- Occupation (SOC & ESCO)
- Industry (SIC)
- Geography (Lat/Lon, TTWA & NUTS3)
- Salary
- Requires degree (true/false)
- Skills / competencies (Nesta’s skills taxonomy v2)
- Level of experience
- Education level and qualifications
"""


from sqlalchemy import Column
from sqlalchemy.types import VARCHAR
from ojd_daps import declarative_base
from ojd_daps.orms.common import fixture

Base = declarative_base()


class SOC(Base):
    __tablename__ = "SOC_codes"
    soc_id = fixture('big_pk')  # An md5 hash of {code + title}
    soc_code = Column(VARCHAR(5), index=True)
    soc_title = fixture("text")


class SIC(Base):
    __tablename__ = "SIC_codes"
    id = Column(VARCHAR(5), primary_key=True)
    label = fixture("text")


class ESCO(Base):
    __tablename__ = "ESCO_codes"
    id = Column(VARCHAR(5), primary_key=True)
    label = fixture("text")


class Salary(Base):
    __tablename__ = "salaries"
    id = fixture("pk")
    min_salary = fixture("salary")
    max_salary = fixture("salary")
    rate = Column(VARCHAR(9), nullable=True, index=True)
    min_annualised_salary = fixture("salary")
    max_annualised_salary = fixture("salary")


class RequiresDegree(Base):
    __tablename__ = "requires_degree"
    id = fixture("pk")
    requires_degree = fixture("boolean")

class StdJobTitle(Base):
    __tablename__ = "std_job_titles"
    id = fixture("pk")
    label = fixture("text")


class TTWA(Base):
    __tablename__ = "ttwas"
    id = Column(VARCHAR(20), primary_key=True)
    label = fixture("text")


class NUTS3(Base):
    __tablename__ = "nuts3_codes"
    id = Column(VARCHAR(20), primary_key=True)
    label = fixture("text")


class Skill(Base):
    __tablename__ = "skills"
    id = Column(VARCHAR(20), primary_key=True)
    label = fixture("text")


class Experience(Base):
    __tablename__ = "experiences"
    id = fixture("pk")
    label = fixture("text")


class Qualification(Base):
    __tablename__ = "qualifications"
    id = fixture("pk")
    label = fixture("text")


class Location(Base):
    """
    Based on https://tinyurl.com/4nushphc, processed
    via flows/pre_enrich/locations_lookup.py
    """

    __tablename__ = "locations"
    ipn_18_code = fixture("ipn_pk")
    ipn_18_name = fixture("text")
    country_18_name = Column(VARCHAR(8), index=True)
    lad_18_code = fixture("gss_code")
    lad_18_name = fixture("text")
    health_12_code = fixture("gss_code")
    health_12_name = fixture("text")
    region_18_code = fixture("gss_code")
    region_18_name = fixture("text")
    nuts_0_code = fixture("nuts_code")
    nuts_0_name = fixture("text")
    nuts_1_code = fixture("nuts_code")
    nuts_1_name = fixture("text")
    nuts_2_code = fixture("nuts_code")
    nuts_2_name = fixture("text")
    nuts_3_code = fixture("nuts_code")
    nuts_3_name = fixture("text")
