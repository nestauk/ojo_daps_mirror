import pytest
from sqlalchemy_utils import get_declarative_base
from dateutil import parser as date_parser
from daps_utils.db import db_session, insert_data
from ojd_daps.orms.raw_jobs import RawJobAd, ReedAdDetail, JobAdDescriptionVector


@pytest.mark.timeout(10)
def test_raw_job_ad():
    Base = get_declarative_base(RawJobAd)
    data = [
        {
            "id": "762256",
            "data_source": "Indeed",
            "created": date_parser.parse("Oct 20 2020"),
            "s3_location": "s3://example_lake/example_object/file.txt",
            "url": "https://www.indeed.co.uk/viewjob?wf7yhfiao",
            "job_title_raw": "Administration Assistant",
            "job_location_raw": "London",
            "raw_salary": "20500.23",
            "raw_min_salary": "20500.23",
            "raw_max_salary": "21500.23",
            "raw_salary_band": "20500.23-21500.23",
            "raw_salary_unit": "YEAR",
            "raw_salary_currency": "GBP",
            "salary_competitive": True,
            "salary_negotiable": False,
            "company_raw": "BG&GS LLP",
            "contract_type_raw": "Permanent",
            "closing_date_raw": date_parser.parse("Nov 20 2020"),
            "description": "Great Opportunity.",
        }
    ]
    # Insert the data in one transaction
    with db_session() as session:
        engine = session.get_bind()
        Base.metadata.create_all(engine)
        insert_data(data=data, model=RawJobAd, session=session)
    # Tidy up in another transaction
    with db_session() as session:
        engine = session.get_bind()
        Base.metadata.drop_all(engine)


@pytest.mark.timeout(10)
def test_reed_ad_detail():
    Base = get_declarative_base(ReedAdDetail)
    data = [
        {
            "id": "762256",
            "type": "Entry",
            "sector": "Law",
            "parent_sector": "Legal",
            "knowledge_domain": "Administration",
            "occupation": "Administrator",
        }
    ]
    # Insert the data in one transaction
    with db_session() as session:
        engine = session.get_bind()
        Base.metadata.create_all(engine)
        insert_data(data=data, model=ReedAdDetail, session=session)
    # Tidy up in another transaction
    with db_session() as session:
        engine = session.get_bind()
        Base.metadata.drop_all(engine)


@pytest.mark.timeout(10)
def test_job_ad_description_vector():
    Base = get_declarative_base(JobAdDescriptionVector)
    data = [{"id": "762256", "vector": '["5", "6", "7", "8"]'}]
    # Insert the data in one transaction
    with db_session() as session:
        engine = session.get_bind()
        Base.metadata.create_all(engine)
        insert_data(data=data, model=JobAdDescriptionVector, session=session)
    # Tidy up in another transaction
    with db_session() as session:
        engine = session.get_bind()
        Base.metadata.drop_all(engine)
