import pytest
from sqlalchemy_utils import get_declarative_base
from dateutil import parser as date_parser
from daps_utils.db import db_session, insert_data
from ojd_daps.orms.std_features import (
    SOC,
    SIC,
    ESCO,
    Skill,
    Salary,
    StdJobTitle,
    TTWA,
    NUTS3,
    Experience,
    Qualification,
    Location,
)


@pytest.mark.timeout(10)
def test_soc():
    Base = get_declarative_base(SOC)
    data = [
        {
            "soc_id": 5542321423124221,  # BIGINT, and using 16 digits
            "soc_code": "12312",
            "soc_title": "Corporate managers and directors",
        }
    ]
    assert len(str(data[0]["soc_id"])) == 16  # i.e. don't tamper with the test!
    # Insert the data in one transaction
    with db_session() as session:
        engine = session.get_bind()
        Base.metadata.create_all(engine)
        insert_data(data=data, model=SOC, session=session)
    # Tidy up in another transaction
    with db_session() as session:
        engine = session.get_bind()
        Base.metadata.drop_all(engine)


@pytest.mark.timeout(10)
def test_sic():
    Base = get_declarative_base(SIC)
    data = [{"id": "2", "label": "Mining and Quarrying"}]
    # Insert the data in one transaction
    with db_session() as session:
        engine = session.get_bind()
        Base.metadata.create_all(engine)
        insert_data(data=data, model=SIC, session=session)
    # Tidy up in another transaction
    with db_session() as session:
        engine = session.get_bind()
        Base.metadata.drop_all(engine)


@pytest.mark.timeout(10)
def test_skills():
    Base = get_declarative_base(Skill)
    data = [{"id": "678", "label": "Software Development"}]
    # Insert the data in one transaction
    with db_session() as session:
        engine = session.get_bind()
        Base.metadata.create_all(engine)
        insert_data(data=data, model=Skill, session=session)
    # Tidy up in another transaction
    with db_session() as session:
        engine = session.get_bind()
        Base.metadata.drop_all(engine)


@pytest.mark.timeout(10)
def test_salary():
    Base = get_declarative_base(Salary)
    data = [{"id": "567",
    "min_salary": 27000.00,
    "max_salary": 28000.00,
    "rate": "per annum",
    "min_annualised_salary": 27000.00,
    "max_annualised_salary": 28000.00,}]
    # Insert the data in one transaction
    with db_session() as session:
        engine = session.get_bind()
        Base.metadata.create_all(engine)
        insert_data(data=data, model=Salary, session=session)
    # Tidy up in another transaction
    with db_session() as session:
        engine = session.get_bind()
        Base.metadata.drop_all(engine)


@pytest.mark.timeout(10)
def test_std_job_title():
    Base = get_declarative_base(StdJobTitle)
    data = [{"id": "8758", "label": "Accounts Executive"}]
    # Insert the data in one transaction
    with db_session() as session:
        engine = session.get_bind()
        Base.metadata.create_all(engine)
        insert_data(data=data, model=StdJobTitle, session=session)
    # Tidy up in another transaction
    with db_session() as session:
        engine = session.get_bind()
        Base.metadata.drop_all(engine)


@pytest.mark.timeout(10)
def test_ttwa():
    Base = get_declarative_base(TTWA)
    data = [{"id": "829", "label": "Newcastle"}]
    # Insert the data in one transaction
    with db_session() as session:
        engine = session.get_bind()
        Base.metadata.create_all(engine)
        insert_data(data=data, model=TTWA, session=session)
    # Tidy up in another transaction
    with db_session() as session:
        engine = session.get_bind()
        Base.metadata.drop_all(engine)


@pytest.mark.timeout(10)
def test_nuts3():
    Base = get_declarative_base(NUTS3)
    data = [{"id": "59309", "label": "Shropshire"}]
    # Insert the data in one transaction
    with db_session() as session:
        engine = session.get_bind()
        Base.metadata.create_all(engine)
        insert_data(data=data, model=NUTS3, session=session)
    # Tidy up in another transaction
    with db_session() as session:
        engine = session.get_bind()
        Base.metadata.drop_all(engine)


@pytest.mark.timeout(10)
def test_experience():
    Base = get_declarative_base(Experience)
    data = [{"id": "27829", "label": "1 year"}]
    # Insert the data in one transaction
    with db_session() as session:
        engine = session.get_bind()
        Base.metadata.create_all(engine)
        insert_data(data=data, model=Experience, session=session)
    # Tidy up in another transaction
    with db_session() as session:
        engine = session.get_bind()
        Base.metadata.drop_all(engine)


@pytest.mark.timeout(10)
def test_qualifications():
    Base = get_declarative_base(Qualification)
    data = [{"id": "1", "label": "Level 1"}]
    # Insert the data in one transaction
    with db_session() as session:
        engine = session.get_bind()
        Base.metadata.create_all(engine)
        insert_data(data=data, model=Qualification, session=session)
    # Tidy up in another transaction
    with db_session() as session:
        engine = session.get_bind()
        Base.metadata.drop_all(engine)


@pytest.mark.timeout(10)
def test_esco():
    Base = get_declarative_base(ESCO)
    data = [{"id": "92936", "label": "Text Processing"}]
    # Insert the data in one transaction
    with db_session() as session:
        engine = session.get_bind()
        Base.metadata.create_all(engine)
        insert_data(data=data, model=ESCO, session=session)
    # Tidy up in another transaction
    with db_session() as session:
        engine = session.get_bind()
        Base.metadata.drop_all(engine)


@pytest.mark.timeout(10)
def test_location():
    Base = get_declarative_base(Location)
    data = [
        {
            "ipn_18_code": "",
            "ipn_18_name": "",
            "country_18_name": "",
            "lad_18_code": "",
            "lad_18_name": "",
            "health_12_code": "",
            "health_12_name": "",
            "region_18_code": "",
            "region_18_name": "",
            "nuts_0_code": "",
            "nuts_0_name": "",
            "nuts_1_code": "",
            "nuts_1_name": "",
            "nuts_2_code": "",
            "nuts_2_name": "",
            "nuts_3_code": "",
            "nuts_3_name": "",
        }
    ]

    # Insert the data in one transaction
    with db_session() as session:
        engine = session.get_bind()
        Base.metadata.create_all(engine)
        insert_data(data=data, model=Location, session=session)
    # Tidy up in another transaction
    with db_session() as session:
        engine = session.get_bind()
        Base.metadata.drop_all(engine)
