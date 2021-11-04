import pytest
from sqlalchemy_utils import get_declarative_base
from dateutil import parser as date_parser
from daps_utils.db import db_session, insert_data
from ojd_daps.orms.link_tables import (
    JobAdDuplicateLink,
    JobAdJobTitleLink,
    JobAdSOCLink,
    JobAdSICLink,
    JobAdESCOLink,
    JobAdSkillLink,
    JobAdTTWALink,
    JobAdNUTS3Link,
    JobAdStdSalaryLink,
    JobAdExperienceLink,
    JobAdQualificationsLink,
    JobAdReedAdDetailLink,
    JobAdLocationLink,
)


@pytest.mark.timeout(10)
def test_job_ad_duplicate_link():
    Base = get_declarative_base(JobAdDuplicateLink)
    data = [{"first_id": "837456", "second_id": "3684", "weight": "0.99"}]
    # Insert the data in one transaction
    with db_session() as session:
        engine = session.get_bind()
        Base.metadata.create_all(engine)
        insert_data(data=data, model=JobAdDuplicateLink, session=session)
    # Tidy up in another transaction
    with db_session() as session:
        engine = session.get_bind()
        Base.metadata.drop_all(engine)


@pytest.mark.timeout(10)
def test_job_ad_location_link():
    Base = get_declarative_base(JobAdLocationLink)
    data = [{"job_id": "837456", "job_data_source": "Indeed", "location_id": "3684"}]
    # Insert the data in one transaction
    with db_session() as session:
        engine = session.get_bind()
        Base.metadata.create_all(engine)
        insert_data(data=data, model=JobAdLocationLink, session=session)
    # Tidy up in another transaction
    with db_session() as session:
        engine = session.get_bind()
        Base.metadata.drop_all(engine)


@pytest.mark.timeout(10)
def test_job_ad_job_title_link():
    Base = get_declarative_base(JobAdJobTitleLink)
    data = [
        {"job_id": "837456", "job_data_source": "3684", "std_job_title_id": "93409"}
    ]
    # Insert the data in one transaction
    with db_session() as session:
        engine = session.get_bind()
        Base.metadata.create_all(engine)
        insert_data(data=data, model=JobAdJobTitleLink, session=session)
    # Tidy up in another transaction
    with db_session() as session:
        engine = session.get_bind()
        Base.metadata.drop_all(engine)


@pytest.mark.timeout(10)
def test_job_ad_soc_link():
    Base = get_declarative_base(JobAdSOCLink)
    data = [{"job_id": "837456", "job_data_source": "3684", "soc_id": "93409"}]
    # Insert the data in one transaction
    with db_session() as session:
        engine = session.get_bind()
        Base.metadata.create_all(engine)
        insert_data(data=data, model=JobAdSOCLink, session=session)
    # Tidy up in another transaction
    with db_session() as session:
        engine = session.get_bind()
        Base.metadata.drop_all(engine)


@pytest.mark.timeout(10)
def test_job_ad_sic_link():
    Base = get_declarative_base(JobAdSICLink)
    data = [{"job_id": "837456", "job_data_source": "3684", "sic_id": "93409"}]
    # Insert the data in one transaction
    with db_session() as session:
        engine = session.get_bind()
        Base.metadata.create_all(engine)
        insert_data(data=data, model=JobAdSICLink, session=session)
    # Tidy up in another transaction
    with db_session() as session:
        engine = session.get_bind()
        Base.metadata.drop_all(engine)


@pytest.mark.timeout(10)
def test_job_ad_esco_link():
    Base = get_declarative_base(JobAdESCOLink)
    data = [{"job_id": "837456", "job_data_source": "3684", "esco_id": "93409"}]
    # Insert the data in one transaction
    with db_session() as session:
        engine = session.get_bind()
        Base.metadata.create_all(engine)
        insert_data(data=data, model=JobAdESCOLink, session=session)
    # Tidy up in another transaction
    with db_session() as session:
        engine = session.get_bind()
        Base.metadata.drop_all(engine)


@pytest.mark.timeout(10)
def test_job_ad_skill_link():
    Base = get_declarative_base(JobAdSkillLink)
    data = [
        {
            "job_id": "837456",
            "job_data_source": "Reed",
            "surface_form": "tax",
            "surface_form_type": "manual",
            "preferred_label": "calculate_tax",
            "entity": 2521,
            "predicted_q": "0.818104",
            "cluster_0": 6.0,
            "cluster_1": 1.0,
            "cluster_2": 7.0,
            "label_cluster_0": "Transversal skills",
            "label_cluster_1": "General Workplace Skills",
            "label_cluster_2": "General Workplace Skills",
        }
    ]
    # Insert the data in one transaction
    with db_session() as session:
        engine = session.get_bind()
        Base.metadata.drop_all(engine)


@pytest.mark.timeout(10)
def test_job_ad_TTWA_link():
    Base = get_declarative_base(JobAdTTWALink)
    data = [{"job_id": "837456", "job_data_source": "3684", "ttwa_id": "93409"}]
    # Insert the data in one transaction
    with db_session() as session:
        engine = session.get_bind()
        Base.metadata.create_all(engine)
        insert_data(data=data, model=JobAdTTWALink, session=session)
    # Tidy up in another transaction
    with db_session() as session:
        engine = session.get_bind()
        Base.metadata.drop_all(engine)


@pytest.mark.timeout(10)
def test_job_ad_nuts3_link():
    Base = get_declarative_base(JobAdNUTS3Link)
    data = [{"job_id": "837456", "job_data_source": "3684", "nuts3_id": "93409"}]
    # Insert the data in one transaction
    with db_session() as session:
        engine = session.get_bind()
        Base.metadata.create_all(engine)
        insert_data(data=data, model=JobAdNUTS3Link, session=session)
    # Tidy up in another transaction
    with db_session() as session:
        engine = session.get_bind()
        Base.metadata.drop_all(engine)


@pytest.mark.timeout(10)
def test_job_ad_std_salary_link():
    Base = get_declarative_base(JobAdStdSalaryLink)
    data = [{"job_id": "837456", "job_data_source": "3684", "std_salary_id": "93409"}]
    # Insert the data in one transaction
    with db_session() as session:
        engine = session.get_bind()
        Base.metadata.create_all(engine)
        insert_data(data=data, model=JobAdStdSalaryLink, session=session)
    # Tidy up in another transaction
    with db_session() as session:
        engine = session.get_bind()
        Base.metadata.drop_all(engine)


@pytest.mark.timeout(10)
def test_job_ad_experience_link():
    Base = get_declarative_base(JobAdExperienceLink)
    data = [{"job_id": "837456", "job_data_source": "3684", "experience_id": "93409"}]
    # Insert the data in one transaction
    with db_session() as session:
        engine = session.get_bind()
        Base.metadata.create_all(engine)
        insert_data(data=data, model=JobAdExperienceLink, session=session)
    # Tidy up in another transaction
    with db_session() as session:
        engine = session.get_bind()
        Base.metadata.drop_all(engine)


@pytest.mark.timeout(10)
def test_job_ad_qualifications_link():
    Base = get_declarative_base(JobAdQualificationsLink)
    data = [
        {"job_id": "837456", "job_data_source": "3684", "qualification_id": "93409"}
    ]
    # Insert the data in one transaction
    with db_session() as session:
        engine = session.get_bind()
        Base.metadata.create_all(engine)
        insert_data(data=data, model=JobAdQualificationsLink, session=session)
    # Tidy up in another transaction
    with db_session() as session:
        engine = session.get_bind()
        Base.metadata.drop_all(engine)


@pytest.mark.timeout(10)
def test_job_ad_reed_ad_detail_link():
    Base = get_declarative_base(JobAdReedAdDetailLink)
    data = [
        {"job_id": "837456", "job_data_source": "3684", "reed_ad_detail_id": "93409"}
    ]
    # Insert the data in one transaction
    with db_session() as session:
        engine = session.get_bind()
        Base.metadata.create_all(engine)
        insert_data(data=data, model=JobAdReedAdDetailLink, session=session)
    # Tidy up in another transaction
    with db_session() as session:
        engine = session.get_bind()
        Base.metadata.drop_all(engine)
