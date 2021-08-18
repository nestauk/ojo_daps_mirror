import pytest
from sqlalchemy_utils import get_declarative_base
from dateutil import parser as date_parser
from daps_utils.db import db_session, insert_data
from ojd_daps.orms.indicators import Jobs_by_Location

@pytest.mark.timeout(10)
def test_job_ad_duplicate_link():
    Base = get_declarative_base(Jobs_by_Location)
    data = [{'location_id':-1,
    'count':2}]
    # Insert the data in one transaction
    with db_session() as session:
        engine = session.get_bind()
        Base.metadata.create_all(engine)
        insert_data(data=data, model=Jobs_by_Location, session=session)
    # Tidy up in another transaction
    with db_session() as session:
        engine = session.get_bind()
        Base.metadata.drop_all(engine)