import pytest
from sqlalchemy_utils import get_declarative_base
from dateutil import parser as date_parser
from daps_utils.db import db_session, insert_data
from ojd_daps.orms.example_orm import People


@pytest.mark.timeout(10)
def test_people():
    Base = get_declarative_base(People)
    data = [{'url': 'https://swapi.dev/api/people/12345/',
             'name': 'Joel', 'height': 195,
             'id': '12345', 'mass': 2312, 'gender': 'yellow',
             'hair_color': 'brown', 'eye_color': 'brown',
             'birth_year': 1962, 'homeworld': '23',
             'created': date_parser.parse("Aug 28 1999")}]
    # Insert the data in one transaction
    with db_session() as session:
        engine = session.get_bind()
        Base.metadata.create_all(engine)
        insert_data(data=data, model=People, session=session)
    # Tidy up in another transaction
    with db_session() as session:
        engine = session.get_bind()
        Base.metadata.drop_all(engine)
