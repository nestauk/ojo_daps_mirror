"""
Example ORM
===========

An example of an ORM, which integrates with the example batch pipeline,
to collect data from SWAPI.
"""

from ojd_daps import declarative_base
from ojd_daps.orms.common import fixture

Base = declarative_base()


class People(Base):
    """Slimmed down People entity from SWAPI"""

    id = fixture("pk")
    name = fixture("text")
    height = fixture("integer")
    mass = fixture("integer")
    hair_color = fixture("text")
    eye_color = fixture("text")
    birth_year = fixture("integer")
    gender = fixture("text")
    homeworld = fixture("text")
    created = fixture("datetime")
    url = fixture("text")
