from sqlalchemy import Column
from sqlalchemy.types import DATETIME, DECIMAL, INTEGER, TEXT, VARCHAR, BIGINT, BOOLEAN

"""Standard set of features to avoid duplication or inconsistencies"""
FIXTURES = {
    "pk": lambda: Column(INTEGER, primary_key=True, autoincrement=False),
    "big_pk": lambda: Column(BIGINT, primary_key=True, autoincrement=False),
    "text": lambda: Column(TEXT, nullable=True, default=None),
    "datetime": lambda: Column(DATETIME, nullable=False, index=True),
    "integer": lambda: Column(INTEGER, nullable=False, index=True),
    "float": lambda: Column(DECIMAL(6, 3), nullable=False, index=True),
    "lookup_id": lambda: Column(INTEGER, nullable=True, index=True),
    "gss_code": lambda: Column(VARCHAR(9), nullable=True, index=True, default=None),
    "nuts_code": lambda: Column(VARCHAR(6), nullable=True, index=True, default=None),
    "ipn_pk": lambda: Column(VARCHAR(12), primary_key=True),
    "salary": lambda: Column(DECIMAL(8, 2), nullable=True, index=True),
    "boolean": lambda: Column(BOOLEAN, nullable=True, index=True),
}


def fixture(key):
    """Lookup to the fixture list"""
    return FIXTURES[key]()
