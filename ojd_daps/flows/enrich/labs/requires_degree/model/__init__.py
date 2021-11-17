"""Main API for requires_degree regex model."""
from functools import lru_cache
from ojd_daps.flows.common import save_to_s3, load_from_s3
from . import nlp

S3_PATH = "labs/requires_degree/regex"
DEGREES = [
    "ba",
    "bsc",
    "msc",
    "masters",
    "phd",
    "bachelor's",
    "master's",
]
# NB currently only matches whole words with spaces which may be too strict
EXPRESSION = r"(?=(\b" + "\\b|\\b".join(DEGREES) + r"\b))"


@lru_cache()  # <--- Important
def load_model():
    """Loads the model from S3. LRU cache to reduce overhead on repeat calls."""
    regex = load_from_s3(S3_PATH, "regex.txt")
    return nlp.regex_model(regex)


def save_model(regex=EXPRESSION):
    """Save the regex to S3."""
    save_to_s3(S3_PATH, "regex.txt", regex)


def apply_model(row):
    """Loads and applies the model to the given row"""
    model = load_model()  # NB: lru_cached
    description = nlp.clean_description(row["description"])
    return model(description)
