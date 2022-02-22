"""Common utils for indicator production."""
from datetime import datetime, timedelta
from ojd_daps.dqa.data_getters import (
    get_snapshot_ads,
    DEFAULT_START_DATE,
    DEFAULT_END_DATE,
    iterdates,
)
from ojd_daps import __version__
from itertools import groupby, product
from metaflow import S3
from collections import Counter
from cardinality import count
import numpy as np
import json
from csv import DictWriter, QUOTE_NONNUMERIC
from io import StringIO
from copy import deepcopy
import requests
import yaml
from pathlib import Path
import sys


# Stock calculation: first four weeks of April 2021
STOCK_WEEKS = 4
STOCK_IDX_START = datetime(2021, 4, 5)  # first Monday, April 2021
STOCK_IDX_END = STOCK_IDX_START + timedelta(weeks=STOCK_WEEKS) - timedelta(days=1)

# NUTS2 codes
LONDON = ["UKI3", "UKI4", "UKI5", "UKI6", "UKI7"]
DEFAULT_LOCATION_CODE = "ZZZ1"
DEFAULT_LOCATION_NAME = "Unmatched"


# Plotting globals
QUANTILE_MAPPING = {
    "Lower quartile": lambda data: np.percentile(data, 25),
    "Median": lambda data: np.median(data),
    "Upper quartile": lambda data: np.percentile(data, 75),
}
VOLUME_LABEL = (
    f"Volume of adverts (index: 100 = {STOCK_IDX_START.strftime('%B %Y')} average)"
)

# For saving output data to
BUCKET = "open-jobs-indicators"

# --- Data dictionary template fields here --- #


class DataDict:
    def __init__(self):
        # Check type is a valid schema.org object
        response = requests.head(self.type)
        response.raise_for_status()
        # Check name and description are valid
        if type(self.name) is not str:
            raise ValueError("Name must be a string")
        if type(self.description) is not str:
            raise ValueError("Description must be a string")


class VOLUME(DataDict):
    name = VOLUME_LABEL
    type = "https://schema.org/totalJobOpenings"
    description = (
        "Total count of job advertisements, indexed as 100 = "
        f"{STOCK_IDX_START.strftime('%B %Y')} average"
    )


class DATE(DataDict):
    name = "Date"
    type = "https://schema.org/Date"
    description = "Weekly datestamp for this observation. Each datestamp is a Monday."


class LOCATION_NAME(DataDict):
    name = "SOC Name"
    type = "https://schema.org/name"
    description = "The name of this NUTS 2 region"


class LOCATION_CODE(DataDict):
    name = "SOC Code"
    type = "https://schema.org/identifier"
    description = "The code of this NUTS 2 region"


class MIN_LOWER_Q(DataDict):
    name = "Lower quartile MIN salaries (£000 pa)"
    type = "https://schema.org/percentile25"
    description = "Lower quartile of minimum annualised advertised job salaries"


class MIN_MEDIAN(DataDict):
    name = "Median MIN salaries (£000 pa)"
    type = "https://schema.org/median"
    description = "Median of minimum annualised advertised job salaries"


class MIN_UPPER_Q(DataDict):
    name = "Upper quartile MIN salaries (£000 pa)"
    type = "https://schema.org/percentile75"
    description = "Upper quartile of minimum annualised advertised job salaries"


class MAX_LOWER_Q(DataDict):
    name = "Lower quartile MAX salaries (£000 pa)"
    type = "https://schema.org/percentile25"
    description = "Lower quartile of maximum annualised advertised job salaries"


class MAX_MEDIAN(DataDict):
    name = "Median MAX salaries (£000 pa)"
    type = "https://schema.org/median"
    description = "Median of maximum annualised advertised job salaries"


class MAX_UPPER_Q(DataDict):
    name = "Upper quartile MAX salaries (£000 pa)"
    type = "https://schema.org/percentile75"
    description = "Upper quartile of maximum annualised advertised job salaries"


class SKILL_GROUP(DataDict):
    name = "Skill Group"
    type = "https://schema.org/name"
    description = "Name of this skill cluster"


class PERCENTAGE(DataDict):
    name = "Percentage"
    type = "https://schema.org/Number"
    description = "Percentage of job adverts with this skill that have this code."


def _extract_features(ad, *feature_names):
    """
    Flatten out and extract specified features from the "features" column.
    Where there are multiple features (e.g. multiple skills) which are in
    'list' form, every combination of 'list' features is returned, i.e.:

    input data:
        {
        "features": {
            "FOO": [{"foo": "a"}, {"foo": "b"}],
            "BAR": [{"bar": 1}, {"bar": 2}],
            "BAZ": {"baz": "boom"},
            }
        }

    output data:
        [
            {"foo": "a", "bar": 1, "baz": "boom"},
            {"foo": "a", "bar": 2, "baz": "boom"},
            {"foo": "b", "bar": 1, "baz": "boom"},
            {"foo": "b", "bar": 2, "baz": "boom"},
        ]

    NB: This function alters the input data.
    """
    raw_features = ad.pop("features")

    # Firstly convert all features to list
    feature_collection = []
    for name in feature_names:
        feat = raw_features.pop(name, {})
        feat = feat.get(name, feat)
        if type(feat) is not list:
            feat = [feat]
        feature_collection.append(feat)

    # Then yield one row for every combination of features
    for _features in product(*feature_collection):
        features = {}
        for feat in _features:
            features.update(feat)
        yield features


def extract_features(job_ads, *feature_names):
    """
    Apply _extract_features to every job advert, in effect flattening out
    job_ads for the specified feature_names
    """
    flat_ads = []
    while job_ads:
        ad = job_ads.pop(0)  # Save memory with while / pop
        for features in _extract_features(ad, *feature_names):
            _ad = deepcopy(ad)
            _ad.update(features)
            flat_ads.append(_ad)
    return flat_ads


def sort_and_groupby(data, *columns):
    """
    Group a list of dicts by the specified keys
    ('columns' to distinguish from the 'key' argument of sorted and groupby)
    """
    sorted_data = sorted(data, key=lambda item: tuple(item[col] for col in columns))
    return groupby(sorted_data, key=lambda item: tuple(item[col] for col in columns))


def get_index_stock_lookup(feature_name, code):
    """
    Create one index value per code value, with the index value
    normalised to the number of stock weeks.

    Args:
        feature_name (str): e.g.
        code (str): e.g.
    Returns:
        count per week of each feature, by code value
    """
    job_ads = get_weekly_ads(start_date=STOCK_IDX_START, end_date=STOCK_IDX_END)
    return _get_index_stock_lookup(job_ads, feature_name=feature_name, code=code)


def _get_index_stock_lookup(job_ads, feature_name, code):
    """
    Create one index value per code value, with the index value
    normalised to the number of stock weeks
    """
    job_ads = extract_features(deepcopy(job_ads), feature_name)
    job_ads = filter(None, (ad.get(code) for ad in job_ads))
    return {
        _code: count(chunk) / STOCK_WEEKS
        for (_code,), chunk in sort_and_groupby(job_ads, code)
    }


def iterquantiles(job_ads):
    """Yield labels and quantiles of salary data, in units of £1k"""
    for bound in ("min", "max"):
        salaries = list(
            filter(None, (ad.get(f"{bound}_annualised_salary") for ad in job_ads))
        )
        for quantile_name, quantile_func in QUANTILE_MAPPING.items():
            try:
                quantile = quantile_func(salaries)
            except IndexError:
                quantile = 0
            # e.g "Median MIN salaries (£000 pa)"
            label = f"{quantile_name} {bound.upper()} salaries (£000 pa)"
            yield label, quantile / 1000


def get_weekly_ads(start_date=DEFAULT_START_DATE, end_date=DEFAULT_END_DATE):
    """Get weekly snapshots for the range of dates specified by `iterdates`"""
    job_ads = []
    # Read data in weekly chunks
    # NB: weekly chunks overlap, however they need to be deduped in their chunk
    # and so it isn't super simple to do this without double collecting ads.
    # There are approaches that would work (and therefore speed up collection by
    # nearly 2x) but we haven't got around to it
    for from_date, to_date in iterdates(start_date=start_date, end_date=end_date):
        # Read the ads in this week chunk and assign week_date to the
        # stock reference date
        for ad in get_snapshot_ads(from_date=from_date, to_date=to_date):
            ad["week_date"] = to_date
            job_ads.append(ad)
    return job_ads


def standardise_location(job_ad):
    """Consolidate London regions and set a default location for unmatched job ads"""
    nuts_code = job_ad.get("nuts_2_code")  # NB: returns None if doesn't exist
    # Assign edge case values
    if nuts_code in LONDON:
        job_ad["nuts_2_code"] = "UKI"
        job_ad["nuts_2_name"] = "London"
    elif nuts_code is None:  # i.e. not assigned a nuts code
        job_ad["nuts_2_code"] = DEFAULT_LOCATION_CODE
        job_ad["nuts_2_name"] = DEFAULT_LOCATION_NAME
    return job_ad


def most_common(items, n):
    """Get top-n most common items"""
    return {item for item, _ in Counter(items).most_common(n)}


def volume_calc(chunk, denominator):
    """Percentage or volume calculation"""
    return 100.0 * count(chunk) / denominator


def aggregate_skills(job_ads, code, name, title, cluster="label_cluster_0"):
    """Group job adverts by a given code-cluster combination, and calculate the
    percentage of of the job adverts with that cluster for the given code.

    `code` refers to a SOC or location code, and `cluster`
    refers to a skills cluster label.
    """
    # Filter out jobs not matched to a skills cluster
    job_ads = list(filter(lambda ad: ad.get(cluster) is not None, job_ads))
    count_by_code = {
        _code: count(chunk) for _code, chunk in sort_and_groupby(job_ads, code)
    }
    aggregation = [
        {
            f"{title} name": _name,
            f"{title} code": _code,
            "Skill Group": _cluster,
            "Percentage": volume_calc(chunk, count_by_code[(_code,)]),
        }
        for (_code, _name, _cluster), chunk in sort_and_groupby(
            job_ads, code, name, cluster
        )
    ]
    return aggregation


def json_dumps(data, decimal_places=2, **dumps_kwargs):
    """
    json.dumps with float formatting.

    Taken from https://stackoverflow.com/a/29066406/1571593
    """
    str_data = json.dumps(data, **dumps_kwargs)
    rounded_data = json.loads(
        str_data, parse_float=lambda x: round(float(x), decimal_places)
    )
    return json.dumps(rounded_data, **dumps_kwargs)


def json_to_csv(data, dialect="excel"):
    """Convert row-wise JSON to CSV"""
    fieldnames = list(data[0].keys())
    with StringIO() as sio:
        writer = DictWriter(
            sio, fieldnames=fieldnames, dialect=dialect, quoting=QUOTE_NONNUMERIC
        )
        writer.writeheader()
        writer.writerows(data)
        sio.seek(0)
        return sio.read()


def get_template(template_name):
    template_mod = getattr(sys.modules[__name__], template_name)
    return {
        k: v
        for k, v in template_mod.__dict__.items()
        if k in ["description", "name", "type"]
    }


def is_template(field):
    return next(iter(field)) == "template"


def generate_data_dict(title):
    path = Path(__file__).resolve().parent
    with open(path / "data_dicts" / f"{title}.yaml") as f:
        data_dict = yaml.safe_load(f)
    fields = (
        get_template(field["template"]) if is_template(field) else field
        for field in data_dict["fields"]
    )
    text = [
        f"{title}",
        f"{'-'*len(title)}",
        "",
        data_dict["description"],
        "",
        "Fields:",
        "-------",
        "",
    ]
    text += [
        f"- {field['name']} ({field['type']}): \"{field['description']}\"\n"
        for field in fields
    ]
    return "\n".join(text)


def save_data(flow, title):
    """Save indicator data to s3"""
    # Write to the output of this flow, for curation purposes
    data = json_dumps(flow.data)
    with S3(run=flow) as s3:
        s3.put(f"{title}_test-{flow.test}.json", data)

    # Write to 'latest' and by 'version'
    prefix = f"s3://{BUCKET}/{flow.db_name}"
    for version in ("latest", __version__):
        with S3(s3root=f"{prefix}/{version}/") as s3:
            s3.put(f"{title}.json", data)
            s3.put(f"{title}.csv", json_to_csv(json.loads(data)))
            s3.put(f"{title}_data_dict.txt", generate_data_dict(title))
