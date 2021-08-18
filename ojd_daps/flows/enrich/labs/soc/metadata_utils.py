"""
metadata_utils
--------------

Tools for populating S3 with metadata required for SOC matching.
The `save_metadata_to_s3` is effectively the "main" function here.
"""
import nltk
import requests
from bs4 import BeautifulSoup
from functools import lru_cache
from collections import defaultdict
from openpyxl import load_workbook
from io import BytesIO, StringIO
from itertools import islice, chain
import csv

from ojd_daps.flows.enrich.labs.soc.common import (
    save_json_to_s3,
    load_json_from_s3,
    load_metadata,
)
from ojd_daps.flows.enrich.labs.soc.common import replace_punctuation

ONS_BASE = "https://www.ons.gov.uk{}"
SOC_URL = ONS_BASE.format(
    "/methodology/classificationsandstandards/"
    "standardoccupationalclassificationsoc/soc2020/"
    "soc2020volume2codingrulesandconventions"
)
SOC_LINK_ELEMENT = {"data-gtm-title": "SOC2020 the coding index (excel)"}
SHEETNAME_CODING = "SOC2020 coding index V4"
SHEETNAME_GROUPS = "SOC2020 structure"
WORLD_CITIES = (
    "https://pkgstore.datahub.io/core/world-cities/"
    "world-cities_csv/data/6cc66692f0e82b18216a48443b6b95da/"
    "world-cities_csv.csv"
)


def get_file_link():
    """Finds string directing to dataset"""
    r = requests.get(SOC_URL)
    soup = BeautifulSoup(r.content, "html.parser")
    for tag in soup.find_all("a", SOC_LINK_ELEMENT, href=True):
        link = tag["href"]
        if not link.endswith("xlsx"):
            continue
        return ONS_BASE.format(link)
    raise ValueError(f"Could not find xlsx file on {SOC_URL}")


@lru_cache()
def WordNetLemmatizer():
    nltk.download("wordnet")
    return nltk.WordNetLemmatizer()


def lemmatise(term):
    """Apply the NLTK WN Lemmatizer to the term"""
    lem = WordNetLemmatizer()
    return lem.lemmatize(term)


def standardise_title(job):
    """Create a standardised verbose title for job"""

    not_null = filter(None, (job["IND"], job["ADD"]))
    not_nos = filter(lambda v: v != "nos", not_null)  # NOS = Not Specified

    # Create the title by combining and standardising the various fields
    verbose_title = list(reversed(job["INDEXOCC"].split(", ")))
    verbose_title += list(not_nos)
    verbose_title = " ".join(verbose_title)
    verbose_title = replace_punctuation(verbose_title).strip().lower()

    # Edge case alert: mfr means 'manufacturing making building repairing'
    manufacturing_terms = " ".join(load_metadata("manufacturing_terms"))
    verbose_title = verbose_title.replace("mfr", manufacturing_terms)
    return verbose_title


@lru_cache()
def load_plurals():
    """Load the local occupation plural metadataset"""
    return set(load_metadata("occupation_plurals"))


def separate_plural_edgecases(terms):
    """Seperate plurals from edgecases efficiently"""
    plurals = load_plurals()
    terms = set(terms)
    edgecases = terms.intersection(plurals)
    non_edgecases = terms - edgecases
    return edgecases, non_edgecases


def standardise_group_title(group):
    """Standardise SOC group titles"""
    title = group["SOC2020 Group Title"].lower().strip().split()
    edgecases, non_edgecases = separate_plural_edgecases(title)
    return " ".join(chain(map(lemmatise, non_edgecases), edgecases))


@lru_cache()
def requests_get(url):
    """Perform a GET request to the URL and return the encoded content"""
    response = requests.get(url)
    response.raise_for_status()
    return response.content


def read_excel_from_url(url, sheetname, head_cleaning=lambda x: x.replace("\n", " ")):
    """Yield row-by-row content a sheet from a remote Excel file"""
    content = requests_get(url)  # Cached
    with BytesIO(content) as f:
        workbook = load_workbook(f, read_only=True)
        sheet = workbook[sheetname]
        # Header is the first row, thereafter rows are values
        raw_vals = sheet.values  # An iterator
        header = list(map(head_cleaning, next(raw_vals)[1:]))  # Clean the header
        rows_of_values = (islice(row, 1, None) for row in raw_vals)  # Skip the idx col
        # Map the header on to each value
        combined_rows = map(lambda row: dict(zip(header, row)), rows_of_values)
        for row in filter(bool, combined_rows):  # Filter empty rows
            yield row


def read_world_cities(header=["town", "ctry", "", ""]):
    """Read towns and cities from the remote world cities CSV file"""
    content = requests_get(WORLD_CITIES)
    with StringIO(content.decode("utf-8")) as f:
        for row in csv.DictReader(f, fieldnames=header):
            yield row


def load_ons_data(sheetname, title_function, title_field):
    """
    Load the ONS SOC Excel file, select a sheet and extract
    the job title and SOC from each row, and also cleaning the title fieldnames
    according to a specified function.
    """
    url = get_file_link()
    coding_data = read_excel_from_url(url=url, sheetname=sheetname)
    return [
        {
            "clean_title": title_function(row),
            "soc": row["SOC2020"],
            "std_title": row[title_field],
        }
        for row in coding_data
        if all(
            (row["SOC2020"] != "}}}}", row["SOC2020"], row[title_field])
        )  # Exclude rows with invalid or blank fields
    ]


def make_lookup(data, lookup_from, lookup_to):
    """Create a one-sided lookup of co-occuring values within a list of dictionaries.

    For example:

         data = [
             {"name": "joel", "surname": "klinger"},
             {"name": "joel", "surname": "bloggs"},
             {"name": "joe", "surname": "bloggs"},
             {"name": "joe", "surname": "blags"},
         ] * 100
         make_lookup(data, "name", "surname")

    will give:

         {"joel": ["klinger", "bloggs"], "joe": ["blags", "bloggs"]}
    """
    lookup = defaultdict(set)
    for row in data:
        lookup[row[lookup_from]].add(row[lookup_to])
    return {key: list(values) for key, values in lookup.items()}


def generate_job_stopwords():
    """Load and sort the local stopword metadata"""
    stopwords = load_metadata("job_stopwords")
    return sorted(stopwords, key=lambda s: s.count(" "), reverse=True)


def generate_locations():
    """
    Generate a standardised list of locations from the world cities data,
    additionally enriched using UK counties data, and removing location names
    which overlap with job titles (e.g. Sale)
    """
    world_locations = read_world_cities()
    uk_locations = filter(lambda row: row["ctry"] == "United Kingdom", world_locations)
    towns = set(row["town"] for row in uk_locations)
    counties = set(load_json_from_s3("uk_locations"))
    bad_locations = set(load_metadata("bad_locations"))
    locations = ({"borough of"} | counties | towns) - bad_locations
    return sorted(map(str.lower, locations))


def save_metadata_to_s3():
    """The 'main' function: populates S3 with JSON files of metadata and lookups"""
    # Extract SOC data from it's consituent parts ("jobs" and "groups")
    soc_jobs = load_ons_data(SHEETNAME_CODING, standardise_title, "INDEXOCC")
    soc_groups = load_ons_data(
        SHEETNAME_GROUPS, standardise_group_title, "SOC2020 Group Title"
    )
    soc_data = soc_jobs + soc_groups

    # Create a lookup of {clean title --> SOC} and {SOC --> standard title}
    for prefix, obj in (
        ("clean_title_to_soc_code", make_lookup(soc_data, "clean_title", "soc")),
        ("clean_title_to_std_title", make_lookup(soc_data, "clean_title", "std_title")),
        ("soc_code_to_std_title", make_lookup(soc_data, "soc", "std_title")),
        ("job_stopwords", generate_job_stopwords()),
        ("locations", generate_locations()),
    ):
        save_json_to_s3(prefix, obj)
