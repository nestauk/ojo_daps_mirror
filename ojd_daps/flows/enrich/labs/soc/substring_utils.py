"""
substring utils
---------------

The `apply_model` method found here takes a row of raw job ad data
and returns a list of SOC codes and Job Titles that match.

The method applies the following procedure, by comparing a "cleaned" job title
(see `metadata_utils.clean_raw_job_title`) and comparing to
cleaned raw job titles from the SOC lookup table (downloaded in `metadata_utils`):

- Firstly, attempt to find an exact match between cleaned titles
- If this fails then find the best partial match from (roughly)
    1) `any(term in query_term for term in search_terms)`, or
    2) `any(query_term in term for term in search_terms)`

    In the case of a match in either 1) or 2), the first match is taken,
    with `search_terms` assumed to be ordered by number of terms in order
    to find the most exact match first. If a match is found by both 1) and 2)
    then the shortest is taken - as this is assumed to be the most exact.
"""

from functools import lru_cache
from .common import clean_raw_job_title, load_json_from_s3


def _load_titles(max_terms):
    """Load and filter titles"""
    titles = load_json_from_s3("clean_title_to_soc_code")
    titles = filter(lambda title: len(title) > 3, titles)  # ignore short titles
    titles = filter(lambda title: title.count(" ") <= max_terms - 1, titles)
    return titles


@lru_cache()
def load_titles(max_terms=7):
    """Load sorted clean titles"""
    return sorted(
        _load_titles(max_terms), key=lambda title: title.count(" "), reverse=True
    )


@lru_cache()
def load_titles_set(max_terms=7):
    """Load a set of clean titles"""
    return set(_load_titles(max_terms))


def lookup_soc_and_title(title):
    """Return the SOC code and standard title for this clean title"""
    # Note that the json lookups are cached
    title_to_soc = load_json_from_s3("clean_title_to_soc_code")
    clean_to_std_title = load_json_from_s3("clean_title_to_std_title")
    # Perform the lookups and return
    soc_codes = title_to_soc[title]
    std_titles = clean_to_std_title[title]
    return soc_codes, std_titles


def gequal(a, b, reverse):
    """A reversable >= operator"""
    if reverse:
        return len(b) - len(a) > 0
    else:
        return len(a) - len(b) > 0


def contains(a, b, reverse):
    """A reversable containment operator"""
    return (b in a) if reverse else (a in b)


def partial_scan(query_term, search_terms, reverse=False):
    """
    Find the first partial match in `search_terms` to the `query_term`.

    The matching strategy is either "forwards" (`reverse = False`) or
    "backwards" (`reverse = True`). The forward approach looks for the
    first occurence where the query contains a search term, whereas the
    backwards approach looks for the first occurence where a search term
    contains the query term.
    """
    search_terms = reversed(search_terms) if reverse else search_terms
    # Only use search terms which can be long/short enough to contain or
    # be contained in the query term
    filtered_terms = filter(
        lambda t: gequal(query_term, t, reverse=reverse), search_terms
    )
    # Terms assumed to be sorted, to find most exact first
    for term in filtered_terms:
        # Find the first partial match
        if contains(term, query_term, reverse):
            return term
    return None


@lru_cache()
def predict_soc_and_title(clean_title):
    """Takes a clean job, returns a soc code"""
    # First try an exact match
    if clean_title in load_titles_set():  # NB: cached
        return lookup_soc_and_title(clean_title)
    # Otherwise search one-by-one for a partial match
    titles = load_titles()  # NB: cached
    fwd_match = partial_scan(clean_title, titles, reverse=False)
    bwd_match = partial_scan(clean_title, titles, reverse=True)
    # Return the longest non-null value
    non_nulls = filter(None, (fwd_match, bwd_match))
    for match in sorted(non_nulls, key=len, reverse=True):
        return lookup_soc_and_title(match)
    return None, None  # No result found


@lru_cache()
def load_model():
    """Load a model for predicting SOC and Job Title"""

    def model(row):
        raw_title = row["job_title_raw"]
        clean_title = clean_raw_job_title(raw_title)
        return predict_soc_and_title(clean_title)

    return model


def apply_model(row):
    """Loads and applies the model to the given row,
    noting the load is implicit in soc_finder"""
    model = load_model()
    return model(row)
