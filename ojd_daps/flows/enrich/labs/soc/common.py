"""
common
------

Text preprocessing and file IO
"""

import json
import yaml
import boto3
from functools import lru_cache
from pathlib import Path
import re
from itertools import filterfalse

S3_PATH = "jobs-metadata/{}"
BUCKET_NAME = "open-jobs-lake"
SELECTED_PUNCT = set(r'!"$%&\()*,/:;<=>?@[\\]^_`{|}~-#.')  # only keep "#'+.-"
RE_SPACES = re.compile(" +")
RE_TERMS = re.compile(r"(\w+)")


@lru_cache()
def load_from_s3(filename):
    """Loads the file contents from the filename at {BUCKET_NAME}/{S3_PATH}"""
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=S3_PATH.format(filename))
    return obj["Body"].read().decode()


def save_to_s3(filename, contents):
    """Saves the contents to the filename in {BUCKET_NAME}/{S3_PATH}"""
    s3 = boto3.resource("s3")
    obj = s3.Object(BUCKET_NAME, S3_PATH.format(filename))
    obj.put(Body=contents)


def save_json_to_s3(prefix, data):
    """Save data as json on S3"""
    save_to_s3(f"{prefix}.json", json.dumps(data))


@lru_cache()
def load_json_from_s3(prefix):
    """Save data as json from S3"""
    return json.loads(load_from_s3(f"{prefix}.json"))


@lru_cache()
def _load_metadata():
    """Preload the local metadata"""
    path_to_here = Path(__file__)
    with open(path_to_here.parent / "metadata.yaml") as f:
        return yaml.load(f, Loader=yaml.BaseLoader)


def load_metadata(key):
    """Load a specific key from the local metadata"""
    metadata = _load_metadata()
    return metadata[key]


def remove_digits(text):
    """Remove and numeric digits from a string"""
    return "".join(filterfalse(str.isdigit, text))


def replace_punctuation(text):
    """Remove all punctuation from the string"""
    return " ".join(RE_TERMS.findall(text))


def standardise_text(text):
    """Standardise a string"""
    text = replace_punctuation(text)
    text = remove_digits(text)
    text = text.strip().lower()
    return RE_SPACES.sub(" ", text)


def replace_or_remove(text_to_change, words_to_remove):
    """
    Replace all occurences of all terms in `words_to_remove`
    from `text_to_change`. If `words_to_remove` is a list, all
    terms will be replaced by "". If `words_to_remove` is a dict,
    each key in `words_to_remove` will be replaced by the correponding value.
    """
    for word in words_to_remove:
        if word not in text_to_change:
            continue
        replace = words_to_remove[word] if type(words_to_remove) is dict else ""
        text_to_change = text_to_change.replace(word, replace)
    return text_to_change


def remove_prefix(text, prefix):
    """Remove prefix words not included in SOC index"""
    try:
        first_part, rest = text.split(" ", 1)
    except ValueError:
        first_part, rest = text, ""
    return rest if first_part == prefix else text


@lru_cache()
def clean_raw_job_title(text):
    for prefix in ["job_stopwords", "locations", "acronyms_lookup"]:
        words_to_remove = load_json_from_s3(prefix)
        text = standardise_text(text)
        text = replace_or_remove(text, words_to_remove)
    text = standardise_text(text)
    for prefix in load_metadata("ignore_prefixes"):
        text = remove_prefix(text, prefix)
    return standardise_text(text)
