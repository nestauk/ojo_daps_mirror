"""
skills.helper_utils
--------------

General helper utils (e.g. for I/O operations)

"""
from pathlib import Path
import boto3
import json
import pandas as pd
import numpy as np
import pickle

# Paths
DATA_PATH = Path(__file__).parent / "data"
MODELS_PATH = DATA_PATH.parent / "models"

# S3
BUCKET_NAME = "open-jobs-lake"
S3_PATH = "labs/skills/{}"

### GETTERS ###
def get_esco_skills():
    return pd.read_csv(DATA_PATH / "raw/esco/ESCO_skills_hierarchy.csv")


def get_skill_embeddings(bert_model="paraphrase-distilroberta-base-v1"):
    # Skill entity ids
    skill_ids = np.load(
        DATA_PATH / f"processed/embeddings/skills_labels_all_entity_ids.npy"
    )
    # Embeddings
    embed = np.load(
        DATA_PATH / f"processed/embeddings/skills_labels_all_{bert_model}.npy"
    )
    return skill_ids, embed


def pickle_model(model, filename, fpath=MODELS_PATH):
    """Dump a pickled model"""
    with open(f"{MODELS_PATH}/{filename}.p", "wb") as file:
        pickle.dump(model, file)


def load_pickled_model(filename, fpath=MODELS_PATH):
    """Load a pickled model"""
    with open(f"{MODELS_PATH}/{filename}.p", "rb") as file:
        return pickle.load(file)


def save_lookup(name, path_name):
    with open(f"{path_name}.json", "w") as outfile:
        json.dump(name, outfile, indent=4)


def get_lookup(path_name):
    with open(f"{path_name}.json", "r") as infile:
        return json.load(infile)


### S3 UTILS ###
def save_to_s3(filename, contents):
    """Saves the contents to the filename in {BUCKET_NAME}/{S3_PATH}"""
    s3 = boto3.resource("s3")
    obj = s3.Object(BUCKET_NAME, S3_PATH.format(filename))
    obj.put(Body=contents)


def load_from_s3(filename):
    """Loads the file contents from the filename at {BUCKET_NAME}/{S3_PATH}"""
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=S3_PATH.format(filename))
    return obj["Body"].read()


def save_json_to_s3(dict_to_save, fpath):
    """Save a json to the S3"""
    save_to_s3(fpath, bytes(json.dumps(dict_to_save, indent=4).encode("UTF-8")))


### I/O HELPER UTILS ###
def save_json(dict_so_save, fpath):
    """Save json locally"""
    json.dump(dict_so_save, open(fpath, "w"), indent=4)


### HELPER UTILS FOR FLOWS ###
def full_path(filename):
    """Gets the absolute path to a file in this directory"""
    return Path(__file__).parent.joinpath(filename)
