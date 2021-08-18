from ojd_daps.flows.pre_enrich.vectorise_descriptions import (
    load_model,
    truncate_and_jsonify,
    encode_job_ads,
    MODEL_NAME,
    SentenceTransformer,
)
import numpy as np


def test_load_model():
    assert type(load_model(MODEL_NAME)) is SentenceTransformer


def test_truncate_and_jsonify():
    vector = np.array([1.1111111, 2.22345444, 3.12])
    expected = '["1.1111", "2.2235", "3.1200"]'
    _vector = truncate_and_jsonify(vector=vector, decimal_places=4)
    assert _vector == expected


def test_encode_job_ads():
    job_ads = [
        {"description": "this is some text", "id": 11},
        {"description": "this is some text", "id": 2},
        {"description": "this is some more text", "id": 32},
    ]
    vecs = encode_job_ads(job_ads, decimal_places=2)
    assert len(vecs) == 3
    assert sorted(v["id"] for v in vecs) == [2, 11, 32]
    assert vecs[0]["vector"] == vecs[1]["vector"]  # First two texts are the same
    assert vecs[0]["vector"] != vecs[2]["vector"]  # Third text is different
