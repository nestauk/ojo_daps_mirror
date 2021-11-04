import pytest
from tempfile import NamedTemporaryFile
from ojd_daps.flows.enrich.labs.skills.skills_detection_utils import (
    setup_spacy_model,
    chunk_forms,
    process_chunk_forms,
    flatten_skills_labels,
    process_label_forms,
    create_surface_forms,
    surface_form_dataframe,
    deduplicate_surface_forms,
    create_documents,
    tfidf_vectorise,
    tfidf_phrase_sums,
    tfidf_representativity_mask,
    find_best_n,
    create_phrase_matcher,
    save_model_locally,
    #     save_model_in_s3,
    save_removed_forms,
    #     load_model,
    detect_skills,
    filter_rows,
    DEF_LANGUAGE_MODEL,
    clean_text,
    remove_surface_forms,
    assign_surface_forms,
    count_surface_forms,
    frequency_refinement,
)

# from ojd_daps.flows.enrich.labs.skills.helper_utils import load_from_s3, save_to_s3
from types import GeneratorType
from spacy.matcher.phrasematcher import PhraseMatcher
from pandas import DataFrame, read_csv
from numpy import array
from numpy.random import rand
import os
from pathlib import Path
import pickle, json
from io import BytesIO
import spacy
import nltk

nltk.download("wordnet")
nltk.download("stopwords")

LOCAL_PATH = Path(__file__).parent


def test_setup_spacy_model():
    nlp = setup_spacy_model()
    pipes = nlp.analyze_pipes()["summary"]
    assert DEF_LANGUAGE_MODEL["model"] == f'{nlp.meta["lang"]}_{nlp.meta["name"]}'
    for component in DEF_LANGUAGE_MODEL["disable"]:
        assert (component in pipes.keys()) is False


def test_chunk_forms():
    nlp = setup_spacy_model()
    text = ["first item, second item", "third"]
    chunk_list_generator = chunk_forms(text, nlp)
    assert type(chunk_list_generator) is GeneratorType
    assert list(chunk_list_generator) == [["first item", "second item"], []]
    assert list(chunk_forms(["use programming languages"], nlp)) == [
        ["programming language"]
    ]


def test_process_chunk_forms():
    nlp = setup_spacy_model()
    assert (
        process_chunk_forms(
            entities=[1, 2, 3],
            texts=["First item, Second item", "Third", "Fourth item"],
            nlp=nlp,
        )
        == [(1, "first item"), (1, "second item"), (3, "fourth item")]
    )
    assert (
        process_chunk_forms(
            entities=[1, 1, 1],
            texts=["first item, second item", "third", "fourth item"],
            nlp=nlp,
        )
        == [(1, "first item"), (1, "second item"), (1, "fourth item")]
    )


def test_flatten_skills_labels():
    assert flatten_skills_labels([1, 2], ["one\nanother one", "two\nanother two"]) == (
        [1, 1, 2, 2],
        ["one", "another one", "two", "another two"],
    )


def test_process_label_forms():
    assert process_label_forms(
        entities=[1, 2], list_of_labels=["First label", "Second label"]
    ) == [(1, "first label"), (2, "second label")]
    assert process_label_forms(
        entities=[1, 1], list_of_labels=["First label", "Second label"]
    ) == [(1, "first label"), (1, "second label")]
    assert process_label_forms(
        entities=[1, 2], list_of_labels=["First-label", "2nd label"]
    ) == [(1, "first label"), (2, "2nd label")]


def test_create_surface_forms():
    nlp = setup_spacy_model()
    assert (
        create_surface_forms(
            entities=[1, 2],
            texts=["First item, Second item", "Third"],
            chunk=False,
            nlp=nlp,
        )
        == [(1, "first item second item"), (2, "third")]
    )
    assert create_surface_forms(
        entities=[1, 2], texts=["First item, Second item", "Third"], chunk=True, nlp=nlp
    ) == [(1, "first item"), (1, "second item")]


def test_surface_form_dataframe():
    surface_forms = {
        "label": [(1, "manage musical staff"), (2, "perform planning")],
        "chunk": [(1, "musical staff"), (2, "planning")],
    }
    df = surface_form_dataframe(surface_forms)
    assert type(df) is DataFrame
    assert list(df.columns) == ["entity", "surface_form", "surface_form_type"]
    assert set(df.entity.unique()) == set([1, 2])
    for form_type in surface_forms:
        assert form_type in df.surface_form_type.unique()
        assert len(df[df.surface_form_type == form_type]) == len(
            surface_forms[form_type]
        )


def test_deduplicate_surface_forms():
    surface_form_table = DataFrame(
        data={
            "entity": [1, 1, 2],
            "surface_form": ["unique form", "duplicate form", "duplicate form"],
            "surface_form_type": ["label", "chunk", "label"],
        }
    )
    deduplicated_table = deduplicate_surface_forms(
        surface_form_table, sort_order=["label", "chunk"]
    )
    assert len(deduplicated_table) == 2
    assert sum(deduplicated_table.surface_form.duplicated()) == 0
    assert len(deduplicated_table.surface_form_type.unique()) == 1
    # Test using a different sort order
    table_with_different_sort_order = deduplicate_surface_forms(
        surface_form_table, sort_order=["chunk", "label"]
    )
    assert len(table_with_different_sort_order.surface_form_type.unique()) == 2


def test_create_documents():
    assert list(create_documents((["one", "two"], ["cat", "dogs"]))) == [
        "one cat",
        "two dogs",
    ]
    with pytest.raises(ValueError):
        create_documents((["one"], ["cat", "dogs"]))


def test_tfidf_vectorise():
    documents = ["blue house", "red house"]
    tfidf_matrix, vocabulary = tfidf_vectorise(documents)
    assert tfidf_matrix.shape == (2, 3)
    assert type(vocabulary) is dict
    for token in [token for doc in documents for token in doc.split(" ")]:
        assert token in vocabulary.keys()


def test_tfidf_phrase_sums():
    mock_vocabulary = {"red": 0, "blue": 1, "house": 2}
    mock_tfidf_matrix = array([[1, 0, 1], [0, 1, 1]])
    assert (tfidf_phrase_sums("red", mock_vocabulary, mock_tfidf_matrix)) == [1, 0]
    assert (tfidf_phrase_sums("blue", mock_vocabulary, mock_tfidf_matrix)) == [0, 1]
    assert (tfidf_phrase_sums("house", mock_vocabulary, mock_tfidf_matrix)) == [1, 1]
    assert (tfidf_phrase_sums("red house", mock_vocabulary, mock_tfidf_matrix)) == [
        2,
        1,
    ]


def test_tfidf_representativity_mask():
    # Test with an illustrative example: imagine two skills entities that are
    # related to 'using phone' (entity=0) and 'computer science' (entity=1)
    mock_surface_form_df = DataFrame(
        data={
            "entity": [0, 0, 0, 1],
            "surface_form": ["use phone", "phone", "computer", "computer science"],
            "surface_form_type": ["label", "chunk", "chunk", "chunk"],
        }
    )
    mock_vocabulary = {"use": 0, "phone": 1, "computer": 2, "science": 3}
    mock_tfidf_matrix = array([[1, 1, 0, 0], [0, 0, 1, 1]])
    mask = tfidf_representativity_mask(
        mock_surface_form_df, mock_tfidf_matrix, mock_vocabulary
    )
    # Surface form 'computer' is not representative of the 0-th entity
    assert mask == [True, True, False, True]


def test_find_best_n():
    list_of_numbers = [0.1, 0.2, 100]
    n = 2
    best_n, best_n_values = find_best_n(list_of_numbers, n=n)
    assert len(best_n) == n
    assert best_n[0] == 2
    assert best_n_values[0] == max(list_of_numbers)


def test_filter_rows():
    df = DataFrame(data={"surface_form": ["one", "two"]}, index=[1, 100])
    # Test when all rows are kept
    new_df, removed_forms = filter_rows(df, rows_to_keep=[True, True], forms_to_keep=[])
    assert new_df.equals(df)
    assert len(removed_forms) == 0
    # Test when when some rows are removed
    new_df, removed_forms = filter_rows(
        df, rows_to_keep=[True, False], forms_to_keep=[]
    )
    assert len(new_df) == 1
    assert len(removed_forms) == 1
    assert removed_forms[0]["surface_form"] == "two"
    # Test when when some rows are manually kept
    new_df, removed_forms = filter_rows(
        df, rows_to_keep=[True, False], forms_to_keep=["two"]
    )
    assert new_df.equals(df)
    assert len(removed_forms) == 0


def test_remove_surface_forms():
    mock_df = DataFrame(data={"surface_form": ["a", "b", "c"]})
    new_df = DataFrame(data={"surface_form": ["b", "c"]})
    assert remove_surface_forms(mock_df, ["a"]).reset_index(drop=True).equals(new_df)
    assert remove_surface_forms(mock_df, ["d"]).equals(mock_df)


def test_assign_surface_forms():
    # Initial table with surface forms
    mock_surface_form_table = DataFrame(
        data={
            "entity": [0, 1],
            "surface_form": ["apple", "orange"],
            "surface_form_type": ["label", "label"],
        }
    )
    # Instructions for manual surface form assignments
    mock_forms_to_assign = [
        # Surface form that will be re-assigned to a new entity
        {"entity": 123, "surface_form": "apple", "surface_form_type": "manual"},
        # New surface form
        {"entity": 2, "surface_form": "new form", "surface_form_type": "manual"},
    ]
    new_df = assign_surface_forms(mock_surface_form_table, mock_forms_to_assign)
    assert len(new_df) == 3
    assert new_df[new_df.surface_form == "apple"].entity.iloc[0] == 123
    assert "new form" in new_df.surface_form.to_list()


def test_create_phrase_matcher():
    nlp = spacy.load("en_core_web_sm")
    list_of_phrases = ["red apple", "orange"]
    matcher = create_phrase_matcher(list_of_phrases, nlp)
    assert len(matcher(nlp.make_doc("buy a red apple and an orange"))) == 2
    assert len(matcher(nlp.make_doc("a sentence without the phrases"))) == 0


def test_save_model_locally():
    mock_model = {
        "name": "mock_model",
        "surface_forms": DataFrame(data={"column": [1, 2, 3]}),
        "matcher": [],
        "nlp": DEF_LANGUAGE_MODEL,
    }
    save_model_locally(mock_model, local_path=LOCAL_PATH)
    csv_file = LOCAL_PATH / f"surface_forms_table_{mock_model['name']}.csv"
    pickle_file = LOCAL_PATH / f"surface_form_matcher_{mock_model['name']}.pickle"
    try:
        # Check the CSV table
        assert os.path.exists(csv_file)
        assert read_csv(csv_file).equals(mock_model["surface_forms"])
        assert os.path.exists(pickle_file)
        # Check the pickle file
        loaded_model = pickle.load(open(pickle_file, "rb"))
        assert type(loaded_model) is dict
        assert loaded_model["name"] == mock_model["name"]
        assert loaded_model["matcher"] == mock_model["matcher"]
        assert loaded_model["surface_forms"].equals(mock_model["surface_forms"])
        assert loaded_model["nlp"] == DEF_LANGUAGE_MODEL
        os.remove(csv_file)
        os.remove(pickle_file)
    except (AssertionError, TypeError, ValueError) as error:
        for file in [csv_file, pickle_file]:
            if os.path.exists(file):
                os.remove(file)
        raise error


def test_save_removed_forms():
    mock_removed_forms_dict = {
        "refinement_step_name": [
            {"entity": 0, "surface_form": "removed_form", "surface_form_type": "chunk"}
        ]
    }
    model_name = "mock"
    save_removed_forms(
        mock_removed_forms_dict,
        model_name=model_name,
        local_path=LOCAL_PATH,
        s3_copy=False,
    )
    pickle_file = LOCAL_PATH / f"surface_forms_removed_{model_name}.json"
    try:
        assert json.load(open(pickle_file, "rb")) == mock_removed_forms_dict
        os.remove(pickle_file)
    except AssertionError as error:
        if os.path.exists(pickle_file):
            os.remove(pickle_file)


def test_detect_skills():
    # Create a mock spacy matcher model
    nlp = spacy.load("en_core_web_sm")
    list_of_surface_forms = ["red apple", "orange"]
    mock_model = {
        "name": "mock_model",
        "surface_forms": DataFrame(
            data={
                "surface_form": list_of_surface_forms,
                "surface_form_type": ["label", "label"],
                "entity": [0, 1],
                "preferred_label": ["apple", "orange"],
                "predicted_q": [0.7, 0.8],
                "is_predicted_OK": [1, 1],
                "manual_OK": [1, 1],
                "cluster_0": [0, 1],
                "cluster_1": [1, 2],
                "cluster_2": [0, 4],
                "label_cluster_0": [0, 1],
                "label_cluster_1": [2, 3],
                "label_cluster_2": [4, 5],
            }
        ),
        "matcher": create_phrase_matcher(list_of_surface_forms, nlp),
        "nlp": DEF_LANGUAGE_MODEL,
    }
    nlp = setup_spacy_model(mock_model["nlp"])
    text = "red apples grow in the garden"
    assert len(detect_skills(text, mock_model, nlp)) == 0
    skills = detect_skills(clean_text(text), mock_model, nlp)
    assert type(skills) is DataFrame
    assert len(skills) == 1
    skills = detect_skills(clean_text(text), mock_model, nlp, return_dict=True)
    assert type(skills) is list
    assert type(skills[0]) is dict
    for key in ["surface_form", "preferred_label", "entity", "surface_form_type"]:
        assert key in skills[0]


def test_count_surface_forms():
    assert count_surface_forms(
        [[{"surface_form": "a"}], [{"surface_form": "a"}, {"surface_form": "b"}]]
    ) == {"a": 2, "b": 1}
    assert (
        count_surface_forms(
            [
                [{"arbitrary_key": "a"}],
                [{"arbitrary_key": "a"}, {"arbitrary_key": "b"}],
            ],
            key="arbitrary_key",
        )
        == {"a": 2, "b": 1}
    )


def test_frequency_refinement():
    # Initial table with surface forms
    mock_surface_form_table = DataFrame(
        data={
            "entity": [0, 1, 2, 3, 4],
            "surface_form": ["apple", "two apples", "orange", "banana", "flower"],
            "surface_form_type": [
                "label_pref",
                "label_alt",
                "label_alt",
                "chunk",
                "chunk",
            ],
        }
    )
    # Surface form counts
    mock_counts = {"apple": 20, "two apples": 30, "orange": 30, "banana": 2}
    assert frequency_refinement(mock_surface_form_table, mock_counts, 25) == [
        True,
        True,
        False,
        True,
        True,
    ]
    mock_surface_form_table.surface_form_type = ["chunk"] * 5
    assert frequency_refinement(mock_surface_form_table, mock_counts, 25) == [
        False,
        True,
        False,
        True,
        True,
    ]


# ### Tests with a dependency on S3
# def test_save_model_in_s3():
#     mock_model = {
#         "name": "mock_model",
#         "surface_forms": DataFrame(data={"column": [1, 2, 3]}),
#         "matcher": [],
#         "nlp": DEF_LANGUAGE_MODEL,
#     }
#     model_name = mock_model["name"]
#     save_model_in_s3(mock_model)
#     # Check the pickle file
#     loaded_model = pickle.loads(
#         load_from_s3(f"models/surface_form_matcher_{model_name}.pickle")
#     )
#     assert loaded_model["name"] == mock_model["name"]
#     assert loaded_model["matcher"] == mock_model["matcher"]
#     assert loaded_model["surface_forms"].equals(mock_model["surface_forms"])
#     assert loaded_model["nlp"] == DEF_LANGUAGE_MODEL
#     # Check the CSV table
#     df = read_csv(BytesIO(load_from_s3(f"models/surface_forms_table_{model_name}.csv")))
#     assert df.equals(mock_model["surface_forms"])


# def test_load_model():
#     # Create a mock model
#     mock_model = {
#         "name": "mock_model",
#         "surface_forms": DataFrame(data={"column": [1, 2, 3]}),
#         "matcher": [],
#         "nlp": DEF_LANGUAGE_MODEL,
#     }
#     # Test loading from local storage
#     model_file = LOCAL_PATH / f'surface_form_matcher_{mock_model["name"]}.pickle'
#     pickle.dump(mock_model, open(model_file, "wb"))
#     try:
#         loaded_model = load_model(
#             mock_model["name"], from_local=True, local_path=LOCAL_PATH
#         )
#         assert loaded_model["name"] == mock_model["name"]
#         assert loaded_model["matcher"] == mock_model["matcher"]
#         assert loaded_model["surface_forms"].equals(mock_model["surface_forms"])
#         assert loaded_model["nlp"] == DEF_LANGUAGE_MODEL
#         os.remove(model_file)
#     except (AssertionError, NameError, KeyError) as error:
#         if os.path.exists(model_file):
#             os.remove(model_file)
#         raise error
#     # Test loading from s3
#     save_to_s3(
#         f'models/surface_form_matcher_{mock_model["name"]}.pickle',
#         pickle.dumps(mock_model),
#     )
#     loaded_model = load_model(mock_model["name"])
#     assert loaded_model["name"] == mock_model["name"]
#     assert loaded_model["matcher"] == mock_model["matcher"]
#     assert loaded_model["surface_forms"].equals(mock_model["surface_forms"])
#     assert loaded_model["nlp"] == DEF_LANGUAGE_MODEL
