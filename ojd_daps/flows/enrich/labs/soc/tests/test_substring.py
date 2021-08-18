from unittest import mock
from ojd_daps.flows.enrich.labs.soc.substring_utils import (
    load_titles,
    predict_soc_and_title,
    load_model,
    apply_model,
)

PATH = "ojd_daps.flows.enrich.labs.soc.substring_utils.{}"


@mock.patch(
    PATH.format("load_json_from_s3"),
    return_value=["a very long title", "a title", "a longer title"],
)
def test_load_titles(mocked_load):
    assert load_titles() == ["a very long title", "a longer title", "a title"]


@mock.patch(
    PATH.format("load_titles"),
    return_value=["yet another clean title", "clean title"],
)
@mock.patch(PATH.format("load_json_from_s3"))
def test_predict_soc_and_title(mocked_load, mocked_titles):
    """Tests several cases in the prediction logic, including:

    - (query_term in search_terms) --> 'Exact match'
    - any(query_term in term for term in search_terms) --> 'Forward partial match'
    - any(term in query_term for term in search_terms) --> 'Reverse partial match'
    """

    title_to_soc = {
        "clean title": "CleanTitleCode",
        "yet another clean title": "AnotherTitleCode",
    }
    title_to_std_title = {
        "clean title": "CleanTitle",
        "yet another clean title": "AnotherTitle",
    }
    load_dict = {
        "clean_title_to_soc_code": title_to_soc,
        "clean_title_to_std_title": title_to_std_title,
    }
    mocked_load.side_effect = lambda x: load_dict[x]

    # Exact: match to 'the name of yet another clean title'
    assert predict_soc_and_title("yet another clean title") == (
        "AnotherTitleCode",
        "AnotherTitle",
    )

    # Forward partial: 'yet another clean title' is in 'the name of yet another clean title'
    assert predict_soc_and_title("the name of yet another clean title") == (
        "AnotherTitleCode",
        "AnotherTitle",
    )

    # Forward partial: 'another clean title' is in 'the name of yet another clean title'
    assert predict_soc_and_title("another clean title") == (
        "AnotherTitleCode",
        "AnotherTitle",
    )

    # Reverse partial: 'BONUS the name of yet another clean title' is in
    # 'the name of yet another clean title'
    assert predict_soc_and_title("BONUS the name of yet another clean title") == (
        "AnotherTitleCode",
        "AnotherTitle",
    )

    # Exact: match to 'clean title'
    assert predict_soc_and_title("clean title") == (
        "CleanTitleCode",
        "CleanTitle",
    )

    # Forward partial: 'title' is in 'clean title'
    assert predict_soc_and_title("title") == (
        "CleanTitleCode",
        "CleanTitle",
    )

    # Reverse partial: 'clean title' is in 'yet clean title'
    assert predict_soc_and_title("yet clean title") == (
        "CleanTitleCode",
        "CleanTitle",
    )

    # No match
    assert predict_soc_and_title("something else") == (None, None)


@mock.patch(PATH.format("clean_raw_job_title"), side_effect=lambda x: x + 2)
@mock.patch(PATH.format("predict_soc_and_title"), side_effect=lambda x: x * 3 - 1)
def test_load_model(mocked_clean, mocked_predict):
    model = load_model()
    assert model is load_model()
    assert model({"job_title_raw": 10}) == (10 + 2) * 3 - 1
    assert model({"job_title_raw": 7}) == (7 + 2) * 3 - 1


@mock.patch(PATH.format("load_model"), return_value=lambda x: x * 3 - 1)
def test_apply_model(mocked_load):
    assert apply_model(10) == 10 * 3 - 1
    assert apply_model(7) == 7 * 3 - 1
