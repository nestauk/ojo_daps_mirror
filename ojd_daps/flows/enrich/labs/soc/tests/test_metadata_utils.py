from unittest import mock
import types
from ojd_daps.flows.enrich.labs.soc.metadata_utils import (
    get_file_link,
    WordNetLemmatizer,
    lemmatise,
    standardise_title,
    separate_plural_edgecases,
    load_plurals,
    standardise_group_title,
    requests_get,
    read_world_cities,
    read_excel_from_url,
    load_ons_data,
    make_lookup,
    generate_job_stopwords,
    generate_locations,
    save_metadata_to_s3,
)
from ojd_daps.flows.enrich.labs.soc.metadata_utils import requests, SHEETNAME_CODING

PATH = "ojd_daps.flows.enrich.labs.soc.metadata_utils.{}"


def test_get_file_link():
    """Check the file link actually exists (might occasionally have a transient fail)"""
    link = get_file_link()
    r = requests.head(link)
    r.raise_for_status()


def test_WordNetLemmatizer():  # NB: WordNetLemmatizer is actually a local patch, not NLTK WNL
    assert WordNetLemmatizer() is WordNetLemmatizer()


def test_lemmatise():
    assert lemmatise("policies") == lemmatise("policy")


@mock.patch(PATH.format("load_metadata"), return_value=["bonus terms"])
def test_standardise_title(mocked):
    # All fields
    row = {
        "IND": "the--- ind Title",
        "ADD": "the ADD !title mfr?",
        "INDEXOCC": "the indexocc title, and OTHER things!!",
        "foo": "bar",
    }
    expected = (
        "and other things the indexocc title the ind title the add title bonus terms"
    )
    assert standardise_title(row) == expected

    # NOS value in IND, Null ADD, no mfr
    row = {
        "IND": "nos",
        "ADD": None,
        "INDEXOCC": "the indexocc title, and OTHER things!!",
        "foo": "bar",
    }
    expected = "and other things the indexocc title"
    assert standardise_title(row) == expected


@mock.patch(PATH.format("load_metadata"), return_value=[1, 2, 3])
def test_load_plurals(mocked):
    assert load_plurals() is load_plurals()


@mock.patch(
    PATH.format("load_plurals"), return_value={"policies", "academies", "sugars"}
)
def test_load_plurals(mocked):
    edgecases, non_edgecases = separate_plural_edgecases(
        {"policies", "elephants", "other", "sugars"}
    )
    assert edgecases == {"policies", "sugars"}
    assert non_edgecases == {"elephants", "other"}


@mock.patch(PATH.format("separate_plural_edgecases"))
@mock.patch(PATH.format("lemmatise"))
def test_standardise_group_title(mocked_lem, mocked_seperate):
    mocked_seperate.side_effect = lambda terms: (terms[:3], terms[3:])
    mocked_lem.side_effect = lambda x: x.title()
    row = {"SOC2020 Group Title": "This is A TITLE with EDgecases"}
    assert standardise_group_title(row) == "Title With Edgecases this is a"


def test_requests_get():
    content = requests_get("http://example.com/")
    assert content is requests_get("http://example.com/")
    assert isinstance(content, bytes)
    assert len(content) > 0


def test_read_excel_from_url():
    iterable = read_excel_from_url(
        "https://www.ons.gov.uk/file?uri=/methodology/classificationsandstandards/standardoccupationalclassificationsoc/soc2020/soc2020volume2codingrulesandconventions/soc2020volume2thecodingindex010221.xlsx",
        sheetname=SHEETNAME_CODING,
    )
    assert isinstance(iterable, types.GeneratorType)
    for i, line in enumerate(iterable):
        assert isinstance(line, dict)
        assert len(line) > 2
    assert i > 100


def test_read_world_cities():
    iterable = read_world_cities()
    assert isinstance(iterable, types.GeneratorType)
    for i, line in enumerate(iterable):
        assert isinstance(line, dict)
        assert len(line) > 2
    assert i > 100


@mock.patch(PATH.format("get_file_link"))
@mock.patch(PATH.format("read_excel_from_url"))
def test_load_ons_data(mocked_read, mocked_get):
    mocked_read.return_value = [
        {"SOC2020": "abc", "title field": "the Title", "other field": "the other"}
    ] * 10

    data = load_ons_data(
        sheetname="dummy",
        title_function=lambda row: row["other field"].title(),
        title_field="title field",
    )
    assert all(
        row == {"clean_title": "The Other", "soc": "abc", "std_title": "the Title"}
        for row in data
    )


def test_make_lookup():
    data = [
        {"name": "joel", "surname": "klinger"},
        {"name": "joel", "surname": "bloggs"},
        {"name": "joe", "surname": "bloggs"},
        {"name": "joe", "surname": "blags"},
    ] * 100
    lookup = make_lookup(data, "name", "surname")
    expected = {"joel": ["klinger", "bloggs"], "joe": ["blags", "bloggs"]}
    assert lookup.keys() == expected.keys()
    for k in lookup:
        assert sorted(lookup[k]) == sorted(expected[k])


@mock.patch(PATH.format("load_metadata"))
def test_generate_job_stopwords(mocked_load):
    mocked_load.return_value = ["two spaces ", "three space s ", "one space"]
    assert generate_job_stopwords() == ["three space s ", "two spaces ", "one space"]


@mock.patch(PATH.format("read_world_cities"))
@mock.patch(PATH.format("load_json_from_s3"))
@mock.patch(PATH.format("load_metadata"))
def test_generate_locations(mocked_load, mocked_s3, mocked_read):
    mocked_read.return_value = [
        {"ctry": "United Kingdom", "town": "Newcastle"},
        {"ctry": "United States", "town": "Alabama"},
        {"ctry": "United Kingdom", "town": "Mobile"},
        {"ctry": "United Kingdom", "town": "Aberdeen"},
        {"ctry": "United Kingdom", "town": "Sale"},
        {"ctry": "United Kingdom", "town": "Dewsbury"},
    ]
    mocked_load.return_value = ["Sale", "Mobile"]
    mocked_s3.return_value = ["county 1", "county 3"]
    assert generate_locations() == [
        "aberdeen",
        "borough of",
        "county 1",
        "county 3",
        "dewsbury",
        "newcastle",
    ]


@mock.patch(PATH.format("load_ons_data"))
@mock.patch(PATH.format("generate_job_stopwords"), return_value="foo")
@mock.patch(PATH.format("generate_locations"), return_value="bar")
@mock.patch(PATH.format("save_json_to_s3"))
def test_save_metadata_to_s3(mocked_s3, mocked_locs, mocked_stops, mocked_load):
    mocked_load.side_effect = lambda *args: [
        {"clean_title": args[2], "soc": args[0], "std_title": args[0]}
    ]
    save_metadata_to_s3()
    call_args = [(prefix, obj) for ((prefix, obj), kwargs) in mocked_s3.call_args_list]
    assert call_args == [
        (
            "clean_title_to_soc_code",
            {
                "INDEXOCC": ["SOC2020 coding index V4"],
                "SOC2020 Group Title": ["SOC2020 structure"],
            },
        ),
        (
            "clean_title_to_std_title",
            {
                "INDEXOCC": ["SOC2020 coding index V4"],
                "SOC2020 Group Title": ["SOC2020 structure"],
            },
        ),
        (
            "soc_code_to_std_title",
            {
                "SOC2020 coding index V4": ["SOC2020 coding index V4"],
                "SOC2020 structure": ["SOC2020 structure"],
            },
        ),
        ("job_stopwords", "foo"),
        ("locations", "bar"),
    ]
