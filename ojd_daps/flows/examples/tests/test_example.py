import pytest
from unittest import mock
from ojd_daps.flows.examples.example import halfway
from ojd_daps.flows.examples.example import request_row_as_json
from ojd_daps.flows.examples.example import page_exists
from ojd_daps.flows.examples.example import find_last_page
from ojd_daps.flows.examples.example import generate_page_numbers


PATH = "ojd_daps.flows.examples.example.{}"


def test_halfway():
    assert halfway(-1, 1) == 0
    assert halfway(-1, 0) == 0
    assert halfway(-1, 2) == 1
    assert halfway(-1, 3) == 1
    assert halfway(1, 2) == 2
    assert halfway(1, 3) == 2
    assert halfway(253, 21434) == 10844
    assert halfway(253, 21435) == 10844
    assert halfway(253, 21436) == 10845


@mock.patch(PATH.format("requests"))
def test_request(mocked_requests):
    request_row_as_json("joel", 23)
    request_row_as_json("nesta", 214)
    mocked_requests.assert_has_calls(
        [
            mock.call.get(f"https://swapi.dev/api/joel/23/"),
            mock.call.get().json(),
            mock.call.get(f"https://swapi.dev/api/nesta/214/"),
            mock.call.get().json(),
        ]
    )


@mock.patch(PATH.format("requests"))
def test_page_exists_200(mocked_requests):
    mocked_requests.head().status_code = 200
    mocked_requests.head().raise_for_status.side_effect = Exception
    assert page_exists("blah ", "blah blah") == True


@mock.patch(PATH.format("requests"))
def test_page_exists_404(mocked_requests):
    mocked_requests.head().status_code = 404
    mocked_requests.head().raise_for_status.side_effect = Exception
    page_exists("blah ", "blah blah") == False


@mock.patch(PATH.format("requests"))
def test_page_exists_anything_else(mocked_requests):
    mocked_requests.head().raise_for_status.side_effect = Exception
    with pytest.raises(Exception):
        page_exists("blah ", "blah blah")


@mock.patch(PATH.format("page_exists"))
def test_find_last_page(mocked_page_exists):
    # (0, 100), (50, 100), (50, 75), (63, 75), (69, 75), (69, 72), (71, 72)
    mocked_page_exists.side_effect = [True, False, True, True, False, True]
    assert find_last_page("blah", 0, 100) == 71


def test_generate_page_numbers():
    assert generate_page_numbers(1, None, 5) == range(1, 5)
    assert generate_page_numbers(1, 2, 5) == range(1, 2)
    assert generate_page_numbers(1, 5, 5) == range(1, 5)
    assert generate_page_numbers(1, 10, 3) == range(1, 3)
    with pytest.raises(ValueError):
        generate_page_numbers(10, 1, 3)
    with pytest.raises(ValueError):
        generate_page_numbers(10, None, 3)
