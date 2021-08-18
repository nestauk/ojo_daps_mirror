import pytest

from ojd_daps.flows.pre_enrich.common import flatten, get_chunks


def test_flatten():
    assert flatten([["a", "b"], ["c", "d"]]) == ["a", "b", "c", "d"]
    assert flatten([["e", "f", "g"]]) == ["e", "f", "g"]


def test_get_chunks():
    assert get_chunks(["a", "b"], 1) == [["a"], ["b"]]
    assert get_chunks(["a", "b"], 2) == [["a", "b"]]
