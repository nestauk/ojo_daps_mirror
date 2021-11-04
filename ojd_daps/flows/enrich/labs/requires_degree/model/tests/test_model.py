"""tests for requires_degree regex model."""
from unittest import mock
from ojd_daps.flows.enrich.labs.requires_degree import model


def test_model_hits():
    regex = model.EXPRESSION
    requires_degree = model.nlp.regex_model(regex)
    assert requires_degree("qualifications: educated to phd level")
    assert requires_degree("candidate should have a bachelor's degree")
    assert requires_degree("masters degree in educational")


def test_model_correct_rejections():
    regex = model.EXPRESSION
    requires_degree = model.nlp.regex_model(regex)
    assert not requires_degree("no degree qualification required")
    assert not requires_degree("lorem ipsum")
    assert not requires_degree("bam")
    assert not requires_degree("mscaa")
