from ojd_daps.flows.pre_enrich.soc_lookup import short_hash


def test_short_hash():
    """Check that nearly similar strings give different and stable hashes"""
    assert short_hash("Foo Bar 123") == 2115373645493407
    assert short_hash("Foo Bar 12") == 1572476239067751
    assert short_hash("foo Bar 123") == 1737237057932100
