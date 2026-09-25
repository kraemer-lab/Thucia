# Probing tests for the WorldPop covariate source (geo/sources/worldpop.py).
# No network: rasters are faked via raster_stats_gid2 mocks.
import pandas as pd
import pytest
from thucia.core.geo.sources.worldpop import resolve_country_key


def _polygons(country=None):
    cols = {"region": ["north", "south"]}
    if country:
        cols["COUNTRY"] = [country] * 2
    return pd.DataFrame({**cols, "geometry": [None, None]})


def test_resolve_country_key_explicit_iso3_wins():
    assert resolve_country_key("BRA", _polygons(country="ARG"), ["a", "b"]) == "BRA"


def test_resolve_country_key_from_regions_map():
    assert resolve_country_key(None, _polygons(country="ARG"), ["a", "b"]) == "ARG"


def test_resolve_country_key_from_gadm_prefix():
    assert resolve_country_key(None, None, ["BRA.1.1_2", "BRA.1.2_2"]) == "BRA"


def test_resolve_country_key_no_regions_map_uses_gadm_prefix():
    # The historical behaviour: derive the country from the code prefix.
    assert resolve_country_key(None, None, ["BRA.1.1_2"]) == "BRA"


def test_resolve_country_key_ambiguous_map_raises():
    polys = pd.DataFrame(
        {"region": ["a", "b"], "COUNTRY": ["ARG", "BRA"], "geometry": [None, None]}
    )
    with pytest.raises(ValueError, match="single country"):
        resolve_country_key(None, polys, ["a", "b"])


def test_resolve_country_key_no_key_raises():
    with pytest.raises(ValueError, match="single-country key"):
        resolve_country_key(None, None, ["north", "south"])


def test_worldpop_merge_generic_geo_col(tmp_path, monkeypatch):
    from thucia.core.geo.sources.worldpop import WorldPop
    import thucia.core.geo.sources.worldpop as worldpop

    monkeypatch.setattr(worldpop, "cache_folder", str(tmp_path))

    wp = WorldPop()
    monkeypatch.setattr(wp, "get_filename", lambda metric, gid_1, year: "fake.tif")

    seen = {}

    def fake_stats(tif, codes, stats=None, geo_col="GID_2", iso3=None, polygons=None):
        seen["geo_col"] = geo_col
        seen["iso3"] = iso3
        seen["polygons"] = polygons
        return pd.DataFrame({geo_col: codes, "sum": [100.0] * len(codes)})

    monkeypatch.setattr(worldpop, "raster_stats_gid2", lambda *a, **k: None)
    monkeypatch.setattr(wp, "get_cached_stats", fake_stats)
    monkeypatch.setattr(
        wp, "_get_cached_stats_gadm", lambda *a, **k: pytest.fail("gadm path")
    )

    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2020-01-31", "2020-01-31"]),
            "region": ["north", "south"],
            "Cases": [5.0, 7.0],
        }
    )
    regions = _polygons(country="ARG")
    out = wp.merge(df, metrics=["pop_count"], geo_col="region", polygons=regions)

    assert "pop_count" in out.columns
    assert out["pop_count"].tolist() == [100.0, 100.0]
    assert "GID_2" not in out.columns
    assert seen["geo_col"] == "region"
    assert seen["iso3"] is None
    assert seen["polygons"] is regions
