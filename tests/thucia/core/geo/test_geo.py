from unittest.mock import patch

import pandas as pd
import pytest
from thucia.core.geo import add_incidence_rate
from thucia.core.geo import align_admin2_regions
from thucia.core.geo import ensure_all_regions
from thucia.core.geo import fuzzy_match_one
from thucia.core.geo import remove_accents


@pytest.fixture
def admin2_list():
    return pd.DataFrame(
        {
            "GID_0": ["X"] * 3,
            "GID_1": ["X.1_1"] * 3,
            "NAME_1": ["State"] * 3,
            "GID_2": ["X.1.1_2", "X.1.2_2", "X.1.3_2"],
            "NAME_2": ["A", "B", "C"],
            "HASC_2": ["XA", "XB", "XC"],
        }
    )


def test_remove_accents():
    assert remove_accents("São Paulo") == "Sao Paulo"
    assert remove_accents("München") == "Munchen"


def test_fuzzy_match_one_exact_and_threshold():
    refs = ["São Paulo", "Rio de Janeiro"]
    assert fuzzy_match_one("São Paulo", refs) == "São Paulo"
    assert fuzzy_match_one("Completely Unrelated", refs, threshold=95) == ""


def test_ensure_all_regions_adds_missing_gids(admin2_list):
    df = pd.DataFrame(
        {
            "Date": pd.period_range("2020-01", periods=2, freq="M").repeat(2),
            "GID_1": ["X.1_1"] * 4,
            "GID_2": ["X.1.1_2", "X.1.2_2"] * 2,
            "Cases": [5, 7, 3, 1],
        }
    )
    with patch("thucia.core.geo.get_admin2_list", return_value=admin2_list):
        out = ensure_all_regions(df)
    assert sorted(out.df["GID_2"].unique()) == sorted(admin2_list["GID_2"].tolist())
    # missing region gets zero cases
    assert (out.df[out.df["GID_2"] == "X.1.3_2"]["Cases"] == 0).all()


def test_ensure_all_regions_generic_roster():
    # A non-GADM tagging scheme: codes live under 'region'/'state' and the
    # roster keys them under different names. No GID_* column may appear.
    dates = pd.period_range("2020-01", periods=2, freq="M")
    df = pd.DataFrame(
        {
            "Date": dates,
            "region": ["north", "north"],
            "state": ["stA", "stA"],
            "Cases": [5, 3],
        }
    )
    roster = pd.DataFrame(
        {
            "province": ["north", "south"],
            "payer": ["stA", "stB"],
            "label": ["Northland", "Southland"],
        }
    )
    out = ensure_all_regions(
        df,
        geo_col="region",
        geo_parent="state",
        regions=roster,
        region_col="province",
        parent_col="payer",
    )
    f = out.df
    assert set(f["region"].dropna().astype("object").unique()) == {"north", "south"}
    assert (f[f["region"] == "south"]["Cases"] == 0).all()
    assert (f[f["region"] == "south"]["state"] == "stB").all()
    assert "GID_1" not in f.columns and "GID_2" not in f.columns


def test_ensure_all_regions_categorical_implicit():
    # Categorical geo column: the categories are the implicit region list, so
    # unused categories (never-seen regions) still get padded — .unique() alone
    # would drop them.
    dates = pd.period_range("2020-01", periods=2, freq="M")
    df = pd.DataFrame(
        {
            "Date": dates,
            "region": pd.Categorical(
                ["north", "north"], categories=["north", "south", "west"]
            ),
            "Cases": [5, 3],
        }
    )
    out = ensure_all_regions(df, geo_col="region", geo_parent=None)
    f = out.df
    assert set(f["region"].dropna().astype("object").unique()) == {
        "north",
        "south",
        "west",
    }
    assert (f[f["region"] == "south"]["Cases"] == 0).all()
    assert (f[f["region"] == "west"]["Cases"] == 0).all()


def test_ensure_all_regions_no_roster_raises():
    # Non-categorical, non-GADM codes and no regions= roster: no region list can
    # be resolved, so ensure_all_regions refuses to guess.
    df = pd.DataFrame(
        {
            "Date": pd.period_range("2020-01", periods=2, freq="M"),
            "region": ["north", "south"],
            "Cases": [1, 2],
        }
    )
    with pytest.raises(ValueError, match="regions="):
        ensure_all_regions(df, geo_col="region", geo_parent=None)


def test_add_incidence_rate():
    df = pd.DataFrame(
        {
            "Cases": [5, 10],
            "pop_count": [1000, 2000],
            "Date": ["2020-01"] * 2,
            "GID_2": ["A", "B"],
        }
    )
    out = add_incidence_rate(df)
    assert out["DIR"].tolist() == pytest.approx([500.0, 500.0])


def test_align_admin2_regions_exact_match(admin2_list):
    df = pd.DataFrame(
        {
            "ADM1": ["State", "State"],
            "ADM2": ["A", "B"],
            "Cases": [1, 2],
            "Date": ["2020-01-01"] * 2,
        }
    )
    with patch("thucia.core.geo.get_admin2_list", return_value=admin2_list):
        out = align_admin2_regions(df, iso3="X")
    assert {"GID_1", "GID_2"} <= set(out.columns)
    assert set(out["GID_2"]) == {"X.1.1_2", "X.1.2_2"}


def test_align_admin2_regions_fuzzy_match(admin2_list):
    # Accent-mangled name should still match via homogenisation
    df = pd.DataFrame(
        {"ADM1": ["State"], "ADM2": ["A"], "Cases": [1], "Date": ["2020-01-01"]}
    )
    with patch("thucia.core.geo.get_admin2_list", return_value=admin2_list):
        out = align_admin2_regions(df, iso3="X")
    assert out["GID_2"].tolist() == ["X.1.1_2"]


def test_merge_sources_calls_plugin(admin2_list):
    from thucia.core.geo import merge_sources
    from thucia.core.registry import Registry

    class FakePlugin:
        name = "fake"
        ref = "fake"

        def merge(self, df, metrics):
            df = df.copy()
            df["fake_col"] = 42.0
            return df

    fake_registry = Registry("covariate source")
    fake_registry.register()(FakePlugin)
    with patch("thucia.core.geo.source_registry", fake_registry):
        df = pd.DataFrame(
            {
                "Date": pd.period_range("2020-01", periods=2, freq="M"),
                "GID_2": ["A", "B"],
                "GID_1": ["G1", "G1"],
                "Cases": [1, 2],
            }
        )
        out = merge_sources(df, ["fake.metric"])
    assert "fake_col" in out.columns
    assert (out["fake_col"] == 42.0).all()
