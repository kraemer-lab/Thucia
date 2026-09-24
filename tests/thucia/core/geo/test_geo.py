from unittest.mock import patch

import pandas as pd
import pytest
from thucia.core.geo import add_incidence_rate
from thucia.core.geo import align_admin2_regions
from thucia.core.geo import fuzzy_match_one
from thucia.core.geo import pad_admin2
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


def test_pad_admin2_adds_missing_gids(admin2_list):
    df = pd.DataFrame(
        {
            "Date": pd.period_range("2020-01", periods=2, freq="M").repeat(2),
            "GID_1": ["X.1_1"] * 4,
            "GID_2": ["X.1.1_2", "X.1.2_2"] * 2,
            "Cases": [5, 7, 3, 1],
        }
    )
    with patch("thucia.core.geo.get_admin2_list", return_value=admin2_list):
        out = pad_admin2(df)
    assert sorted(out.df["GID_2"].unique()) == sorted(admin2_list["GID_2"].tolist())
    # missing region gets zero cases
    assert (out.df[out.df["GID_2"] == "X.1.3_2"]["Cases"] == 0).all()


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
