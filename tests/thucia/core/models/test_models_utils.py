import numpy as np
import pandas as pd
import pytest
from thucia.core.models.utils import filter_admin1
from thucia.core.models.utils import interpolate_missing_dates
from thucia.core.models.utils import pca_transform
from thucia.core.models.utils import quantiles
from thucia.core.models.utils import sample_to_quantiles_vec
from thucia.core.models.utils import samples_to_quantiles
from thucia.core.models.utils import sanitise_covariates
from thucia.core.models.utils import season_length_for_freq
from thucia.core.models.utils import set_historical_na_to_zero
from thucia.core.models.utils import validate_unique_keys


def test_season_length_for_freq():
    assert season_length_for_freq("M") == 12
    assert season_length_for_freq("ME") == 12
    assert season_length_for_freq("W-SAT") == 52
    assert season_length_for_freq("W-SUN") == 52
    assert season_length_for_freq("W-MON") == 52
    assert season_length_for_freq("D") == 365
    assert season_length_for_freq(None) == 12
    assert season_length_for_freq("Q") == 12  # unknown -> monthly default


def _monthly_df(n=6, n_gid=2, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.period_range("2020-01", periods=n, freq="M")
    df = pd.DataFrame(
        {
            "Date": np.repeat(dates, n_gid),
            "GID_2": list("AB") * n,
            "GID_1": ["G1"] * (n * n_gid),
            "future": [False] * (n * n_gid),
            "Cases": rng.uniform(0, 100, n * n_gid),
        }
    )
    df["Log_Cases"] = np.log1p(df["Cases"])
    return df


def test_sanitise_covariates_period_start_date():
    df = _monthly_df()
    df["tmax"] = [10.0, np.nan, 12.0, np.nan, np.nan, 11.0] + [10.0] * 6
    df["prec"] = [100.0] * 6 + [np.nan] * 6
    out = sanitise_covariates(df, ["tmax", "prec"], pd.Period("2020-01", freq="M"))
    assert out[["tmax", "prec"]].isna().sum().sum() == 0


def test_sanitise_covariates_returns_same_rows():
    df = _monthly_df()
    df["tmax"] = np.arange(12.0)
    out = sanitise_covariates(df, ["tmax"], pd.Period("2020-01", freq="M"))
    assert len(out) == len(df)
    assert (out["GID_2"] == df["GID_2"]).all()


def test_sample_to_quantiles_vec_bounds():
    rng = np.random.default_rng(0)
    s = rng.normal(0, 1, 1000)
    out = sample_to_quantiles_vec(s, quantiles)
    assert len(out) == len(quantiles)
    # empirical quantiles must be increasing
    assert (out["value"].diff().dropna() >= 0).all()


def test_samples_to_quantiles_monotone():
    rng = np.random.default_rng(0)
    dates = pd.period_range("2020-01", periods=5, freq="M")
    rows = [
        (d, g, rng.normal(5, 1), 2.0)
        for d in dates
        for g in ["A", "B"]
        for _ in range(50)
    ]
    df = pd.DataFrame(rows, columns=["Date", "GID_2", "prediction", "Cases"])
    out = samples_to_quantiles(df)
    q = out[(out["GID_2"] == "A") & (out["Date"] == dates[3])].sort_values("quantile")
    assert (q["prediction"].diff().dropna() >= 0).all()
    assert sorted(out["quantile"].unique()) == sorted(quantiles)


def test_filter_admin1():
    df = _monthly_df()
    out = filter_admin1(df, ["G1"])
    assert len(out) == len(df)
    assert len(filter_admin1(df, ["OTHER"])) == 0


def test_interpolate_missing_dates():
    base = _monthly_df(n=4)
    df = base.drop(base.index[[2, 5]])
    out = interpolate_missing_dates(df, geo_col="GID_2")
    for g in ["A", "B"]:
        assert out[out["GID_2"] == g]["Date"].is_monotonic_increasing
    # all (GID_2, Date) combos present
    assert len(out) == 4 * 2


def test_set_historical_na_to_zero():
    df = _monthly_df()
    df.loc[1, "Cases"] = np.nan
    df.loc[1, "future"] = False
    out = set_historical_na_to_zero(df)
    assert out.loc[1, "Cases"] == 0.0


def test_validate_unique_keys():
    df = pd.DataFrame({"Date": [1, 1], "GID_2": ["A", "A"]})
    with pytest.raises(ValueError, match="duplicate"):
        validate_unique_keys(df, ["Date", "GID_2"])


def test_pca_transform():
    rng = np.random.default_rng(0)
    df = _monthly_df(n=10)
    for i in range(5):
        df[f"c{i}"] = rng.normal(size=len(df))
    out = pca_transform(df, ["c0", "c1", "c2", "c3", "c4"], keep_components=3)
    assert [c for c in out.columns if c.startswith("PC")] == ["PC1", "PC2", "PC3"]
    assert out[["PC1", "PC2", "PC3"]].isna().sum().sum() == 0
