# End-to-end forecast tests: run_model on small synthetic data for every
# stable model, asserting the canonical quantile grid, finite/non-negative
# predictions, and scoreability. Fits are kept small (short trailing window)
# so the whole suite stays fast.
import numpy as np
import pandas as pd
import pytest
from forecast_data import COVARIATE_COLS
from forecast_data import make_forecast_df
from thucia.core.cases import r2
from thucia.core.cases.wis import wis_bracher
from thucia.core.models import baseline
from thucia.core.models import movavg
from thucia.core.models import nhits
from thucia.core.models import run_model
from thucia.core.models import sarima
from thucia.core.models import tcn
from thucia.core.models import tft
from thucia.core.models import xgboost
from thucia.core.quantiles import quantiles

START = pd.Period("2020-07", freq="M")


@pytest.fixture(scope="module")
def df():
    return make_forecast_df()


def _run(model, df, tmp_path, **extra):
    kwargs = {
        "start_date": START,
        "geo_col": "GID_2",
        "geo_parent": "GID_1",
        "geo_parent_filter": None,
        "horizons": [1],
        "case_col": "Log_Cases",
        "covariate_cols": COVARIATE_COLS,
        "train_col": "GID_2",
        "db_file": None,
    }
    kwargs.update(extra)
    out = run_model(
        "e2e", model, df, tmp_path, save_quantiles=True, model_kwargs=kwargs
    )
    return out.df if hasattr(out, "df") else out


# (label, model, extra kwargs, assertion flags)
MODELS = [
    ("baseline", baseline, {"horizon": 1, "num_samples": 50}, {"nan_ok": True}),
    ("movavg", movavg, {}, {"require_horizon": False}),
    ("sarima_retrain_false", sarima, {"retrain": False, "num_samples": 20}, {}),
    ("sarima_retrain_true", sarima, {"retrain": True, "num_samples": 20}, {}),
    (
        "xgboost",
        xgboost,
        {"retrain": False, "num_samples": 20, "multivariate": False},
        {"monotone": False},
    ),
    (
        "tcn",
        tcn,
        {"retrain": False, "num_samples": 20, "multivariate": False},
        {"monotone": False},
    ),
    (
        "nhits",
        nhits,
        {"retrain": False, "num_samples": 20, "multivariate": False},
        {"monotone": False},
    ),
    ("tft", tft, {"retrain": False, "num_samples": 20, "multivariate": False}, {}),
]


def _assert_valid_forecast(
    out, geo_col="GID_2", require_horizon=True, nan_ok=False, monotone=True
):
    cols = {geo_col, "Date", "quantile", "prediction", "Cases"}
    if require_horizon:
        cols.add("horizon")
    assert cols <= set(out.columns)
    assert sorted(out["quantile"].unique()) == quantiles
    valid = out["prediction"].notna()
    assert valid.any()
    assert np.isfinite(out["prediction"][valid]).all()
    assert (out["prediction"][valid] >= 0).all()
    if monotone:
        grouped = (
            out[valid]
            .sort_values([geo_col, "Date", "quantile"])
            .groupby([geo_col, "Date"], observed=False)["prediction"]
        )
        assert (grouped.transform(lambda s: s.diff().dropna() >= -1e-9)).all()


@pytest.mark.parametrize("label,model,extra,flags", MODELS, ids=[m[0] for m in MODELS])
def test_forecast_valid_and_scoreable(label, model, extra, flags, df, tmp_path):
    out = _run(model, df, tmp_path, **extra)
    _assert_valid_forecast(out, **flags)
    scored = out[out["prediction"].notna()]
    wis = wis_bracher(
        scored[["GID_2", "Date", "quantile", "prediction", "Cases"]],
        group_cols=("GID_2", "Date"),
    )
    assert np.isfinite(wis["WIS"]).all()
    r2val = r2(scored, "prediction", "Cases", group_col="GID_2")
    assert np.isfinite(r2val["R2"]).all()


def test_forecast_weekly_baseline(tmp_path):
    # Weekly (W-SAT) cadence end-to-end: the anchor must survive a real fit and
    # the canonical quantile grid must be produced.
    df = make_forecast_df(freq="W-SAT", n_periods=160, start="2017-01-07")
    out = _run(
        baseline,
        df,
        tmp_path,
        horizon=1,
        num_samples=20,
        start_date=pd.Period("2020-01-04", freq="W-SAT"),
    )
    assert str(out["Date"].dtype) == str(df["Date"].dtype)  # period[W-SAT] kept
    assert sorted(out["quantile"].unique()) == quantiles
    valid = out["prediction"][out["prediction"].notna()]
    assert (valid >= 0).all()
    scored = out[out["prediction"].notna()]
    wis = wis_bracher(
        scored[["GID_2", "Date", "quantile", "prediction", "Cases"]],
        group_cols=("GID_2", "Date"),
    )
    assert np.isfinite(wis["WIS"]).all()
