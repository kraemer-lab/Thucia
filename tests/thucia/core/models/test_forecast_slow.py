# Heavy real-model forecast fits. Run with `uv run pytest -m slow` (excluded
# from the default/CI run via `addopts = "-m 'not slow'"`).
import numpy as np
import pandas as pd
import pytest
from forecast_data import COVARIATE_COLS
from forecast_data import make_forecast_df
from thucia.core.models import nbeats
from thucia.core.models import sarima
from thucia.core.models import xgboost
from thucia.core.quantiles import quantiles

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def df():
    return make_forecast_df()


def _as_df(out):
    return out.df if hasattr(out, "df") else out


def test_nbeats_forecast(df):
    out = _as_df(
        nbeats(
            df,
            start_date=pd.Period("2020-07", freq="M"),
            train_start_date=pd.Period("2016-01", freq="M"),
            train_end_date=pd.Period("2020-06", freq="M"),
            geo_parent_filter=None,
            horizons=[1],
            case_col="Log_Cases",
            covariate_cols=COVARIATE_COLS,
            retrain=False,
            db_file=None,
            train_col="GID_2",
            num_samples=20,
            multivariate=False,
        )
    )
    assert sorted(out["quantile"].unique()) == quantiles
    assert np.isfinite(out["prediction"]).all()
    assert (out["prediction"] >= 0).all()


def test_xgboost_multihorizon():
    # output_chunk_length=12 (max horizon) needs >= 48 lags + 12 months of
    # training history, hence the longer fixture.
    df = make_forecast_df(n_periods=96, start="2014-01")
    out = _as_df(
        xgboost(
            df,
            start_date=pd.Period("2020-01", freq="M"),
            train_start_date=pd.Period("2014-01", freq="M"),
            train_end_date=pd.Period("2019-12", freq="M"),
            geo_parent_filter=None,
            horizons=[1, 3, 6, 12],
            case_col="Log_Cases",
            covariate_cols=COVARIATE_COLS,
            retrain=False,
            db_file=None,
            train_col="GID_2",
            num_samples=20,
            multivariate=False,
        )
    )
    assert sorted(out["horizon"].unique()) == [1, 3, 6, 12]
    assert sorted(out["quantile"].unique()) == quantiles
    assert np.isfinite(out["prediction"]).all()


@pytest.mark.network
def test_timesfm_forecast_requires_torch_and_network(df):
    # TimesFM downloads a HuggingFace checkpoint; opt-in only.
    pytest.importorskip("torch")
    try:
        from thucia.core.models import timesfm
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"timesfm unavailable: {exc}")
    out = _as_df(
        timesfm(
            df,
            start_date=pd.Period("2020-07", freq="M"),
            geo_parent_filter=None,
            horizons=[1],
            case_col="Log_Cases",
            covariate_cols=COVARIATE_COLS,
            retrain=False,
            db_file=None,
        )
    )
    assert np.isfinite(out["prediction"]).all()


def test_sarima_weekly_seasonal():
    # SARIMA on a weekly cadence with an explicit 52-period seasonal length;
    # the weekly anchor must be preserved.
    df = make_forecast_df(freq="W-SAT", n_periods=208, start="2016-01-02", n_gid=1)
    out = _as_df(
        sarima(
            df,
            start_date=pd.Period("2019-12-07", freq="W-SAT"),
            geo_parent_filter=None,
            horizons=[1],
            case_col="Log_Cases",
            covariate_cols=COVARIATE_COLS,
            retrain=False,
            db_file=None,
            train_col="GID_2",
            num_samples=20,
            season_length=52,
        )
    )
    assert str(out["Date"].dtype) == "period[W-SAT]"
    assert sorted(out["quantile"].unique()) == quantiles
    assert np.isfinite(out["prediction"]).all()
