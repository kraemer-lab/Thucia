import numpy as np
import pandas as pd
import pytest
from darts import TimeSeries
from forecast_data import make_forecast_df
from thucia.core.models.darts import DartsBase
from thucia.core.quantiles import quantiles


class MockDarts(DartsBase):
    """DartsBase driven by deterministic pseudo-forecasts (no training)."""

    def __init__(self, *args, output="quantiles", **kwargs):
        self.output = output
        super().__init__(*args, **kwargs)

    def build_model(self, horizon=None):
        return None

    def pre_fit(self, target_gids=None, **kwargs):
        pass

    def historical_forecasts(
        self, ts, cov, start_date=None, retrain=True, horizon=1, **kwargs
    ):
        if isinstance(ts, list):
            return [self._make(t, start_date) for t in ts]
        return self._make(ts, start_date)

    def _make(self, ts, start_date):
        times = ts.time_index
        selected = [
            t
            for t in times
            if t >= start_date and t >= times[0] + pd.Timedelta(days=365)
        ]
        idx = pd.DatetimeIndex(selected)
        base = float(np.asarray(ts.values()).reshape(-1)[-1])
        freq = getattr(ts.freq, "freqstr", ts.freq)
        if self.output == "quantiles":
            df = pd.DataFrame(
                {f"Log_Cases_q{q:0.3f}": base + q for q in [0.025, 0.5, 0.975]},
                index=idx,
            )
            return TimeSeries.from_dataframe(df, fill_missing_dates=True, freq=freq)
        rng = np.random.default_rng(0)
        arr = base + rng.normal(0, 1, (len(idx), 1, 100))
        return TimeSeries.from_times_and_values(
            idx, arr, fill_missing_dates=True, freq=freq
        )


@pytest.fixture
def df():
    return make_forecast_df()


def _as_df(out):
    return out.df if hasattr(out, "df") else out


def test_quantile_output_schema_and_parsing(df):
    m = MockDarts(
        df=df,
        case_col="Log_Cases",
        geo_col="GID_2",
        horizons=[1],
        covariate_cols=["tmin", "prec"],
    )
    out = _as_df(
        m.historical_predictions(
            start_date=pd.Period("2020-01", freq="M"), train_col="GID_2"
        )
    )
    assert {
        "Date",
        "GID_2",
        "horizon",
        "quantile",
        "prediction",
        "Log_Cases",
        "Cases",
    } <= set(out.columns)
    # q0.025/0.500/0.975 parsed correctly from the column names
    assert sorted(out["quantile"].unique()) == pytest.approx([0.025, 0.5, 0.975])
    # predictions on the count scale (expm1 of the log-scale pseudo-forecasts)
    assert (out["prediction"] >= 0).all()
    assert np.isfinite(out["prediction"]).all()


def test_samples_output_converted_to_canonical_quantiles(df):
    m = MockDarts(
        df=df,
        case_col="Log_Cases",
        geo_col="GID_2",
        horizons=[1],
        covariate_cols=["tmin", "prec"],
        output="samples",
    )
    out = _as_df(
        m.historical_predictions(
            start_date=pd.Period("2020-01", freq="M"), train_col="GID_2"
        )
    )
    assert sorted(out["quantile"].unique()) == pytest.approx(quantiles)
    q = out.sort_values(["GID_2", "Date", "quantile"])
    groups = q.groupby(["GID_2", "Date"], observed=False)["prediction"]
    assert (groups.transform(lambda s: s.diff().dropna() >= -1e-9)).all()
    assert (out["prediction"] >= 0).all()


def test_horizon_labels(df):
    m = MockDarts(
        df=df,
        case_col="Log_Cases",
        geo_col="GID_2",
        horizons=[1, 3],
        covariate_cols=["tmin", "prec"],
    )
    out = _as_df(
        m.historical_predictions(
            start_date=pd.Period("2020-01", freq="M"), train_col="GID_2"
        )
    )
    assert sorted(out["horizon"].unique()) == [1, 3]


def test_admin_level_2_groups_by_gid2(df):
    m = MockDarts(
        df=df,
        case_col="Log_Cases",
        geo_col="GID_2",
        horizons=[1],
        covariate_cols=["tmin", "prec"],
    )
    out = _as_df(
        m.historical_predictions(
            start_date=pd.Period("2020-01", freq="M"), train_col="GID_2"
        )
    )
    assert set(out["GID_2"].unique()) == set(df["GID_2"].unique())


def test_admin_level_1_groups_by_gid1(df):
    # train_col="GID_1" trains per GID_1; output is still keyed by GID_2
    m = MockDarts(
        df=df,
        case_col="Log_Cases",
        geo_col="GID_2",
        horizons=[1],
        covariate_cols=["tmin", "prec"],
    )
    out = _as_df(
        m.historical_predictions(
            start_date=pd.Period("2020-01", freq="M"), train_col="GID_1"
        )
    )
    assert len(out) > 0
    assert set(out["GID_2"].unique()) == set(df["GID_2"].unique())


def test_admin_level_0_country(df):
    m = MockDarts(
        df=df,
        case_col="Log_Cases",
        geo_col="GID_2",
        horizons=[1],
        covariate_cols=["tmin", "prec"],
    )
    out = _as_df(
        m.historical_predictions(
            start_date=pd.Period("2020-01", freq="M"), train_col=None
        )
    )
    assert len(out) > 0


def test_rejected_zero_incidence_regions_skipped(df):
    # A region with zero cases across the training period is rejected by
    # identify_noincidence_regions(); the per-region fit must skip it instead
    # of crashing on an empty target series list (regression: darts
    # series2seq raised IndexError when fit() received an empty series).
    zero_gid = df["GID_2"].cat.categories[0]
    df = df.copy()
    mask = df["GID_2"] == zero_gid
    df.loc[mask, ["Cases", "Log_Cases"]] = 0.0
    m = MockDarts(
        df=df,
        case_col="Log_Cases",
        geo_col="GID_2",
        horizons=[1],
        covariate_cols=["tmin", "prec"],
    )
    assert zero_gid in m.rejected_gids
    out = _as_df(
        m.historical_predictions(
            start_date=pd.Period("2020-01", freq="M"), train_col="GID_2"
        )
    )
    assert set(out["GID_2"].unique()) == set(df["GID_2"].unique()) - {zero_gid}


def test_geo_col_coerced_to_categorical():
    df = make_forecast_df()
    df["GID_2"] = df["GID_2"].astype(str)  # undo the categorical coercion
    m = MockDarts(
        df=df,
        case_col="Log_Cases",
        geo_col="GID_2",
        horizons=[1],
        covariate_cols=["tmin", "prec"],
    )
    assert isinstance(m.df["GID_2"].dtype, pd.CategoricalDtype)


def test_multivariate_path(df):
    m = MockDarts(
        df=df, case_col="Log_Cases", geo_col="GID_2", horizons=[1], multivariate=True
    )
    out = _as_df(
        m.historical_predictions(
            start_date=pd.Period("2020-01", freq="M"), train_col="GID_2"
        )
    )
    assert len(out) > 0
    assert np.isfinite(out["prediction"]).all()


@pytest.mark.parametrize(
    "freq,start,forecast_from",
    [
        ("W-SAT", "2017-01-07", pd.Period("2020-01-04", freq="W-SAT")),
        ("W-SUN", "2017-01-01", pd.Period("2020-01-05", freq="W-SUN")),
        ("D", "2017-01-01", pd.Period("2020-01-01", freq="D")),
    ],
)
def test_cadence_anchor_preserved(freq, start, forecast_from):
    # The DartsBase machinery must keep the Period anchor (W-SAT stays W-SAT,
    # daily stays daily) end-to-end -- this guards the old freqstr[0] truncation.
    n = 1100 if freq == "D" else 160
    df = make_forecast_df(freq=freq, n_periods=n, start=start)
    m = MockDarts(
        df=df,
        case_col="Log_Cases",
        geo_col="GID_2",
        horizons=[1],
        covariate_cols=["tmin", "prec"],
    )
    out = _as_df(m.historical_predictions(start_date=forecast_from, train_col="GID_2"))
    assert str(out["Date"].dtype) == str(df["Date"].dtype)
    assert sorted(out["quantile"].unique()) == pytest.approx([0.025, 0.5, 0.975])
    assert np.isfinite(out["prediction"]).all()
