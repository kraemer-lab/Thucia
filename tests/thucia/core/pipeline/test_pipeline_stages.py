# Probe tests for the thucia.core.pipeline stages. Geo lookups are mocked and
# model fits use the fast baseline on small synthetic data.
import numpy as np
import pandas as pd
import pytest
from thucia.core.pipeline import aggregate_quantiles
from thucia.core.pipeline import apply_residual_regression
from thucia.core.pipeline import build_ensemble
from thucia.core.pipeline import cases_per_period
from thucia.core.pipeline import fit_model
from thucia.core.pipeline import merge_covariates
from thucia.core.pipeline import PipelineConfig
from thucia.core.pipeline import prepare_model_inputs
from thucia.core.pipeline import score_model


@pytest.fixture
def case_df():
    # line-list style input: string dates, no pop_count
    dates = pd.date_range("2020-01-31", periods=6, freq="ME").strftime("%Y-%m-%d")
    return pd.DataFrame(
        {
            "Date": np.repeat(dates, 2),
            "GID_1": ["G.1_1"] * 12,
            "GID_2": ["G.1.1_2", "G.1.2_2"] * 6,
            "Cases": np.arange(12.0),
        }
    )


@pytest.fixture
def admin2_list():
    return pd.DataFrame(
        {
            "GID_1": ["G.1_1"] * 3,
            "GID_2": ["G.1.1_2", "G.1.2_2", "G.1.3_2"],
            "NAME_1": ["State"] * 3,
            "NAME_2": ["A", "B", "C"],
        }
    )


def test_cases_per_period_pads_and_adds_future(case_df, admin2_list, monkeypatch):
    monkeypatch.setattr("thucia.core.geo.get_admin2_list", lambda iso3: admin2_list)
    cfg = PipelineConfig(path=".")
    out = cases_per_period(case_df, cfg, freq="M")
    # every admin-2 present, plus `future_periods` future rows per region
    assert set(out["GID_2"].unique()) == set(admin2_list["GID_2"])
    n_dates = out["Date"].nunique()
    assert n_dates == 6 + cfg.future_periods
    future = out[out["future"]]
    assert future["Cases"].isna().all()
    assert len(future) == len(admin2_list) * cfg.future_periods
    # historical case total preserved
    assert out[~out["future"]]["Cases"].sum() == case_df["Cases"].sum()


def test_cases_per_period_cutoff_date(case_df, admin2_list, monkeypatch):
    monkeypatch.setattr("thucia.core.geo.get_admin2_list", lambda iso3: admin2_list)
    cfg = PipelineConfig(path=".", cutoff_date="2020-03")
    out = cases_per_period(case_df, cfg, freq="M")
    assert (out[~out["future"]]["Date"] <= pd.Period("2020-03", "M")).all()


@pytest.mark.parametrize(
    "freq,expected_period", [("W-SAT", "period[W-SAT]"), ("D", "period[D]")]
)
def test_cases_per_period_cadence(
    case_df, admin2_list, monkeypatch, freq, expected_period
):
    monkeypatch.setattr("thucia.core.geo.get_admin2_list", lambda iso3: admin2_list)
    cfg = PipelineConfig(path=".")
    out = cases_per_period(case_df, cfg, freq=freq)
    assert str(out["Date"].dtype) == expected_period
    assert out["future"].sum() == len(admin2_list) * cfg.future_periods
    assert out[~out["future"]]["Cases"].sum() == pytest.approx(case_df["Cases"].sum())


def test_merge_covariates(case_df, monkeypatch):
    from thucia.core.registry import Registry

    class FakePlugin:
        name = "fake"
        ref = "fake"

        def merge(self, df, metrics):
            out = df.copy()
            out["fake_col"] = 42.0
            return out

    fake_registry = Registry("covariate source")
    fake_registry.register()(FakePlugin)
    monkeypatch.setattr("thucia.core.geo.source_registry", fake_registry)
    cfg = PipelineConfig(path=".", source_specs=["fake.metric"])
    df = case_df.copy()
    df["pop_count"] = 1000.0
    out = merge_covariates(df, cfg)
    assert "fake_col" in out.columns
    assert "DIR" in out.columns  # incidence rate
    assert (out["fake_col"] == 42.0).all()
    assert out["DIR"].iloc[0] == pytest.approx(1e5 * out["Cases"].iloc[0] / 1000.0)


def test_prepare_model_inputs_with_lag_spec():
    rng = np.random.default_rng(0)
    idx = pd.period_range("2016-01", periods=12, freq="M")
    df = pd.DataFrame(
        {
            "Date": idx.repeat(2),
            "GID_1": ["G.1_1"] * 24,
            "GID_2": ["G.1.1_2", "G.1.2_2"] * 12,
            "future": [False] * 24,
            "Cases": rng.uniform(1, 100, 24),
            "tmax": rng.uniform(20, 30, 24),
            "prec": rng.uniform(50, 200, 24),
        }
    )
    cfg = PipelineConfig(
        path=".",
        train_end_date=pd.Period("2018-01", freq="M"),
        lag_spec=[
            {
                "name": "tmax_lag_1",
                "groupby": ["GID_2"],
                "column": "tmax",
                "pipeline": [{"op": "shift", "periods": 1}],
            },
            {
                "name": "log_cases_lag_1",
                "groupby": ["GID_2"],
                "column": "Log_Cases",
                "pipeline": [{"op": "shift", "periods": 1}],
            },
        ],
    )
    out, cov_cols = prepare_model_inputs(df, cfg)
    assert {"Log_Cases", "tmax_lag_1", "log_cases_lag_1"} <= set(out.columns)
    assert cov_cols == ["tmax_lag_1", "log_cases_lag_1"]
    assert out[cov_cols].isna().sum().sum() == 0  # sanitised
    assert out["Log_Cases"].iloc[0] == pytest.approx(np.log1p(out["Cases"].iloc[0]))


def test_prepare_model_inputs_implicit_covariates():
    rng = np.random.default_rng(1)
    idx = pd.period_range("2016-01", periods=6, freq="M")
    df = pd.DataFrame(
        {
            "Date": idx.repeat(2),
            "GID_1": ["G.1_1"] * 12,
            "GID_2": ["G.1.1_2", "G.1.2_2"] * 6,
            "future": [False] * 12,
            "Cases": rng.uniform(1, 50, 12),
            "tmin": rng.uniform(10, 20, 12),
        }
    )
    out, cov_cols = prepare_model_inputs(df, PipelineConfig(path="."))
    assert cov_cols == ["tmin"]
    assert out["tmin"].notna().all()


def test_fit_model_baseline(tmp_path):
    idx = pd.period_range("2016-01", periods=36, freq="M")
    df = pd.DataFrame(
        {
            "Date": idx.repeat(2),
            "GID_1": ["G.1_1"] * 72,
            "GID_2": ["G.1.1_2", "G.1.2_2"] * 36,
            "future": [False] * 72,
            "Cases": np.tile(np.arange(36.0), 2),
            "Log_Cases": np.log1p(np.tile(np.arange(36.0), 2)),
        }
    )
    # categorical geo columns with the full category set (as DuckDB provides),
    # so the model's per-region appends share consistent ENUM codes
    df["GID_1"] = df["GID_1"].astype("category")
    df["GID_2"] = df["GID_2"].astype("category")
    cfg = PipelineConfig(
        path=tmp_path,
        start_date=pd.Period("2018-01", freq="M"),
        horizons=[1],
        num_samples=50,
    )
    out = fit_model(df, "baseline", cfg, db_file=None)
    frame = out.df if hasattr(out, "df") else out
    assert "quantile" in frame.columns
    assert "prediction" in frame.columns
    assert frame["prediction"].notna().any()


def test_fit_model_unknown_raises(tmp_path):
    with pytest.raises(ValueError, match="Unknown model"):
        fit_model(pd.DataFrame(), "not_a_model", PipelineConfig(path=tmp_path))


def test_fit_model_coerces_integer_geo_codes(tmp_path):
    # geo codes arrive as integers (e.g. from a non-GADM loader): the pipeline
    # must coerce them to str so ENUM writes and per-region group-bys stay
    # consistent, not uint codes.
    idx = pd.period_range("2016-01", periods=36, freq="M")
    df = pd.DataFrame(
        {
            "Date": idx.repeat(2),
            "GID_2": np.tile([1, 2], 36),
            "future": [False] * 72,
            "Cases": np.tile(np.arange(36.0), 2),
        }
    )
    cfg = PipelineConfig(
        path=tmp_path,
        start_date=pd.Period("2018-01", freq="M"),
        horizons=[1],
        num_samples=50,
    )
    inputs, _ = prepare_model_inputs(df, cfg)
    assert inputs["GID_2"].dtype == object
    assert sorted(inputs["GID_2"].unique()) == ["1", "2"]
    out = fit_model(inputs, "baseline", cfg, db_file=None)
    frame = out.df if hasattr(out, "df") else out
    assert sorted({str(g) for g in frame["GID_2"].unique()}) == ["1", "2"]


def test_fit_model_applies_geo_parent_filter(tmp_path):
    # config.adm1 becomes the geo_parent_filter: baseline must only fit the
    # regions whose geo_parent is in the allowlist.
    idx = pd.period_range("2016-01", periods=36, freq="M")
    df = pd.DataFrame(
        {
            "Date": idx.repeat(4),
            "GID_1": ["G.1_1", "G.1_1", "G.2_1", "G.2_1"] * 36,
            "GID_2": ["G.1.1_2", "G.1.2_2", "G.2.1_2", "G.2.2_2"] * 36,
            "future": [False] * 144,
            "Cases": np.tile(np.arange(36.0), 4),
        }
    )
    for col in ("GID_1", "GID_2"):
        df[col] = df[col].astype("category")
    cfg = PipelineConfig(
        path=tmp_path,
        start_date=pd.Period("2018-01", freq="M"),
        horizons=[1],
        num_samples=50,
        adm1=["G.1_1"],
    )
    inputs, _ = prepare_model_inputs(df, cfg)
    out = fit_model(inputs, "baseline", cfg, db_file=None)
    frame = out.df if hasattr(out, "df") else out
    assert set(frame["GID_2"].unique()) == {"G.1.1_2", "G.1.2_2"}


def test_score_model():
    dates = pd.period_range("2020-01", periods=3, freq="M")
    rows = []
    for h in [1, 3]:
        for d in dates:
            for q in [0.05, 0.5, 0.95]:
                rows.append(
                    {
                        "GID_2": "G",
                        "Date": d,
                        "horizon": h,
                        "quantile": q,
                        "prediction": 5.0 * q + d.month,
                        "Cases": 5.0,
                    }
                )
    df = pd.DataFrame(rows)
    scored = score_model(df, PipelineConfig(path=".", horizons=[1, 3]))
    assert {"WIS", "R2", "horizon"} <= set(scored.columns)
    assert set(scored["horizon"].unique()) == {1, 3}
    assert np.isfinite(scored["WIS"]).all()


def test_aggregate_quantiles(tmp_path):
    dates = pd.period_range("2020-01", periods=2, freq="M")
    rows = []
    for g2, g1 in [("G.1.1_2", "G.1_1"), ("G.1.2_2", "G.1_1")]:
        for d in dates:
            for q in [0.05, 0.5, 0.95]:
                rows.append(
                    {
                        "Date": d,
                        "GID_2": g2,
                        "GID_1": g1,
                        "horizon": 1,
                        "quantile": q,
                        "prediction": 1.0 + q,
                        "Cases": 1.0,
                    }
                )
    df = pd.DataFrame(rows)
    cfg = PipelineConfig(path=tmp_path, horizons=[1])
    out = aggregate_quantiles(df, cfg, geo_parent="GID_1", geo_col="GID_2", samples=200)
    frame = out.df
    assert set(frame["GID_1"].unique()) == {"G.1_1"}
    assert frame["prediction"].notna().all()


def test_build_ensemble():
    dates = pd.period_range("2020-01", periods=4, freq="M")
    rows = []
    for d in dates:
        for q in [0.05, 0.5, 0.95]:
            rows.append(
                {"Date": d, "GID_2": "G", "quantile": q, "prediction": q, "Cases": 0.5}
            )
    m1, m2 = pd.DataFrame(rows), pd.DataFrame(rows)
    cfg = PipelineConfig(path=".")
    ens, weights = build_ensemble([m1, m2], cfg, model_names=["m1", "m2"])
    assert {"m1", "m2"} <= set(weights.columns)
    assert "model" in ens.columns


def test_apply_residual_regression():
    dates = pd.period_range("2020-01", periods=6, freq="M")
    rows = []
    for d in dates:
        for q in [0.5]:
            rows.append(
                {
                    "Date": d,
                    "GID_2": "G",
                    "horizon": 1,
                    "quantile": q,
                    "prediction": 5.0,
                    "Cases": 6.0,
                }
            )
    df = pd.DataFrame(rows)
    emb = pd.DataFrame({"GID_2": ["G"], "feature0": [1.0], "feature1": [0.0]})
    out = apply_residual_regression(
        df, emb, PipelineConfig(path=".", horizons=[1]), method="ridge", geo_col="GID_2"
    )
    assert len(out) == len(df)
    assert out["prediction"].notna().all()
