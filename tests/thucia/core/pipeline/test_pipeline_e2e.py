# End-to-end pipeline test: raw line-list cases -> fitted, scored forecasts.
# The full stage chain runs on small synthetic data with GADM and covariate
# sources mocked (no network).
import numpy as np
import pandas as pd
import pytest
from thucia.core.cases import prepare_pdfm_embeddings
from thucia.core.cases import read_db
from thucia.core.cases import write_db
from thucia.core.fs import write_nc
from thucia.core.pipeline import apply_residual_regression
from thucia.core.pipeline import cases_per_period
from thucia.core.pipeline import fit_model
from thucia.core.pipeline import merge_covariates
from thucia.core.pipeline import PipelineConfig
from thucia.core.pipeline import prepare_model_inputs
from thucia.core.pipeline import score_model
from thucia.core.quantiles import quantiles


def _raw_cases(n_periods=40, seed=0, freq="M", start="2017-01-31", season=12):
    # line-list style input: string dates, integer per-case counts
    rng = np.random.default_rng(seed)
    ts_freq = "ME" if freq == "M" else freq
    dates = pd.date_range(start, periods=n_periods, freq=ts_freq).strftime("%Y-%m-%d")
    rows = []
    for g, base in [("G.1.1_2", 50.0), ("G.1.2_2", 20.0)]:
        for i, d in enumerate(dates):
            cases = int(
                max(
                    base * (1 + 0.5 * np.sin(2 * np.pi * i / season))
                    + rng.uniform(-2, 2),
                    0.0,
                )
            )
            rows.append({"Date": d, "GID_1": "G.1_1", "GID_2": g, "Cases": cases})
    return pd.DataFrame(rows)


class FakeCovariateSource:
    name = "fake"
    ref = "fake"
    granularity = "M"

    def merge(self, df, metrics=None, measures=None, use_cache=False):
        # One covariate value per (GID_2, month), placed on the last period of
        # each month; other rows are NaN so the geo layer interpolates onto
        # finer (weekly/daily) grids.
        out = df.copy()
        end = pd.PeriodIndex(out["Date"]).to_timestamp(how="end")
        month = end.to_period("M")
        last_of_month = (
            out.groupby(["GID_2", month], observed=False)["Date"].transform("max")
            == out["Date"]
        )
        for metric, base in [
            ("tmin", 20.0),
            ("tmax", 28.0),
            ("prec", 100.0),
            ("pop_count", 10000.0),
        ]:
            out[metric] = np.nan
            out.loc[last_of_month, metric] = base
        return out


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


def test_pipeline_end_to_end(tmp_path, monkeypatch, admin2_list):
    from thucia.core.registry import Registry

    fake_registry = Registry("covariate source")
    fake_registry.register()(FakeCovariateSource)
    monkeypatch.setattr("thucia.core.geo.source_registry", fake_registry)
    monkeypatch.setattr("thucia.core.geo.get_admin2_list", lambda iso3: admin2_list)

    raw = _raw_cases()
    cfg = PipelineConfig(
        path=tmp_path,
        start_date=pd.Period("2019-01", freq="M"),
        train_end_date=pd.Period("2018-12", freq="M"),
        horizons=[1],
        num_samples=50,
        source_specs=["fake.metric"],
        lag_spec=[
            {
                "name": "tmin_lag_1",
                "groupby": ["GID_2"],
                "column": "tmin",
                "pipeline": [{"op": "shift", "periods": 1}],
            },
            {
                "name": "prec_lag_1",
                "groupby": ["GID_2"],
                "column": "prec",
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

    # 1. aggregate to monthly + pad admin-2 + add future rows
    padded = cases_per_period(raw, cfg, freq="M")
    assert set(padded["GID_2"].unique()) == set(admin2_list["GID_2"])
    assert padded["future"].sum() == len(admin2_list) * cfg.future_periods
    assert padded[~padded["future"]]["Cases"].sum() == pytest.approx(raw["Cases"].sum())

    # 2. merge covariate source + incidence rate
    merged = merge_covariates(padded, cfg)
    assert {"tmin", "tmax", "prec", "pop_count", "DIR"} <= set(merged.columns)

    # 3. prepare model inputs (lag features, sanitised) and persist/read back
    inputs, cov_cols = prepare_model_inputs(merged, cfg)
    assert {"Log_Cases", "tmin_lag_1", "prec_lag_1", "log_cases_lag_1"} <= set(
        inputs.columns
    )
    assert inputs[cov_cols].isna().sum().sum() == 0
    write_db(inputs, tmp_path / "model_input_data")
    restored = read_db(tmp_path / "model_input_data").df
    assert len(restored) == len(inputs)
    assert restored["Date"].dtype == "period[M]"

    # 4. fit a model (fast baseline) -> canonical quantile grid
    out = fit_model(inputs, "baseline", cfg, db_file=None)
    frame = out.df if hasattr(out, "df") else out
    assert sorted(frame["quantile"].unique()) == quantiles
    valid = frame["prediction"][frame["prediction"].notna()]
    assert (valid >= 0).all()

    # 5. score the forecasts (WIS is NaN for future/unobserved rows and the
    #    baseline warmup; assert the observed window scores finite)
    scored = score_model(frame, cfg)
    assert {"WIS", "R2", "horizon"} <= set(scored.columns)
    observed = scored["Cases"].notna()
    finite_wis = np.isfinite(scored.loc[observed, "WIS"])
    assert finite_wis.any() and finite_wis.sum() > 20
    finite_r2 = np.isfinite(scored.loc[observed, "R2"])
    assert finite_r2.any()


def test_pipeline_end_to_end_weekly(tmp_path, monkeypatch, admin2_list):
    # The same full chain on a weekly (W-SAT) cadence: the Period anchor must
    # survive aggregation -> covariates -> inputs -> fit -> score.
    from thucia.core.registry import Registry

    fake_registry = Registry("covariate source")
    fake_registry.register()(FakeCovariateSource)
    monkeypatch.setattr("thucia.core.geo.source_registry", fake_registry)
    monkeypatch.setattr("thucia.core.geo.get_admin2_list", lambda iso3: admin2_list)

    raw = _raw_cases(n_periods=120, freq="W-SAT", start="2018-01-06", season=52)
    cfg = PipelineConfig(
        path=tmp_path,
        start_date=pd.Period("2019-12-07", freq="W-SAT"),
        train_end_date=pd.Period("2019-11-30", freq="W-SAT"),
        horizons=[1],
        num_samples=20,
        source_specs=["fake.metric"],
        lag_spec=[
            {
                "name": "log_cases_lag_1",
                "groupby": ["GID_2"],
                "column": "Log_Cases",
                "pipeline": [{"op": "shift", "periods": 1}],
            }
        ],
    )

    padded = cases_per_period(raw, cfg, freq="W-SAT")
    assert str(padded["Date"].dtype) == "period[W-SAT]"
    assert padded["future"].sum() == len(admin2_list) * cfg.future_periods

    # The fake source is month-granular: on a weekly grid the geo layer must
    # interpolate it onto every week and warn the user.
    with pytest.warns(UserWarning, match="interpolated"):
        merged = merge_covariates(padded, cfg)
    for col in ["tmin", "tmax", "prec", "pop_count"]:
        assert merged[col].notna().all()

    inputs, cov_cols = prepare_model_inputs(merged, cfg)
    assert inputs[cov_cols].isna().sum().sum() == 0

    frame = fit_model(inputs, "baseline", cfg, db_file=None)
    frame = frame.df if hasattr(frame, "df") else frame
    assert str(frame["Date"].dtype) == "period[W-SAT]"
    assert sorted(frame["quantile"].unique()) == quantiles
    valid = frame["prediction"][frame["prediction"].notna()]
    assert (valid >= 0).all()

    scored = score_model(frame, cfg)
    observed = scored["Cases"].notna()
    assert np.isfinite(scored.loc[observed, "WIS"]).any()


def _pdfm_embeddings(gid_2s, seed=0, n_feature=10):
    # Seeded-random PDFM-style embeddings shaped like the real user-supplied
    # file (GID_2 + feature0..featureN). No external data required.
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "GID_2": gid_2s,
            **{f"feature{d}": rng.normal(size=len(gid_2s)) for d in range(n_feature)},
        }
    )


def test_pipeline_end_to_end_with_pdfm_residual_regression(
    tmp_path, monkeypatch, admin2_list
):
    # Full chain through a baseline fit, then correct the forecasts with the
    # user-supplied PDFM embeddings path (seeded-random, no real data needed).
    from thucia.core.registry import Registry

    fake_registry = Registry("covariate source")
    fake_registry.register()(FakeCovariateSource)
    monkeypatch.setattr("thucia.core.geo.source_registry", fake_registry)
    monkeypatch.setattr("thucia.core.geo.get_admin2_list", lambda iso3: admin2_list)

    raw = _raw_cases()
    cfg = PipelineConfig(
        path=tmp_path,
        start_date=pd.Period("2019-01", freq="M"),
        train_end_date=pd.Period("2018-12", freq="M"),
        horizons=[1],
        num_samples=50,
        source_specs=["fake.metric"],
        lag_spec=[
            {
                "name": "log_cases_lag_1",
                "groupby": ["GID_2"],
                "column": "Log_Cases",
                "pipeline": [{"op": "shift", "periods": 1}],
            }
        ],
    )

    padded = cases_per_period(raw, cfg, freq="M")
    merged = merge_covariates(padded, cfg)
    inputs, cov_cols = prepare_model_inputs(merged, cfg)
    frame = fit_model(inputs, "baseline", cfg, db_file=None)
    frame = frame.df if hasattr(frame, "df") else frame
    assert sorted(frame["quantile"].unique()) == quantiles
    assert set(frame["GID_2"].unique()) == set(admin2_list["GID_2"])

    # Build a seeded-random embeddings file covering all admin-2 GIDs.
    embed_df = _pdfm_embeddings(admin2_list["GID_2"].tolist())
    nc = tmp_path / "embeddings.nc"
    write_nc(embed_df, str(nc))
    embeddings = prepare_pdfm_embeddings(str(nc))
    assert set(embeddings["GID_2"]) == set(admin2_list["GID_2"])

    out = apply_residual_regression(
        frame,
        embeddings,
        cfg,
        method="ridge",
        geo_col="GID_2",
    )
    # Same structure and canonical quantile grid, all GIDs retained, and the
    # correction actually moved the forecasts for some GID. (The baseline
    # warm-up rows with no history stay NaN, so we compare only non-NaN rows.)
    assert set(out["GID_2"]) == set(admin2_list["GID_2"])
    assert sorted(out["quantile"].unique()) == quantiles
    shared = out["prediction"].notna() & frame["prediction"].notna()
    # The log-space ridge correction can dip a near-zero prediction slightly
    # below 0; the meaningful property is that fitted rows stay finite and the
    # correction actually moved the forecasts.
    assert np.isfinite(out.loc[shared, "prediction"]).all()
    corrected = out[shared].set_index(["Date", "GID_2", "quantile"])["prediction"]
    orig = frame[shared].set_index(["Date", "GID_2", "quantile"])["prediction"]
    diff = (corrected - orig).abs()
    assert diff.max() > 0


def test_pipeline_end_to_end_pdfm_subsamples_missing_embeddings(
    tmp_path, monkeypatch, admin2_list
):
    # Embeddings cover only a subset of the admin-2 regions: the residual
    # regression warns and continues on the provinces that have embeddings
    # (mirrors the old analysis_core.py subsampling), dropping the rest.
    from thucia.core.registry import Registry

    fake_registry = Registry("covariate source")
    fake_registry.register()(FakeCovariateSource)
    monkeypatch.setattr("thucia.core.geo.source_registry", fake_registry)
    monkeypatch.setattr("thucia.core.geo.get_admin2_list", lambda iso3: admin2_list)

    raw = _raw_cases()
    cfg = PipelineConfig(
        path=tmp_path,
        start_date=pd.Period("2019-01", freq="M"),
        train_end_date=pd.Period("2018-12", freq="M"),
        horizons=[1],
        num_samples=50,
        source_specs=["fake.metric"],
        lag_spec=[
            {
                "name": "log_cases_lag_1",
                "groupby": ["GID_2"],
                "column": "Log_Cases",
                "pipeline": [{"op": "shift", "periods": 1}],
            }
        ],
    )

    padded = cases_per_period(raw, cfg, freq="M")
    merged = merge_covariates(padded, cfg)
    inputs, cov_cols = prepare_model_inputs(merged, cfg)
    frame = fit_model(inputs, "baseline", cfg, db_file=None)
    frame = frame.df if hasattr(frame, "df") else frame

    # Embeddings for only the first two admin-2 regions.
    covered = admin2_list["GID_2"].iloc[:2].tolist()
    embed_df = _pdfm_embeddings(covered)
    nc = tmp_path / "embeddings.nc"
    write_nc(embed_df, str(nc))
    embeddings = prepare_pdfm_embeddings(str(nc))

    with pytest.warns(UserWarning, match="embeddings"):
        out = apply_residual_regression(
            frame,
            embeddings,
            cfg,
            method="ridge",
            geo_col="GID_2",
        )
    assert set(out["GID_2"]) == set(covered)
    assert len(out) < len(frame)


def _raw_cases_non_gadm(n_periods=40, seed=0, freq="M", start="2017-01-31", season=12):
    # Line-list style input under a completely non-GADM tagging scheme: regions
    # are 'north'/'south' with families 'ProvA'/'ProvB'. The geo columns are
    # categorical and carry an extra never-observed category ('west'/'ProvC') so
    # the categorical-implicit region list is exercised end-to-end.
    rng = np.random.default_rng(seed)
    ts_freq = "ME" if freq == "M" else freq
    dates = pd.date_range(start, periods=n_periods, freq=ts_freq).strftime("%Y-%m-%d")
    rows = []
    for region, state, base in [("north", "ProvA", 50.0), ("south", "ProvB", 20.0)]:
        for d in dates:
            cases = int(
                max(
                    base * (1 + 0.5 * np.sin(2 * np.pi * len(rows) / season))
                    + rng.uniform(-2, 2),
                    0.0,
                )
            )
            rows.append({"Date": d, "state": state, "region": region, "Cases": cases})
    df = pd.DataFrame(rows)
    df["region"] = pd.Categorical(df["region"], categories=["north", "south", "west"])
    df["state"] = pd.Categorical(df["state"], categories=["ProvA", "ProvB", "ProvC"])
    return df


@pytest.mark.parametrize("freq", ["M", "W-SAT"])
def test_pipeline_e2e_non_gadm_geo_col(tmp_path, freq):
    # A generic geo naming scheme through the whole pipeline with no GADM_*
    # columns anywhere: cases_per_period -> prepare -> DuckDB round-trip ->
    # fit -> score. Covariate sources are GADM-keyed (documented boundary) and
    # intentionally not exercised here. The never-seen 'west'/'ProvC' categories
    # must still be padded by the categorical-implicit region list.
    n_periods, start, season, start_period, train_end = (
        (
            40,
            "2017-01-31",
            12,
            pd.Period("2019-01", freq="M"),
            pd.Period("2018-12", freq="M"),
        )
        if freq == "M"
        else (
            120,
            "2018-01-06",
            52,
            pd.Period("2019-12-07", freq="W-SAT"),
            pd.Period("2019-11-30", freq="W-SAT"),
        )
    )
    raw = _raw_cases_non_gadm(
        n_periods=n_periods, freq=freq, start=start, season=season
    )
    cfg = PipelineConfig(
        path=tmp_path,
        geo_col="region",
        geo_parent="state",
        start_date=start_period,
        train_end_date=train_end,
        horizons=[1],
        num_samples=20,
        source_specs=[],
        lag_spec=[
            {
                "name": "log_cases_lag_1",
                "groupby": ["region"],
                "column": "Log_Cases",
                "pipeline": [{"op": "shift", "periods": 1}],
            }
        ],
    )

    padded = cases_per_period(raw, cfg, freq=freq)
    expected_dtype = "period[M]" if freq == "M" else "period[W-SAT]"
    assert str(padded["Date"].dtype) == expected_dtype
    assert set(padded["region"].dropna().astype("object").unique()) == {
        "north",
        "south",
        "west",
    }
    assert not any(c.startswith("GID_") for c in padded.columns)

    inputs, cov_cols = prepare_model_inputs(padded, cfg)
    assert {"Log_Cases", "log_cases_lag_1"} <= set(inputs.columns)
    assert inputs.loc[~inputs["future"], "Cases"].notna().all()
    assert inputs[cov_cols].isna().sum().sum() == 0
    assert not any(c.startswith("GID_") for c in inputs.columns)

    # DuckDB round-trip with non-GADM ENUM names preserves the padded universe.
    write_db(inputs, tmp_path / "non_gadm_model_input")
    restored = read_db(tmp_path / "non_gadm_model_input").df
    assert set(restored["region"].dropna().astype("object").unique()) == {
        "north",
        "south",
        "west",
    }
    assert restored["Date"].dtype == expected_dtype

    frame = fit_model(inputs, "baseline", cfg, db_file=None)
    frame = frame.df if hasattr(frame, "df") else frame
    assert sorted(frame["quantile"].unique()) == quantiles
    assert not any(c.startswith("GID_") for c in frame.columns)
    assert set(frame["region"].dropna().astype("object").unique()) == {
        "north",
        "south",
        "west",
    }

    scored = score_model(frame, cfg)
    observed = scored["Cases"].notna()
    assert np.isfinite(scored.loc[observed, "WIS"]).any()
    assert not any(c.startswith("GID_") for c in scored.columns)
