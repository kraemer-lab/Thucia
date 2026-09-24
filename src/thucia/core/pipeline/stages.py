# Pipeline computation stages.
#
# These wrap the library primitives (aggregation, geo padding, covariate
# merging, model input prep, model fitting, scoring) into the stage functions
# consumed by `thucia.core.validation.run_backtest` and other entry points.
# They are deliberately thin, data-in/data-out functions so they can be tested
# in isolation and reused.
from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import Optional

import numpy as np
import pandas as pd
import thucia.core.models as models
from thucia.core.cases import aggregate_cases
from thucia.core.cases import cases_per_month
from thucia.core.cases import check_covars_for_nans
from thucia.core.cases import check_index_combinations
from thucia.core.cases import quantile_sum_gid
from thucia.core.cases import r2
from thucia.core.cases import wis
from thucia.core.fs import DataFrame
from thucia.core.geo import add_incidence_rate
from thucia.core.geo import lookup_gid1
from thucia.core.geo import merge_sources
from thucia.core.geo import pad_admin2
from thucia.core.models import get_model
from thucia.core.models import run_model
from thucia.core.models.ensemble import create_ensemble
from thucia.core.models.utils import sanitise_covariates
from thucia.core.models.utils.adapter import residual_regression
from thucia.core.models.utils.covariates import build_features

from .config import PipelineConfig


def base_columns(config: PipelineConfig) -> tuple[str, ...]:
    """Columns that are structural, not covariates.

    ``Date``, the geo column, ``future`` and ``Cases`` are always base; the
    geo-parent column is included only when the config declares one.
    """
    cols = ["Date", config.geo_col, "future", "Cases"]
    if config.geo_parent:
        cols.insert(1, config.geo_parent)
    return tuple(cols)


def cases_per_period(
    df: pd.DataFrame | DataFrame,
    config: PipelineConfig,
    freq: str = "M",
) -> pd.DataFrame:
    """Aggregate cases to a period, pad admin-2 regions, and add future rows.

    Returns the padded frame (historical rows plus `future_months` of future
    placeholder rows with NaN Cases and ``future=True``).
    """
    if freq == "M":
        tdf = cases_per_month(df, cutoff_date=config.cutoff_date)
    else:
        tdf = aggregate_cases(df, cutoff_date=config.cutoff_date, freq=freq)
    tdf = pad_admin2(
        tdf,
        geo_col=config.geo_col,
        geo_parent=config.geo_parent or "GID_1",
        iso3=config.iso3,
    )

    last_date = tdf["Date"].max()
    future_dates = pd.period_range(
        start=last_date + 1,
        periods=config.future_periods,
        freq=last_date.freq,
    )
    out = tdf.df
    out["future"] = False
    last_row = out[out["Date"] == last_date].copy()
    for date in future_dates:
        future = last_row.copy()
        future["Date"] = date
        future["Cases"] = np.nan
        future["future"] = True
        out = pd.concat([out, future], ignore_index=True)
    return out.sort_values(by=["Date", config.geo_col]).reset_index(drop=True)


def merge_covariates(
    df: pd.DataFrame,
    config: PipelineConfig,
) -> pd.DataFrame:
    """Merge each covariate source and add the incidence-rate column."""
    out = df.copy()
    for spec in config.source_specs:
        merged = merge_sources(df, [spec], method=config.covariate_interpolation)
        new_cols = [c for c in merged.columns if c not in df.columns]
        out = out.merge(
            merged[[config.geo_col, "Date"] + new_cols],
            on=[config.geo_col, "Date"],
            how="left",
        )
    return add_incidence_rate(out)


def prepare_model_inputs(
    df: pd.DataFrame,
    config: PipelineConfig,
) -> tuple[pd.DataFrame, list[str]]:
    """Build the model-input frame: Log_Cases, lag features, sanitisation.

    Returns ``(frame, covariate_cols)``. When ``config.lag_spec`` is set the
    features are built via ``build_features``; otherwise every non-base column
    of the input is treated as a covariate.
    """
    out = df.copy()
    out[config.case_col] = np.log1p(out["Cases"])

    if config.lag_spec:
        out = build_features(out, config.lag_spec)
        covariate_cols = [
            f["name"] for f in config.lag_spec if f["name"] != config.case_col
        ]
    else:
        covariate_cols = [
            c
            for c in out.columns
            if c not in set(base_columns(config)) | {config.case_col}
        ]

    keep = [
        c
        for c in [*base_columns(config), config.case_col] + covariate_cols
        if c in out.columns
    ]
    out = out[keep]

    if covariate_cols:
        out = sanitise_covariates(out, covariate_cols, config.train_end_date)
        check_covars_for_nans(out, covariate_cols)
    check_covars_for_nans(out[~out["future"]], [config.case_col])
    index_cols = ["Date", config.geo_col] if config.geo_col in out.columns else ["Date"]
    check_index_combinations(out, index_cols)
    return out, covariate_cols


def fit_model(
    df: pd.DataFrame,
    model_name: str,
    config: PipelineConfig,
    *,
    db_file: str | Path | None = None,
) -> Any:
    """Fit a named model on prepared inputs and return the quantile frame."""
    model = get_model(model_name)

    base_cols = set(base_columns(config)) | {config.case_col}
    covariate_cols = [c for c in df.columns if c not in base_cols]
    db_file = db_file or config.path / f"{model_name}_cases_quantiles.duckdb"

    model_kwargs: dict[str, Any] = {
        "start_date": config.start_date,
        "geo_col": config.geo_col,
        "geo_parent": config.geo_parent,
        "geo_parent_filter": (
            lookup_gid1(config.iso3, config.adm1) if config.iso3 else config.adm1
        ),
        "horizons": config.horizons,
        "case_col": config.case_col,
        "covariate_cols": covariate_cols,
        "train_col": config.train_col or config.geo_col,
        "db_file": db_file,
    }
    # Each model declares, via its ModelSpec, which extra config knobs it
    # accepts (beyond the common ones above). Map supported knobs to their
    # PipelineConfig values instead of special-casing model names.
    spec = models.get_model_spec(model_name)
    knob_value = {
        "train_end_date": config.train_end_date,
        "retrain": config.retrain,
        "multivariate": config.multivariate,
        "samples": config.num_samples,
        "season_length": config.season_length,
    }
    kwarg_name = {"samples": "num_samples"}  # others are identical to knob names
    for knob, value in knob_value.items():
        if knob in spec.supports:
            model_kwargs[kwarg_name.get(knob, knob)] = value

    return run_model(model_name, model, df, config.path, model_kwargs=model_kwargs)


def score_model(
    df_quantiles: pd.DataFrame,
    config: PipelineConfig,
    geo_col: str = "GID_2",
) -> pd.DataFrame:
    """Score a quantile frame: WIS and R2 per geo and horizon."""
    parts = []
    for h in config.horizons:
        dfh = df_quantiles[df_quantiles["horizon"] == h].copy()
        if dfh.empty:
            continue
        dfh["prediction"] = np.log1p(dfh["prediction"])
        dfh["Cases"] = np.log1p(dfh["Cases"])
        scored = wis(dfh, "prediction", "Cases", geo_col=geo_col)
        scored["horizon"] = h
        scored = scored.merge(
            r2(dfh, "prediction", "Cases", group_col=geo_col),
            on=[geo_col],
            how="left",
        )
        parts.append(scored)
    return pd.concat(parts, ignore_index=True)


def aggregate_quantiles(
    df_quantiles: pd.DataFrame,
    config: PipelineConfig,
    *,
    geo_parent: str = "GID_1",
    geo_col: str = "GID_2",
    db_file: str | Path | None = None,
    samples: int = 10000,
) -> DataFrame:
    """Aggregate per-geo quantiles to a coarser admin level (MC sum)."""
    db_file = db_file or config.path / "aggregated_cases_quantiles.duckdb"
    return quantile_sum_gid(
        df_quantiles,
        db_file=str(db_file),
        new_file=True,
        samples=samples,
        geo_col=geo_col,
        geo_parent=geo_parent,
    )


def build_ensemble(
    dfs: list[pd.DataFrame],
    config: PipelineConfig,
    model_names: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Weighted ensemble over per-model quantile frames."""
    return create_ensemble(dfs, model_names)


def apply_residual_regression(
    df_quantiles: pd.DataFrame,
    embeddings: pd.DataFrame,
    config: PipelineConfig,
    *,
    method: str = "pinball",
    geo_col: str = "GID_2",
    horizons: Optional[list[int]] = None,
) -> pd.DataFrame:
    """Post-hoc residual regression against PDFM-style embeddings."""
    return residual_regression(
        df_quantiles,
        embeddings,
        method,
        geo_col=geo_col,
        horizons=horizons or config.horizons,
    )
