# Pipeline computation stages.
#
# These wrap the library primitives (aggregation, geo padding, covariate
# merging, model input prep, model fitting, scoring) into the stage functions
# consumed by `thucia.core.validation.run_backtest` and other entry points.
# They are deliberately thin, data-in/data-out functions so they can be tested
# in isolation and reused.
from __future__ import annotations

import dataclasses
import warnings
from pathlib import Path
from typing import Any
from typing import Iterable
from typing import Mapping
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
from thucia.core.geo import _load_regions_gdf
from thucia.core.geo import add_incidence_rate
from thucia.core.geo import ensure_all_regions
from thucia.core.geo import merge_sources
from thucia.core.models import get_model
from thucia.core.models import run_model
from thucia.core.models._meta import as_selection
from thucia.core.models._meta import ModelSelection
from thucia.core.models._meta import ModelSpec
from thucia.core.models._meta import validate_model_params
from thucia.core.models.ensemble import create_ensemble
from thucia.core.models.utils import sanitise_covariates
from thucia.core.models.utils.adapter import residual_regression
from thucia.core.models.utils.covariates import build_features
from thucia.core.models.utils.residual_quantiles import add_residual_quantiles

from .config import PipelineConfig
from .provenance import check_expected_code
from .provenance import run_metadata_docs

#: Config knobs `fit_model` always forwards to the model callable (as opposed
#: to the per-model knobs a `ModelSpec` declares support for).
_ALWAYS_PASSED = frozenset(
    {
        "adm1",
        "case_col",
        "geo_col",
        "geo_parent",
        "horizons",
        "path",
        "start_date",
        "train_col",
    }
)


def base_columns(config: PipelineConfig) -> tuple[str, ...]:
    """Columns that are structural, not covariates.

    ``Date``, the geo column, ``future`` and ``Cases`` are always base; the
    geo-parent column is included only when the config declares one.
    """
    cols = ["Date", config.geo_col, "future", "Cases"]
    if config.geo_parent:
        cols.insert(1, config.geo_parent)
    return tuple(cols)


def coerce_geo_cols(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Coerce the geo code columns to ``str`` (geo codes are strings).

    Applied at pipeline entry so downstream joins, group-bys and ENUM writes
    all see the same string-typed codes regardless of the loader's dtype.
    """
    for col in (config.geo_col, config.geo_parent):
        if col is None or col not in df.columns:
            continue
        dtype = df[col].dtype
        if pd.api.types.is_object_dtype(dtype):
            continue
        if isinstance(dtype, pd.CategoricalDtype):
            # String categoricals remain categorical: the categories are a
            # legitimate implicit region list that ensure_all_regions and the
            # aggregation's observed=False grouping rely on.
            if pd.api.types.is_string_dtype(dtype.categories.dtype):
                continue
        elif pd.api.types.is_string_dtype(dtype):
            continue
        df[col] = df[col].astype(str)
    return df


def cases_per_period(
    df: pd.DataFrame | DataFrame,
    config: PipelineConfig,
    freq: str = "M",
) -> pd.DataFrame:
    """Aggregate cases to a period, pad admin-2 regions, and add future rows.

    Returns the padded frame (historical rows plus `future_months` of future
    placeholder rows with NaN Cases and ``future=True``).
    """
    if isinstance(df, DataFrame):
        df = df.df
    coerce_geo_cols(df, config)
    # Ensure all regions BEFORE aggregation: the grid fill in aggregate_cases
    # and the fs ENUM layer only ever see *observed* geo codes, so a never-seen
    # region won't survive aggregation for the region list to discover later.
    # The authoritative list (explicit roster, categorical categories, or GADM)
    # is resolved here against the raw line-list, then aggregation runs on the
    # padded frame so every region is present from the start.
    tdf = ensure_all_regions(
        df,
        geo_col=config.geo_col,
        geo_parent=config.geo_parent,
        iso3=config.iso3,
        regions=_load_regions_gdf(config.regions, config.region_col, config.geo_col),
    )
    tdf = tdf.df if isinstance(tdf, DataFrame) else tdf
    if freq == "M":
        tdf = cases_per_month(
            tdf, cutoff_date=config.cutoff_date, geo_col=config.geo_col
        )
    else:
        tdf = aggregate_cases(
            tdf, cutoff_date=config.cutoff_date, freq=freq, geo_col=config.geo_col
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
        merged = merge_sources(
            df,
            [spec],
            method=config.covariate_interpolation,
            geo_col=config.geo_col,
            iso3=config.iso3,
            regions=config.regions,
            region_col=config.region_col,
            use_cache=config.use_cache,
        )
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
    out = coerce_geo_cols(df, config).copy()
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
        out = sanitise_covariates(
            out, covariate_cols, config.train_end_date, geo_col=config.geo_col
        )
        check_covars_for_nans(out, covariate_cols)
    check_covars_for_nans(out[~out["future"]], [config.case_col])
    index_cols = ["Date", config.geo_col] if config.geo_col in out.columns else ["Date"]
    check_index_combinations(out, index_cols)
    return out, covariate_cols


def _resolve_overrides(
    selection: ModelSelection,
    config: PipelineConfig,
    spec: ModelSpec,
) -> PipelineConfig:
    """Validate and apply a selection's per-model config overrides.

    Overrides must name real ``PipelineConfig`` fields. A field the selected
    model never consumes (neither always forwarded nor declared by its
    ``ModelSpec``) is kept, but warned about, since it would otherwise be a
    silent no-op.
    """
    if not selection.overrides:
        return config
    fields = {f.name for f in dataclasses.fields(PipelineConfig)}
    unknown = set(selection.overrides) - fields
    if unknown:
        raise ValueError(
            f"{selection.run_label}: unknown config overrides {sorted(unknown)}; "
            "use PipelineConfig field names."
        )
    unused = set(selection.overrides) - _ALWAYS_PASSED - set(spec.supports)
    if unused:
        warnings.warn(
            f"{selection.run_label}: override(s) {sorted(unused)} are not used by "
            f"model '{selection.name}' (declared: {sorted(spec.supports)}); "
            "they are applied to the config but not passed to the model.",
            UserWarning,
            stacklevel=3,
        )
    return dataclasses.replace(config, **selection.overrides)


def fit_model(
    df: pd.DataFrame,
    model: str | ModelSelection,
    config: PipelineConfig,
    *,
    db_file: str | Path | None = None,
    expect: str | Path | Mapping[str, Any] | None = None,
    expect_commit: str | None = None,
    expect_version: str | None = None,
) -> Any:
    """Fit a model on prepared inputs and return the quantile frame.

    ``model`` is either an advertised model name or a
    :class:`~thucia.core.models._meta.ModelSelection` carrying a label and
    per-model configuration. The label names the output DuckDB file and the
    run metadata recorded in it.

    ``expect`` / ``expect_commit`` / ``expect_version`` pin the *implementation*:
    the fit is refused when the running Thucia is not the one that produced
    ``expect`` (a saved artefact or its metadata), is not at ``expect_commit``
    (source installs only, clean tree required), or is not ``expect_version``.
    """
    selection = as_selection(model)
    check_expected_code(
        expect=expect, expect_commit=expect_commit, expect_version=expect_version
    )
    label = selection.run_label
    model_fn = get_model(selection.name)
    validate_model_params(selection, model_fn)
    spec = models.get_model_spec(selection.name)
    cfg = _resolve_overrides(selection, config, spec)

    base_cols = set(base_columns(cfg)) | {cfg.case_col}
    covariate_cols = [c for c in df.columns if c not in base_cols]
    db_file = db_file or cfg.path / f"{label}_cases_quantiles.duckdb"

    model_kwargs: dict[str, Any] = {
        "start_date": cfg.start_date,
        "geo_col": cfg.geo_col,
        "geo_parent": cfg.geo_parent,
        "geo_parent_filter": cfg.adm1,
        "horizons": cfg.horizons,
        "case_col": cfg.case_col,
        "covariate_cols": covariate_cols,
        "train_col": cfg.train_col or cfg.geo_col,
        "db_file": db_file,
    }
    # Each model declares, via its ModelSpec, which extra config knobs it
    # accepts (beyond the common ones above). Map supported knobs to their
    # PipelineConfig values instead of special-casing model names.
    knob_value = {
        "train_end_date": cfg.train_end_date,
        "retrain": cfg.retrain,
        "multivariate": cfg.multivariate,
        "samples": cfg.num_samples,
        "season_length": cfg.season_length,
        "quantiles": cfg.quantiles,
        "median_only": cfg.median_only,
        "past_covariate_cols": cfg.past_covariate_cols,
        "future_covariate_cols": cfg.future_covariate_cols,
    }
    kwarg_name = {"samples": "num_samples"}  # others are identical to knob names
    for knob, value in knob_value.items():
        if knob in spec.supports:
            model_kwargs[kwarg_name.get(knob, knob)] = value
    # Per-run params win over the config projection (validated above).
    model_kwargs.update(selection.params)

    docs = run_metadata_docs(selection, cfg, df)
    frame = run_model(
        label,
        model_fn,
        df,
        cfg.path,
        model_kwargs=model_kwargs,
        metadata=docs,
    )
    if isinstance(frame, DataFrame):
        # Models that persist their own output (e.g. `baseline` with a
        # `db_file`) never reach `run_model`'s write path, so record the run
        # metadata on the file they produced.
        _attach_metadata(db_file, docs)
    return frame


def _attach_metadata(db_file: str | Path, docs: dict[str, Any]) -> None:
    """Record run metadata on an artefact a model wrote itself."""
    path = Path(db_file)
    if path.suffix not in ("", ".duckdb") or not path.exists():
        return
    DataFrame(db_path=str(path)).set_metadata(docs)


def fit_models(
    df: pd.DataFrame,
    selections: Iterable[str | ModelSelection],
    config: PipelineConfig,
    *,
    expect: str | Path | Mapping[str, Any] | None = None,
    expect_commit: str | None = None,
    expect_version: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Fit several models (or model variants) on the same prepared inputs.

    Returns ``{label: quantile frame}``, so one prepared frame and one set of
    assumptions can drive a whole model comparison. Labels must be unique —
    two variants of the same model need distinct labels (otherwise they would
    share an output file and be indistinguishable in score rows). The
    ``expect*`` pins are checked once, before any model is fitted.
    """
    normalised = [as_selection(s) for s in selections]
    labels = [s.run_label for s in normalised]
    duplicates = sorted({lb for lb in labels if labels.count(lb) > 1})
    if duplicates:
        raise ValueError(
            f"duplicate model labels {duplicates}; give each variant a distinct "
            "ModelSelection(label=...)."
        )

    check_expected_code(
        expect=expect, expect_commit=expect_commit, expect_version=expect_version
    )

    fitted: dict[str, pd.DataFrame] = {}
    for selection in normalised:
        frame = fit_model(df, selection, config)
        fitted[selection.run_label] = frame.df if hasattr(frame, "df") else frame
    return fitted


def score_model(
    df_quantiles: pd.DataFrame,
    config: PipelineConfig,
    geo_col: str | None = None,
) -> pd.DataFrame:
    """Score a quantile frame: WIS and R2 per geo and horizon."""
    geo_col = geo_col or config.geo_col
    if "horizon" not in df_quantiles.columns:
        df_quantiles = df_quantiles.assign(horizon=1)
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


def apply_residual_quantiles(
    df_forecast: pd.DataFrame,
    config: PipelineConfig,
    *,
    geo_col: str | None = None,
    db_file: str | Path | None = None,
) -> pd.DataFrame:
    """Spread a point (median) forecast into the canonical quantile grid.

    The counterpart to :func:`apply_residual_regression` for models that emit no
    spread at all (e.g. a foundation checkpoint fitted with ``median_only=True``):
    the distribution is built from the empirical quantiles of the forecast's own
    past errors. The pass is causal — at each date the residual history holds
    only earlier errors — so the result is out-of-sample with respect to the
    correction itself.

    ``config`` supplies the behaviour: ``residual_quantile_window`` (None keeps
    an expanding history), ``residual_min_history`` and
    ``residual_pool_across_gids``. Regions that have not yet accumulated
    ``residual_min_history`` residuals fall back to the point forecast at every
    level (a degenerate distribution) unless the pooled cross-region history is
    long enough.

    ``df_forecast`` must be single-level; a multi-level frame is rejected because
    per-level predictions cannot be attributed to the residual history.
    """
    if "quantile" in df_forecast.columns and df_forecast["quantile"].nunique() > 1:
        raise ValueError(
            "apply_residual_quantiles expects a single-level (point) forecast; "
            f"got {df_forecast['quantile'].nunique()} quantile levels. Use "
            "apply_residual_regression to correct an existing distribution."
        )
    spread = add_residual_quantiles(
        # The quantile step log1p-transforms its input in place.
        df_forecast.copy(),
        date_col="Date",
        geo_col=geo_col or config.geo_col,
        horizon_col="horizon",
        window=config.residual_quantile_window,
        min_history=config.residual_min_history,
        pool_across_gids_if_sparse=config.residual_pool_across_gids,
        db_file=str(db_file) if db_file is not None else None,
    )
    return spread.df if hasattr(spread, "df") else spread
