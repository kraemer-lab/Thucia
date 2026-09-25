# Forecast-quality validation over the reusable pipeline stages.
#
# A backtest cuts the history at a series of cutoff dates, fits a model on the
# past only, scores the held-out window, and aggregates WIS / RMSE / R2 per
# horizon. It consumes the output of ``prepare_model_inputs`` so the expensive
# geo/covariate steps run once; the full chain is:
#
#     cases_per_period -> merge_covariates -> prepare_model_inputs -> run_backtest
from __future__ import annotations

import dataclasses
import tempfile
from dataclasses import dataclass
from typing import Callable
from typing import Optional

import numpy as np
import pandas as pd
from thucia.core.models import get_model_spec as _get_model_spec
from thucia.core.pipeline import fit_model as _fit_model
from thucia.core.pipeline import PipelineConfig
from thucia.core.pipeline import score_model


def _is_fast(model_name: str) -> bool:
    """Whether a model is cheap enough to fit across backtest windows.

    Reads the model's declarative ModelSpec (`fast=True` for baseline/movavg).
    """
    return _get_model_spec(model_name).fast


def expand_cutoffs(
    dates,
    min_history: int = 12,
    step: int = 1,
    max_horizon: int = 1,
) -> list[pd.Period]:
    """Candidate cutoff dates for a backtest.

    A cutoff must have at least ``min_history`` observed periods up to and
    including it (so the model has history to fit on), and must leave at least
    ``max_horizon`` observed periods after it (so the longest horizon can be
    scored). Cutoffs are taken every ``step`` periods.
    """
    dates = pd.PeriodIndex(dates).sort_values()
    first = min_history - 1
    last = len(dates) - 1 - max_horizon
    if last < first:
        return []
    return [dates[i] for i in range(first, last + 1, step)]


@dataclass
class BacktestConfig:
    """Knobs for :func:`run_backtest` (model selection and the cutoff sweep)."""

    model_name: str = "baseline"
    #: If set, per-horizon skill is reported relative to this model's WIS
    #: (skill = 1 - model_WIS / reference_WIS at the same cutoff).
    reference_model: Optional[str] = None
    #: Explicit cutoff dates; when None the sweep is generated automatically.
    cutoffs: Optional[list[pd.Period]] = None
    min_history: int = 12
    step: int = 1
    #: None = expanding window; int = rolling window of that many periods.
    window: Optional[int] = None
    #: Refuse models that are expensive to fit repeatedly across cutoffs.
    fast_only: bool = True
    #: Retain each cutoff's held-out quantile frame.
    keep_forecasts: bool = False


@dataclass
class BacktestResult:
    """Output of :func:`run_backtest`.

    ``scores`` is one row per (cutoff, GID, date, horizon) with ``WIS``,
    ``RMSE``, ``R2`` and the observed ``Cases``. ``summary`` aggregates per
    horizon (mean WIS/RMSE/R2, ``n`` windows, and ``skill`` when a reference
    model was requested). ``forecasts`` maps cutoff -> held-out quantile frame
    when ``keep_forecasts`` is set.
    """

    scores: pd.DataFrame
    summary: pd.DataFrame
    forecasts: Optional[dict[pd.Period, pd.DataFrame]] = None


def _holdout_for_cutoff(
    inputs: pd.DataFrame,
    cutoff: pd.Period,
    max_horizon: int,
    window: Optional[int],
) -> pd.DataFrame:
    end = cutoff + max_horizon
    if window is None:
        mask = inputs["Date"] <= end
    else:
        mask = (inputs["Date"] > cutoff - window) & (inputs["Date"] <= end)
    return inputs[mask]


def _fit_quantiles(
    slice_df: pd.DataFrame,
    model_name: str,
    config: PipelineConfig,
    cutoff: pd.Period,
    fit_fn: Callable,
    work_dir,
) -> pd.DataFrame:
    # The fit may persist quantile duckdb files; run it in a scratch dir so a
    # backtest never litters the caller's filesystem.
    cfg = dataclasses.replace(config, train_end_date=cutoff, path=work_dir)
    frame = fit_fn(slice_df, model_name, cfg, db_file=None)
    if hasattr(frame, "df"):
        frame = frame.df
    if "horizon" not in frame.columns or "Date" not in frame.columns:
        raise ValueError(f"Model '{model_name}' returned no scored frame.")
    return frame[frame["Date"] > cutoff]


def _rmse_by_gid_horizon(holdout: pd.DataFrame, geo_col: str = "GID_2") -> pd.DataFrame:
    """Per (geo, horizon) RMSE of the median prediction vs observed Cases."""
    median = holdout[holdout["quantile"] == 0.5]
    if median.empty:
        return pd.DataFrame(columns=[geo_col, "horizon", "RMSE"])
    rmse = (
        median.groupby([geo_col, "horizon"], observed=False)
        .apply(
            lambda d: float(np.sqrt(np.mean((d["prediction"] - d["Cases"]) ** 2))),
            include_groups=False,
        )
        .rename("RMSE")
        .reset_index()
    )
    return rmse


def _skill_vs_reference(
    inputs: pd.DataFrame,
    config: PipelineConfig,
    bt: BacktestConfig,
    cutoffs: list[pd.Period],
    model_wis_by_cutoff: dict[pd.Period, pd.DataFrame],
    fit_fn: Callable,
    work_dir,
) -> pd.DataFrame:
    """Per-horizon skill of the primary model relative to the reference model."""
    max_horizon = max(config.horizons)
    ref_rows = []
    for cutoff in cutoffs:
        slice_df = _holdout_for_cutoff(inputs, cutoff, max_horizon, bt.window)
        try:
            holdout = _fit_quantiles(
                slice_df, bt.reference_model, config, cutoff, fit_fn, work_dir
            )
        except ValueError:
            continue
        scored = score_model(
            holdout,
            dataclasses.replace(config, train_end_date=cutoff),
            geo_col=config.geo_col,
        )
        ref_rows.append(
            scored.groupby("horizon", observed=False)["WIS"]
            .mean()
            .reset_index()
            .assign(cutoff=cutoff)
        )

    if not ref_rows:
        return pd.DataFrame(columns=["horizon", "skill"])

    ref = pd.concat(ref_rows).reset_index()
    model = pd.concat(model_wis_by_cutoff.values()).reset_index()
    merged = model.merge(ref, on=["cutoff", "horizon"], suffixes=("_model", "_ref"))
    merged = merged[merged["WIS_ref"].notna() & (merged["WIS_ref"] > 0)]
    merged["skill"] = 1.0 - merged["WIS_model"] / merged["WIS_ref"]
    return merged.groupby("horizon", observed=False)["skill"].mean().reset_index()


def run_backtest(
    inputs: pd.DataFrame,
    config: PipelineConfig,
    bt: Optional[BacktestConfig] = None,
    *,
    fit_fn: Optional[Callable] = None,
    reference_fit_fn: Optional[Callable] = None,
) -> BacktestResult:
    """Expand a forecast over a sequence of cutoff dates and score each window.

    Parameters
    ----------
    inputs:
        Prepared model inputs (the output of ``prepare_model_inputs``), with
        a Period ``Date`` column and observed ``Cases`` for the history.
    config:
        Pipeline config; ``horizons`` drive the scored windows.
    bt:
        Backtest knobs (model, cutoff sweep, window, reference model). Uses
        defaults when None.
    fit_fn:
        Optional ``(df, model_name, config, *, db_file=None) -> quantile frame``
        hook for the primary model (testing / custom drivers); defaults to
        :func:`thucia.core.pipeline.fit_model`.
    reference_fit_fn:
        Optional hook for the reference model used in skill scores; defaults
        to ``fit_fn`` (i.e. the real model driver).
    """
    bt = bt or BacktestConfig()

    if bt.fast_only and not _is_fast(bt.model_name):
        raise ValueError(
            f"model '{bt.model_name}' is not a fast backtest model "
            "(baseline/movavg); set fast_only=False to allow it."
        )
    if (
        bt.reference_model is not None
        and bt.fast_only
        and not _is_fast(bt.reference_model)
    ):
        raise ValueError(
            f"reference model '{bt.reference_model}' is not a fast backtest model "
            "(baseline/movavg); set fast_only=False to allow it."
        )

    horizons = list(config.horizons)
    max_horizon = max(horizons)
    freq = inputs["Date"].dt.freq
    observed = inputs[~inputs["future"]]
    dates = pd.PeriodIndex(observed["Date"].unique()).sort_values()

    if bt.cutoffs is not None:
        cutoffs = [pd.Period(c, freq=freq) for c in bt.cutoffs]
    else:
        cutoffs = expand_cutoffs(dates, bt.min_history, bt.step, max_horizon)

    fit = fit_fn if fit_fn is not None else _fit_model
    # The reference model always uses the real model driver unless a dedicated
    # reference_fit_fn is supplied (fit_fn is a test hook for the primary only).
    ref_fit = reference_fit_fn if reference_fit_fn is not None else _fit_model

    score_rows = []
    model_wis_by_cutoff: dict[pd.Period, pd.DataFrame] = {}
    forecasts: dict[pd.Period, pd.DataFrame] = {}
    # Fits may persist quantile duckdb files; keep them out of the caller's tree.
    with tempfile.TemporaryDirectory() as work_dir:
        for cutoff in cutoffs:
            slice_df = _holdout_for_cutoff(inputs, cutoff, max_horizon, bt.window)
            try:
                holdout = _fit_quantiles(
                    slice_df, bt.model_name, config, cutoff, fit, work_dir
                )
            except ValueError:
                continue
            if holdout.empty:
                continue

            cfg = dataclasses.replace(config, train_end_date=cutoff)
            scored = score_model(holdout, cfg, geo_col=config.geo_col)
            scored["cutoff"] = cutoff
            scored = scored.merge(
                _rmse_by_gid_horizon(holdout, config.geo_col).assign(cutoff=cutoff),
                on=["cutoff", config.geo_col, "horizon"],
                how="left",
            )
            score_rows.append(scored)
            model_wis_by_cutoff[cutoff] = (
                scored.groupby("horizon", observed=False)["WIS"]
                .mean()
                .reset_index()
                .assign(cutoff=cutoff)
            )
            if bt.keep_forecasts:
                forecasts[cutoff] = holdout

        if bt.reference_model is not None:
            skill = _skill_vs_reference(
                inputs, config, bt, cutoffs, model_wis_by_cutoff, ref_fit, work_dir
            )

    empty = pd.DataFrame(
        columns=[
            "cutoff",
            config.geo_col,
            "Date",
            "horizon",
            "WIS",
            "Cases",
            "R2",
            "RMSE",
        ]
    )
    if not score_rows:
        return BacktestResult(empty, _empty_summary(), forecasts or None)

    scores = pd.concat(score_rows, ignore_index=True)
    summary = (
        scores.groupby("horizon", observed=False)[["WIS", "RMSE", "R2"]]
        .mean()
        .reset_index()
    )
    summary["n"] = scores.groupby("horizon", observed=False).size().values

    if bt.reference_model is not None and not skill.empty:
        summary = summary.merge(skill, on="horizon", how="left")

    return BacktestResult(scores, summary, forecasts or None)


def _empty_summary() -> pd.DataFrame:
    return pd.DataFrame(columns=["horizon", "WIS", "RMSE", "R2", "n", "skill"])
