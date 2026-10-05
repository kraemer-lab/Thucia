# Adapt wide, pre-aggregated case panels to the forecasting pipeline.
#
# ``cases_per_period`` covers the line-list path (raw case rows -> period totals
# + covariate merges). This module covers the *panel* path: data that already
# arrives aggregated to one row per (region, period) with covariates attached.
# ``panel_to_frame`` maps such a panel onto the schema the pipeline expects
# (``Date``, geo columns, ``Cases``, ``future``); ``read_rds`` loads an R
# data.frame saved as ``.rds``/``.rda`` via the optional ``rdata`` dependency.
from __future__ import annotations

from os import PathLike
from pathlib import Path

import pandas as pd
from thucia.core.fs import DataFrame


def read_rds(path: str | PathLike, *, name: str | None = None) -> pd.DataFrame:
    """Read an R ``.rds``/``.rda`` object into a pandas DataFrame.

    Requires the optional ``rdata`` package (``pip install thucia[panels]``).
    ``name`` selects a named object when the file holds more than one.
    """
    try:
        import rdata
    except ImportError as exc:  # pragma: no cover - exercised via error path
        raise ImportError(
            "read_rds requires the optional 'rdata' package; install it with "
            "`pip install thucia[panels]`."
        ) from exc

    parsed = rdata.parser.parse_file(str(Path(path).expanduser()))
    converted = rdata.conversion.convert(parsed)
    if isinstance(converted, dict):
        if name is None:
            if len(converted) != 1:
                raise ValueError(
                    f"{path!r} holds multiple objects "
                    f"({list(converted)}); pass name= to select one."
                )
            name = next(iter(converted))
        converted = converted[name]
    if not isinstance(converted, pd.DataFrame):
        raise TypeError(
            f"Expected an R data.frame in {path!r}, got {type(converted).__name__}."
        )
    return converted


def panel_to_frame(
    df,
    *,
    cases_col: str,
    geo_col: str,
    geo_parent: str | None = None,
    date_col: str | None = None,
    index_col: str | None = None,
    anchor=None,
    freq: str = "W-SAT",
    covariate_cols=None,
    pop_col: str | None = None,
    future: bool = False,
) -> pd.DataFrame:
    """Map a wide, pre-aggregated panel onto the pipeline's model-input schema.

    Exactly one of ``date_col`` (an existing period/datetime column) or
    ``index_col`` + ``anchor`` (a contiguous integer period index mapped onto
    ``pd.period_range(anchor, ...)``) names the time axis — the latter suits
    source panels whose calendar weeks are irregular (e.g. epi weeks with
    week 53). ``cases_col`` becomes ``Cases``; ``geo_col``/``geo_parent`` are
    passed through; ``covariate_cols`` and ``pop_col`` (emitted as
    ``pop_count``) are carried alongside. NA outcomes are preserved.

    The panel must already be unique on ``(Date, geo_col)``; a ``ValueError``
    is raised otherwise.
    """
    if isinstance(df, DataFrame):
        df = df.df
    else:
        df = df.copy()

    if date_col is not None and index_col is not None:
        raise ValueError("Pass only one of date_col= or index_col= (+ anchor=).")
    if date_col is None and index_col is None:
        if "Date" in df.columns:
            date_col = "Date"
        else:
            raise ValueError(
                "Pass date_col= (an existing period/datetime column) or "
                "index_col= + anchor=."
            )
    if geo_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{geo_col}' column.")
    if cases_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{cases_col}' column.")

    if index_col is not None:
        if anchor is None:
            raise ValueError("index_col= requires anchor= (start period).")
        idx = df[index_col]
        if not pd.api.types.is_integer_dtype(idx) or (idx < 1).any():
            raise ValueError(f"'{index_col}' must be positive integers from 1.")
        periods = pd.period_range(anchor, periods=int(idx.max()), freq=freq)
        dates = periods[idx.to_numpy() - 1]
    else:
        if not isinstance(df[date_col].dtype, pd.PeriodDtype):
            df[date_col] = pd.to_datetime(df[date_col]).dt.to_period(freq)
        dates = df[date_col]

    out = pd.DataFrame(
        {
            "Date": pd.PeriodIndex(dates),
            geo_col: df[geo_col].to_numpy(),
            "Cases": df[cases_col].to_numpy(),
            "future": future,
        }
    )
    if geo_parent is not None:
        if geo_parent not in df.columns:
            raise ValueError(f"DataFrame must contain '{geo_parent}' column.")
        out[geo_parent] = df[geo_parent].to_numpy()
    for c in covariate_cols or []:
        if c not in df.columns:
            raise ValueError(f"DataFrame must contain covariate '{c}'.")
        out[c] = df[c].to_numpy()
    if pop_col is not None:
        out["pop_count"] = df[pop_col].to_numpy()

    if out.duplicated(["Date", geo_col]).any():
        raise ValueError(
            f"Panel is not unique on ('Date', {geo_col!r}); aggregate it first."
        )
    return out.sort_values(["Date", geo_col]).reset_index(drop=True)
