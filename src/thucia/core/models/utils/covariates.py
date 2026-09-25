from __future__ import annotations

from typing import Any
from typing import Dict
from typing import Iterable
from typing import List

import numpy as np
import pandas as pd


def _apply_pipeline_to_group(s: pd.Series, steps: List[Dict[str, Any]]) -> pd.Series:
    out = s
    for step in steps:
        op = step["op"]
        if op == "shift":
            out = out.shift(step.get("periods", 1))
        elif op == "rolling":
            window = step["window"]
            min_p = step.get("min_periods", window)
            agg = step.get("agg", "mean")
            out = getattr(out.rolling(window=window, min_periods=min_p), agg)()
        elif op == "diff":
            out = out.diff(step.get("periods", 1))
        elif op == "pct_change":
            out = out.pct_change(step.get("periods", 1))
        elif op == "ema":
            span = step["span"]
            out = out.ewm(span=span, adjust=False).mean()
        elif op == "fillna":
            out = out.fillna(step.get("value", 0))
        elif op == "clip":
            out = out.clip(lower=step.get("lower"), upper=step.get("upper"))
        elif op == "lambda":
            func = step["func"]
            out = func(out)
        else:
            raise ValueError(f"Unknown op: {op}")
    return out


def build_features(df: pd.DataFrame, specs: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    df = df.copy()
    for spec in specs:
        name = spec["name"]
        keys = spec.get("groupby", None)
        col = spec["column"]
        steps = spec["pipeline"]

        # per-group apply ensures rolling windows don’t bleed across groups
        def _one_group(g: pd.DataFrame) -> pd.Series:
            return _apply_pipeline_to_group(g[col], steps)

        # group_keys=False keeps the original index; result aligns with df
        if keys:
            series = df.groupby(keys, group_keys=False, observed=False).apply(
                _one_group, include_groups=False
            )
        else:
            series = _one_group(df)

        # if groupby returns MultiIndex in edge cases, align back
        series = series.reindex(df.index)

        df[name] = series
    return df


def prepare_covariates(
    df,
    geo_col: str = "GID_2",
):
    df = df.copy()
    case_col = "Log_Cases"

    # Pre-compute covariates
    df["MONTH"] = pd.to_datetime(df["Date"]).dt.month.astype(int)

    features = [
        {
            "name": case_col,
            "column": "Cases",
            "pipeline": [{"op": "lambda", "func": np.log1p}],
        },
        {
            "name": "lag_1_log_cases",
            "groupby": [geo_col],
            "column": "Log_Cases",
            "pipeline": [{"op": "shift", "periods": 1}],
        },
        {
            "name": "tmin_roll_2",
            "groupby": [geo_col],
            "column": "tmin",
            "pipeline": [
                {"op": "rolling", "window": 2},
            ],
        },
        {
            "name": "lag_1_tmin_roll_2",
            "groupby": [geo_col],
            "column": "tmin_roll_2",
            "pipeline": [{"op": "shift", "periods": 1}],
        },
        {
            "name": "prec_roll_2",
            "groupby": [geo_col],
            "column": "prec",
            "pipeline": [
                {"op": "rolling", "window": 2},
            ],
        },
        {
            "name": "lag_1_prec_roll_2",
            "groupby": [geo_col],
            "column": "prec_roll_2",
            "pipeline": [{"op": "shift", "periods": 1}],
        },
    ]

    df = build_features(df, features)
    covariate_cols = [
        "lag_1_log_cases",
        "lag_1_tmin_roll_2",
        "lag_1_prec_roll_2",
    ]

    return df, case_col, covariate_cols
