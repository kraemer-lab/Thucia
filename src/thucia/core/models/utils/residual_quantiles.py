import logging
from collections import deque
from pathlib import Path
from typing import List
from typing import Sequence

import numpy as np
import pandas as pd
from thucia.core.fs import DataFrame

from ..quantiles import quantiles  # canonical grid; do not redefine


def add_residual_quantiles(
    df: pd.DataFrame,
    *,
    date_col: str = "Date",
    geo_col: str = "GID_2",
    y_col: str = "Cases",
    pred_col: str = "prediction",
    horizon_col: str = "horizon",
    quantile_levels: Sequence[float] = quantiles,
    window: int = None,  # None => expanding; integer => moving window size
    min_history: int = 20,
    pool_across_gids_if_sparse: bool = True,
    db_file: str = None,
) -> pd.DataFrame:
    """
    Post-hoc predictive distribution from deterministic forecasts via residual quantiles.
    Returns a long dataframe with columns:
      [Date, geo_col, horizon, quantile, prediction]
    """
    if horizon_col not in df.columns:
        # df = df.copy()
        df[horizon_col] = 1
    else:
        # df = df.copy()
        ...

    # Output dataframe
    tdf = (
        DataFrame(db_file=Path(db_file), new_file=True)
        if db_file
        else DataFrame()  # fallback to in-memory DataFrame
    )

    # Log transform
    df["prediction"] = np.log1p(df["prediction"])
    df["Cases"] = np.log1p(df["Cases"])

    # If your point predictions have NaNs, we can’t recover—surface early.
    if df[pred_col].isna().all():
        # Nothing to do—return empty result to make the problem obvious.
        return pd.DataFrame(
            columns=[date_col, geo_col, horizon_col, "quantile", pred_col]
        )

    # Keep only rows with valid dates and predictions
    df = df[~df[date_col].isna()]  # .copy()

    for h in sorted(df[horizon_col].unique()):
        logging.info(f"Processing horizon {h}")
        out_records = []

        dfh = df[df[horizon_col] == h].copy()
        # Sort deterministically for causal pass
        dfh = dfh.sort_values([date_col, geo_col], kind="mergesort")

        hist_by_gid: dict[str, deque] = {}
        pooled_hist: deque = deque()

        dates = dfh[date_col].values
        unique_dates = np.unique(dates)

        for d in unique_dates:
            logging.info(f"  Processing date {d}")
            day_mask = dates == d
            df_apply = dfh.loc[day_mask]

            preds_today = df_apply[pred_col].to_numpy(dtype=float)
            gids_today: List[str] = df_apply[geo_col].astype(str).tolist()

            # ---- APPLY STEP (use only past residuals; history buffers track that) ----
            for i, g in enumerate(gids_today):
                # fallback is the point prediction (never NaN if input isn’t)
                base = preds_today[i]

                ghist = hist_by_gid.get(g, deque())
                if len(ghist) >= min_history:
                    resids = np.fromiter(ghist, dtype=float)
                elif pool_across_gids_if_sparse and len(pooled_hist) >= min_history:
                    resids = np.fromiter(pooled_hist, dtype=float)
                else:
                    resids = None

                if resids is not None and resids.size > 0:
                    # quantiles of residuals + base prediction
                    qs = np.quantile(resids, quantile_levels, method="linear")
                    preds_q = base + qs
                else:
                    # not enough history yet -> use the base prediction at all quantiles
                    preds_q = np.full(len(quantile_levels), base, dtype=float)

                for q, v in zip(quantile_levels, preds_q):
                    out_records.append(
                        {
                            date_col: d,
                            geo_col: g,
                            horizon_col: h,
                            "quantile": float(q),
                            pred_col: np.expm1(float(v)).clip(min=0.0),
                            y_col: df_apply[y_col].iloc[i],
                            "Cases": np.expm1(df_apply["Cases"].iloc[i]).clip(min=0.0),
                        }
                    )

            # ---- UPDATE STEP: after applying to day d, reveal truth at d and update histories ----
            if y_col in df_apply.columns:
                # If y is missing for some rows, skip those residuals
                y_vals = df_apply[y_col].to_numpy(dtype=float)
                valid_mask = ~np.isnan(y_vals) & ~np.isnan(preds_today)
                for g, r in zip(
                    np.array(gids_today)[valid_mask], (y_vals - preds_today)[valid_mask]
                ):
                    qbuf = hist_by_gid.get(g)
                    if qbuf is None:
                        qbuf = deque()
                        hist_by_gid[g] = qbuf
                    qbuf.append(float(r))
                    pooled_hist.append(float(r))
                    if window is not None:
                        while len(qbuf) > window:
                            qbuf.popleft()
                        while len(pooled_hist) > window:
                            pooled_hist.popleft()

        tdf.append(pd.DataFrame.from_records(out_records))

    return tdf
