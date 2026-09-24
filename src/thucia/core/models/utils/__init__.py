import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from thucia.core.cases import period_freq_str

from ..quantiles import quantiles  # canonical grid; re-exported for compatibility
from .adapter import residual_regression as residual_regression
from .residual_quantiles import add_residual_quantiles as add_residual_quantiles


def season_length_for_freq(freq: str | None) -> int:
    """Number of periods per seasonal cycle for a pandas frequency string.

    Monthly -> 12, weekly (any anchor) -> 52, daily -> 365. Unknown or missing
    frequencies default to monthly (12).
    """
    if not freq:
        return 12
    f = str(freq)
    if f.startswith("W"):
        return 52
    if f.startswith("D"):
        return 365
    return 12


def sample_to_quantiles_vec(samples, quantiles=quantiles):
    samples = np.asarray(samples)
    q_values = np.quantile(samples, quantiles)
    return pd.DataFrame({"quantile": quantiles, "value": q_values})


def samples_to_quantiles(
    df: pd.DataFrame,
    quantiles=quantiles,
    geo_col: str = "GID_2",
    geo_parent: str = "GID_1",
    geo_parent_filter: list[str] | None = None,
) -> pd.DataFrame:
    """
    Convert samples in a DataFrame to quantiles using groupby for efficiency.
    Assumes 'prediction' contains multiple samples per date per region.
    """

    if "horizon" in df.columns:
        df_horizons = []
        for h in df["horizon"].unique().tolist():
            logging.info(f"Adding residual quantiles for horizon {h}")
            df_h = df[df["horizon"] == h].drop(columns=["horizon"])
            if not df_h.empty:
                df_q = samples_to_quantiles(
                    df_h,
                    quantiles=quantiles,
                    geo_col=geo_col,
                    geo_parent=geo_parent,
                    geo_parent_filter=geo_parent_filter,
                )
                df_q["horizon"] = h
                df_horizons.append(df_q)
        return pd.concat(df_horizons, ignore_index=True)

    # Parent-level filter
    if geo_parent_filter is not None:
        df = df[df[geo_parent].isin(geo_parent_filter)]

    # Group by region and horizon
    results = []
    for gid, group in df.groupby(geo_col, observed=True):
        logging.info(f"Processing region {gid} for quantiles conversion")
        group = group.sort_values("Date")  # ensure Date order
        dates = group["Date"].unique()

        # Map Date -> predictions for vectorized access
        date_groups = group.groupby("Date", observed=True)

        # Loop through date positions, skipping boundaries
        for k in range(2, len(dates) - 1):
            date = dates[k]
            predictions = date_groups.get_group(date)["prediction"].values

            # Apply quantile transform
            quantile_values = sample_to_quantiles_vec(
                predictions,
                quantiles,
            )["value"].values

            # Pull Cases once
            cases_val = date_groups.get_group(date)["Cases"].iloc[0]

            results.append(
                pd.DataFrame(
                    {
                        geo_col: gid,
                        "Date": date,
                        "quantile": quantiles,
                        "prediction": quantile_values,
                        "Cases": cases_val,
                    }
                )
            )

    logging.info("Quantiles conversion complete.")
    return pd.concat(results, ignore_index=True)


def filter_admin1(
    df: pd.DataFrame,
    geo_parent_filter: str | list[str] | None = None,
    geo_parent: str = "GID_1",
) -> pd.DataFrame:
    """
    Filter DataFrame by geo-parent region (e.g. admin-1).
    """
    if geo_parent_filter is not None:
        df = df[df[geo_parent].isin(geo_parent_filter)]
    return df.reset_index(drop=True)


def sanitize_dates_inplace(
    df: pd.DataFrame,
    date_col: str = "Date",
    start_date: pd.Timestamp | pd.Period | str = pd.Timestamp.min,
    end_date: pd.Timestamp | pd.Period | str = pd.Timestamp.max,
) -> pd.DataFrame:
    """
    Ensure the date column is in datetime format and at month-end.
    """
    freq = period_freq_str(df["Date"].dtype)
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    if isinstance(start_date, pd.Timestamp):
        start_date = start_date.to_period(freq)
    if isinstance(end_date, pd.Timestamp):
        end_date = end_date.to_period(freq)

    # Determine start and end dates
    start_date = max(start_date, df[date_col].min())
    end_date = min(end_date, df[date_col].max())
    date_range = pd.period_range(
        start=start_date,
        end=end_date,
        freq=freq,
    )
    return date_range


def validate_unique_keys(df: pd.DataFrame, col_names: list[str]) -> None:
    # Check that all (Date, geo unit) combinations are unique
    if df[col_names].duplicated().any():
        raise ValueError(
            "DataFrame contains duplicate (Date, geo unit) combinations. "
            "Ensure that the data is aggregated correctly."
        )


def interpolate_missing_dates(
    df,
    start_date: pd.Timestamp | str = pd.Timestamp.min,
    end_date: pd.Timestamp | str = pd.Timestamp.max,
    date_col: str = "Date",
    geo_col: str = "GID_2",
) -> None:
    # Get date range
    date_range = sanitize_dates_inplace(
        df,
        date_col=date_col,
        start_date=start_date,
        end_date=end_date,
    )
    validate_unique_keys(df, col_names=[date_col, geo_col])

    # Interpolate missing dates
    multi_index = pd.MultiIndex.from_product(
        [df[geo_col].unique(), date_range], names=[geo_col, date_col]
    )
    df = df.set_index([geo_col, date_col]).reindex(multi_index).reset_index()
    return df


def set_nan_zero(df: pd.DataFrame, col: str = "Cases", filter_col: str = "future"):
    if df[filter_col].isna().any():
        logging.warning(
            f"Column {filter_col} contains NaNs. Removing rows from dataset."
        )
        df = df[~df[filter_col].isna()]
        df.loc[:, filter_col] = df[filter_col].astype(bool)
    df.loc[~df[filter_col], col] = df.loc[~df[filter_col], col].fillna(0)


def set_historical_na_to_zero(
    df: pd.DataFrame, col: str = "Cases", filter_col: str = "future"
) -> pd.DataFrame:
    """
    Treat historical NAs as zero for the specified column.
    """
    df = df.copy()
    set_nan_zero(df, col=col, filter_col=filter_col)
    return df


def pca_transform(
    df,
    covariate_cols,
    keep_components: Optional[int] = 5,
    case_col="Cases",
    geo_col: str = "GID_2",
):
    # PCA transform covariates and project to fewer dimensions
    df_covs = df[["Date", geo_col] + covariate_cols].copy()
    logging.info("Fit PCA")
    pca = PCA(n_components=min(keep_components, len(covariate_cols)))
    logging.info("Transform covariates by PCA")
    covs_transformed = pca.fit_transform(df_covs[covariate_cols])
    df = df.drop(columns=covariate_cols)
    covariate_cols = [f"PC{i + 1}" for i in range(covs_transformed.shape[1])]
    df[covariate_cols] = covs_transformed
    df = df[["Date", geo_col, "future", "Cases", case_col] + covariate_cols]
    return df


def sanitise_covariates(df, covariate_cols, start_date, geo_col="GID_2"):
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(start_date, pd.Timestamp):
        freq = period_freq_str(df["Date"].dtype)
        start_date = start_date.to_period(freq)
    if not start_date:
        start_date = df["Date"].max()

    # Covariate sanitisation
    for c in covariate_cols:
        # NaN replacement: seasonal mean, forward and back fill
        df[c] = df.groupby([geo_col, df["Date"].dt.month], observed=False)[c].transform(
            lambda s: s.fillna(s.mean())
        )
        df[c] = df.groupby(geo_col, observed=False)[c].ffill().bfill()
        # Standardise using pre- start date values
        mask = df["Date"] < start_date
        if False:
            mu = df[mask][c].mean()
            sd = np.max([1e-8, df[mask][c].std()])
            df[c] = (df[c] - mu) / sd
        if True:
            # norm = np.abs(df[mask][c].mean())
            norm = np.abs(df[mask][c]).mean()  # scale by mean absolute value
            if norm > 1e-3:
                df[c] = df[c] / norm
        # Sanity check
        if df[c].isna().any():
            raise Exception("NaN found in covariates")
    return df


def aggregate_to_admin1(
    df: pd.DataFrame,
    weight_col: None,
    geo_col: str = "GID_2",
    geo_parent: str = "GID_1",
):
    # Aggregate to geo-parent level (e.g. admin-1) by summing Cases and averaging covariates
    df = df.drop(columns=[geo_col])
    # Weight covars
    if weight_col:
        df["weight"] = df.groupby(["Date", geo_parent])[weight_col].transform(
            lambda x: x / x.sum()
        )
        covars = df.columns.difference(
            ["Date", geo_parent, "Cases", "future", "weight"]
        )
        for c in covars:
            df[c] = df[c] * df["weight"]

    df = (
        df.groupby(["Date", geo_parent], observed=True)
        .agg(
            Cases=("Cases", "sum"),
            future=("future", "first"),
            **{
                c: (c, "sum")
                for c in df.columns
                if c not in ["Date", geo_parent, "Cases", "future"]
            },
        )
        .reset_index()
    )
    if "Log_Cases" in df.columns:
        df["Log_Cases"] = np.log1p(df["Cases"])
    df = df.drop(columns=["weight"], errors="ignore")

    return df
