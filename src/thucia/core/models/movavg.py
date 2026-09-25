import logging

import numpy as np
import pandas as pd
from thucia.core.cases import align_date_types
from thucia.core.cases import period_freq_str

from ._meta import ModelSpec

SPEC = ModelSpec(name="movavg", family="statistical", fast=True)


def _season_unit(dates: pd.Series, freq: str) -> pd.Series:
    """Season unit for a pandas frequency: month, ISO week, or day-of-year.

    `dates` is expected to be a datetime64 series (periods already converted to
    timestamps).
    """
    if freq.startswith("W"):
        return dates.dt.isocalendar()["week"].astype(int)
    if freq.startswith("D"):
        return dates.dt.dayofyear
    return dates.dt.month


def movavg(
    df: pd.DataFrame,
    start_date: pd.Timestamp | pd.Period = None,
    end_date: pd.Timestamp | pd.Period = None,
    geo_col: str = "GID_2",
    geo_parent: str | None = "GID_1",
    geo_parent_filter: list[str] | None = None,
    method: str = "historical",  # historical / predict
    *args,
    **kwargs,
) -> pd.DataFrame:
    logging.info("Starting Seasonal Moving Average model...")

    # Parent-level filter
    if geo_parent is not None and geo_parent_filter is not None:
        df = df[df[geo_parent].isin(geo_parent_filter)]

    # Determine start and end dates
    if start_date is None:
        start_date = pd.Timestamp.min
    if end_date is None:
        end_date = pd.Timestamp.max
    start_date = max(
        align_date_types(start_date, df["Date"]),
        df["Date"].min(),
    ).to_timestamp(how="end")
    end_date = min(
        align_date_types(end_date, df["Date"]),
        df["Date"].max(),
    ).to_timestamp(how="end")

    # Interpolate date range, ensuring we don't skip any gaps in the data
    freq = period_freq_str(df["Date"].dtype)
    freq_ts = "ME" if freq == "M" else freq  # pandas timestamp alias for month-end
    date_range = pd.date_range(
        start=start_date - pd.DateOffset(years=5),  # need 5 years of history
        end=end_date,
        freq=freq_ts,
    )

    # Combine Cases over Status=Confirmed, Probable
    df = (
        df.groupby(["Date", geo_col], observed=True)
        .agg({"Cases": "sum", "future": "first"})
        .reset_index()
    )
    df["Date"] = df["Date"].dt.to_timestamp(how="end")

    # Interpolate missing dates
    multi_index = pd.MultiIndex.from_product(  # <-- implicit conversion to Timestamp
        [df[geo_col].unique(), date_range], names=[geo_col, "Date"]
    )
    df = df.set_index([geo_col, "Date"]).reindex(multi_index).reset_index()

    df["Cases"] = df["Cases"].fillna(0)

    df_forecast = df.copy()
    df_forecast["Year"] = df_forecast["Date"].dt.year
    df_forecast["Season"] = _season_unit(df_forecast["Date"], freq)

    df_forecast = df_forecast.groupby(
        [geo_col, "Year", "Season"], observed=True, as_index=False
    ).agg({"Cases": "mean", "future": "first", "Date": "first"})

    df_forecast["prediction"] = (
        df_forecast.groupby([geo_col, "Season"], observed=True)["Cases"]
        .apply(lambda s: s.shift(1).rolling(window=5, min_periods=5).mean())
        .reset_index(level=[0, 1], drop=True)
    )
    df_forecast["sample"] = 0

    df_forecast["Date"] = df_forecast["Date"].dt.to_period(freq)
    df_forecast.drop(columns=["Year", "Season"], inplace=True)

    df_forecast = df_forecast[df_forecast["Date"] >= start_date.to_period(freq)]
    df_forecast.loc[df_forecast["future"], "Cases"] = np.nan

    logging.info("Seasonal Moving Average model complete.")
    return df_forecast
