import logging
from pathlib import Path
from typing import List
from typing import Optional

import numpy as np
import pandas as pd
import torch
from darts import TimeSeries
from thucia.core.cases import align_date_types
from thucia.core.cases import period_freq_str
from thucia.core.fs import DataFrame
from thucia.core.models.utils import quantiles as default_quantiles
from thucia.core.models.utils import sample_to_quantiles_vec

torch.set_float32_matmul_precision(
    "medium"
)  # medium=bfloat, high=tfloat, highest=float32


class DartsBase:
    def __init__(
        self,
        df: pd.DataFrame,
        case_col: str = "Cases",
        date_col="Date",
        geo_col="GID_2",
        geo_parent: Optional[str] = "GID_1",
        covariate_cols: Optional[List[str]] = None,
        horizons=[1],
        num_samples: int | None = None,
        db_file: str | Path | None = None,
        train_start_date=None,
        train_end_date=None,
        multivariate: bool = False,
        quantiles: List[float] | None = None,
    ):
        self.df = df
        self.case_col = case_col
        self.date_col = date_col
        self.geo_col = geo_col
        self.geo_parent = geo_parent
        self.covariate_cols = covariate_cols or []
        self.horizons = horizons
        self.num_samples = num_samples or 1000
        self.db_file = Path(db_file) if db_file else None
        self.multivariate = multivariate
        self.quantiles = quantiles or default_quantiles
        self.fit_delta = False

        # Models assume the geo column is categorical (e.g. multivariate
        # encoding via .cat.codes, and stable category ordering across geo
        # units). Coerce on the local copy if the caller supplied plain strings.
        if isinstance(self.df, pd.DataFrame) and self.geo_col in self.df.columns:
            if not isinstance(self.df[self.geo_col].dtype, pd.CategoricalDtype):
                self.df = self.df.copy()
                self.df[self.geo_col] = self.df[self.geo_col].astype("category")

        gid_codes_col = f"{self.geo_col}_codes"
        if self.multivariate and gid_codes_col not in self.covariate_cols:
            self.covariate_cols.append(gid_codes_col)  # added in get_cases

        if self.db_file:
            logging.debug(f"Darts model initialized with file store: {self.db_file}")
        else:
            logging.debug("Darts model initialized without file store.")

        if len(self.quantiles) == self.num_samples:
            raise ValueError(
                "num_samples cannot be equal to the number of quantiles; "
                "please set num_samples to None or a different value."
            )

        # Training period
        if train_start_date is None:
            train_start_date = pd.Timestamp.min
        if train_end_date is None:
            train_end_date = pd.Timestamp.max
        train_start_date = max(
            align_date_types(train_start_date, self.df[self.date_col]),
            self.df[self.date_col].min(),
        )
        train_end_date = min(
            align_date_types(train_end_date, self.df[self.date_col]),
            self.df[self.date_col].max(),
        )
        self.train_start_date = train_start_date
        self.train_end_date = train_end_date

        # Skip training / forecasting in GID_2s with all zeros in training period
        self.identify_noincidence_regions()

        # Parameters and functionality provided by subclasses
        self.sampling_method = None

    def identify_noincidence_regions(self):
        # Reject GID_2 with all zeros in training period
        def has_nonzero_cases(gdf):
            gdf_train = gdf[
                (gdf[self.date_col] >= self.train_start_date)
                & (gdf[self.date_col] <= self.train_end_date)
            ]
            return (gdf_train[self.case_col] > 0).any()

        self.valid_gids = (
            self.df.groupby(self.geo_col, observed=False)
            .filter(has_nonzero_cases)[self.geo_col]
            .unique()
        )
        self.rejected_gids = set(self.df[self.geo_col].unique()) - set(self.valid_gids)
        if self.rejected_gids:
            logging.info(
                f"Excluding {len(self.rejected_gids)} {self.geo_col} regions with no "
                "incidence in training period."
            )

    # Child classes must override this method to provide concrete functionality
    def build_model(self):
        raise NotImplementedError

    def get_cases(
        self,
        future=None,
        target_gids: Optional[List[str]] = None,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
    ):
        if future is None:
            # Use all data
            df = self.df
        elif future:
            df = self.df[self.df["future"]]
        else:
            df = self.df[~self.df["future"]]

        if target_gids is None:
            target_gids = df[self.geo_col].unique()

        if start_date is None:
            start_date = df[self.date_col].min()
        if end_date is None:
            end_date = df[self.date_col].max()

        # Add geo units as numeric code for multivariate encoding
        if self.multivariate:
            gid_codes_col = f"{self.geo_col}_codes"
            if gid_codes_col not in df.columns:
                df = df.assign(**{gid_codes_col: df[self.geo_col].cat.codes})
            if gid_codes_col not in self.covariate_cols:
                self.covariate_cols.append(gid_codes_col)

        start_date = align_date_types(start_date, df[self.date_col])
        end_date = align_date_types(end_date, df[self.date_col])

        target_list = []
        covar_list = []
        # darts needs a timestamp frequency; derive it from the Period dtype so
        # weekly/daily cadences keep their anchor (fallback: month-end).
        if isinstance(df[self.date_col].dtype, pd.PeriodDtype):
            freq = period_freq_str(df[self.date_col].dtype)
            if freq == "M":
                freq = "ME"  # darts' timestamp alias for month-end
        else:
            freq = "ME"
        for gid in target_gids:
            gdf = df[
                (df[self.geo_col] == gid)
                & (df["Date"] >= start_date)
                & (df["Date"] <= end_date)
            ].copy()
            # darts requires timestamp
            gdf["Date"] = (
                gdf["Date"].dt.to_timestamp(how="end").dt.normalize().astype("<M8[ns]")
            )

            if self.fit_delta:
                gdf["Log_Cases"] = gdf["Log_Cases"].diff()

            ts = TimeSeries.from_dataframe(
                gdf,
                time_col="Date",
                value_cols=["Log_Cases"],
                fill_missing_dates=True,
                freq=freq,
            ).astype(np.float32)

            cov = TimeSeries.from_dataframe(
                gdf,
                time_col="Date",
                value_cols=self.covariate_cols,
                fill_missing_dates=True,
                freq=freq,
            ).astype(np.float32)

            target_list.append(ts)
            covar_list.append(cov)

        return target_list, covar_list, target_gids

    def _historical_predictions_per_region(
        self,
        *,
        tdf_out: DataFrame,
        horizon: int,
        retrain: bool = True,  # only turn off for faster testing
        start_date: pd.Timestamp | None = None,
        geo_col: str | None = None,
    ) -> DataFrame | pd.DataFrame:
        geo_col = geo_col or self.geo_col
        gid_list = self.df[geo_col].unique().tolist()
        for ix, gid in enumerate(gid_list):
            logging.info(f"Processing {geo_col}: {gid}...")
            tic = pd.Timestamp.now()
            df_gid = self.df[self.df[geo_col] == gid].copy()

            # predictions are always pd.DataFrame
            self._historical_predictions_onepass(
                df=df_gid,
                tdf_out=tdf_out,
                retrain=retrain,
                start_date=start_date,
                horizon=horizon,
            )

            # Estimate time remaining
            toc = pd.Timestamp.now()
            logging.info(f"Completed {geo_col}: {gid} in {toc - tic}.")
            estimated_time_remaining = (toc - tic) * (len(gid_list) - ix - 1)
            logging.info(f"Estimated time remaining: {estimated_time_remaining}.")
        return tdf_out

    def historical_predictions(
        self,
        *,
        train_col: str | None = None,
        retrain: bool = True,  # only turn off for faster testing
        start_date: pd.Timestamp | None = None,
    ) -> DataFrame | pd.DataFrame:
        """
        Pre-fits on all regions, then generates historical forecasts for each
        region separately.

        ``train_col`` names the column to fit per-region at (e.g. ``"GID_1"``
        or ``"GID_2"``). When ``None``, the whole dataset is trained in a
        single pass (country-level fit).
        """
        if train_col is None:
            logging.info("Generating historical predictions on the whole dataset.")
        else:
            logging.info(
                f"Generating historical predictions per region grouped by {train_col}."
            )

        tdf = (
            DataFrame(db_file=Path(self.db_file), new_file=True)
            if self.db_file
            else DataFrame()  # fallback to in-memory DataFrame
        )

        if train_col is None:  # Train on entire dataset
            for horizon in self.horizons:
                self._historical_predictions_onepass(
                    tdf_out=tdf,
                    retrain=retrain,
                    start_date=start_date,
                    horizon=horizon,
                )
        else:  # Train per region grouped by train_col
            for horizon in self.horizons:
                self._historical_predictions_per_region(
                    tdf_out=tdf,
                    retrain=retrain,
                    start_date=start_date,
                    geo_col=train_col,
                    horizon=horizon,
                )

        return tdf

    def _merge_cases(
        self,
        df: pd.DataFrame,
        preds: pd.DataFrame,
    ) -> pd.DataFrame:
        # Ensure Date is in original format
        freq = period_freq_str(df["Date"].dtype)
        if not isinstance(preds["Date"].dtype, pd.PeriodDtype):
            # Coerce to period
            preds["Date"] = preds["Date"].dt.to_period(freq)
        elif preds["Date"].dtype.freq.freqstr != freq:
            # Coalesce frequency
            preds["Date"] = preds["Date"].dt.asfreq(freq)

        # Merge Cases back in to preds
        preds = preds.merge(
            df[["Date", self.geo_col, "Log_Cases"]],
            on=["Date", self.geo_col],
            how="left",
        )
        # Restore GID categories
        preds[self.geo_col] = pd.Categorical(
            preds[self.geo_col],
            categories=df[self.geo_col].cat.categories,
            ordered=df[self.geo_col].cat.ordered,
        )
        # Return Cases to original scale
        preds["Cases"] = np.expm1(preds["Log_Cases"]).clip(lower=0)
        return preds

    def _historical_predictions_onepass(
        self,
        *,
        horizon: int,
        df: pd.DataFrame | None = None,
        tdf_out: DataFrame,
        retrain: bool = True,  # only turn off for faster testing
        start_date: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """
        Pre-fits on all regions, then generates historical forecasts for each region
        separately.
        """
        df = df if df is not None else self.df  # use provided df, fallback to self.df
        self.model = self.build_model(horizon=horizon)

        if start_date is None:
            start_date = df["Date"].min()

        # Convert floats to 32-bit precision
        float_cols = df.select_dtypes(include="float").columns
        df[float_cols] = df[float_cols].astype(np.float32)

        # Model pre-fit
        all_target_gids = df[self.geo_col].unique()
        target_gids = [gid for gid in all_target_gids if gid not in self.rejected_gids]
        self.pre_fit(target_gids=target_gids)

        # Now include future data for forecasting, ensuring the same GID mapping
        target_list, covar_list, _ = self.get_cases(target_gids=target_gids)

        if self.multivariate:
            logging.info(f"Forecasting for {len(target_gids)} regions (multivariate)")
            tic = pd.Timestamp.now()
            start_date_timestamp = start_date.to_timestamp(how="end")
            try:
                bt = self.historical_forecasts(
                    target_list,
                    covar_list,
                    start_date=start_date_timestamp,
                    retrain=retrain,
                    horizon=horizon,
                )
                # for multivariate, bt is a list per time-series:
                # bt = [gid][time]series[horizon][1][samples]
            except ValueError as e:
                logging.warning(
                    f"Failed to fit for {self.geo_col} {target_gids} (multivariate) "
                    f"(msg: {e}), skipping..."
                )
                self.rejected_gids.update(target_gids)
            for ix, gid in enumerate(target_gids):
                out = bt[ix].to_dataframe().reset_index(names="Date")
                out = out.melt(
                    id_vars="Date",
                    var_name="quantile",
                    value_name="prediction",
                )
                out["quantile"] = (
                    out["quantile"].str.split(".").str[-1].astype(float) / 1000
                )
                out["horizon"] = horizon
                out[self.geo_col] = gid
                out["prediction"] = np.expm1(out["prediction"]).clip(lower=0)
                tdf_out.append(self._merge_cases(df, out))
            toc = pd.Timestamp.now()
            logging.info(f"Regions {target_gids} done in {toc - tic}")
        else:
            for ts, cov, gid in zip(target_list, covar_list, target_gids):
                logging.info(f"Forecasting for {self.geo_col} {gid}")
                tic = pd.Timestamp.now()
                start_date_timestamp = start_date.to_timestamp(how="end")
                try:
                    bt = self.historical_forecasts(
                        ts,
                        cov,
                        gid=gid,
                        start_date=start_date_timestamp,
                        retrain=retrain,
                        horizon=horizon,
                    )
                    # for univariate, bt relates to a single time-series:
                    # bt = TimeSeries[time][quantiles][1]
                except ValueError as e:
                    logging.warning(
                        f"Failed to fit for {self.geo_col} {gid} (msg: {e}), skipping..."
                    )
                    self.rejected_gids.add(gid)
                    continue
                out = bt.to_dataframe().reset_index(names="Date")
                out = out.melt(
                    id_vars="Date",
                    var_name="var",
                    value_name="prediction",
                )
                out["horizon"] = horizon
                out[self.geo_col] = gid
                if out["var"].str.contains("_s").any():
                    # Sample-based output (e.g. darts ARIMA with num_samples):
                    # collapse samples to the canonical quantile grid.
                    qparts = []
                    for date, g in out.groupby("Date"):
                        s2q = sample_to_quantiles_vec(
                            np.clip(np.expm1(g["prediction"].to_numpy()), 0, None),
                            self.quantiles,
                        )
                        qparts.append(
                            pd.DataFrame(
                                {
                                    "Date": date,
                                    "horizon": horizon,
                                    self.geo_col: gid,
                                    "quantile": s2q["quantile"],
                                    "prediction": s2q["value"],
                                }
                            )
                        )
                    out = pd.concat(qparts, ignore_index=True)
                else:
                    out["quantile"] = (
                        out["var"].str.split(".").str[-1].astype(float) / 1000
                    )
                    out = out.drop(columns=["var"])
                    if self.fit_delta:
                        out["prediction"] = out["prediction"].cumsum()
                    out["prediction"] = np.expm1(out["prediction"]).clip(lower=0)
                tdf_out.append(self._merge_cases(df, out))
                toc = pd.Timestamp.now()
                logging.info(f"Region {gid} done in {toc - tic}")

        # Substitute back all-zero regions
        #
        # Re-compute target_gids to account for new entried due to fitting errors; we do
        # not use self.rejected_gids directly as it is not specific to the geographic
        # region being assessed.
        # target_gids = [gid for gid in all_target_gids if gid not in self.rejected_gids]
        # if self.rejected_gids:
        #     logging.info(
        #         f"Adding all-zero predictions for {len(self.rejected_gids)} "
        #         "GID_2s with no incidence in training period, or estimation errors..."
        #     )
        #     logging.info(f"Rejected GID_2s: {self.rejected_gids}")
        #     dates = df["Date"].unique()
        #     dates = dates[dates >= start_date]
        #     dates = dates.to_timestamp(how="end")
        #     dates = np.sort(dates)
        # for gid in set(all_target_gids) - set(target_gids):
        #     logging.info(f"Adding all-zero predictions for GID_2 {gid}...")
        #     # Create rows with predictions equal to zero
        #     out = pd.DataFrame(
        #         [
        #             (d, gid, q, 0, h + 1)
        #             for d in dates
        #             for h in range(self.horizon)
        #             for q in self.quantiles
        #         ],
        #         columns=["Date", "GID_2", "quantile", "prediction", "horizon"],
        #     )
        #     # Remove predictions outside forecasting range (to match other regions)
        #     for ix, date in enumerate(dates):
        #         if ix < self.horizon:
        #             # Remove predictions before horizon available
        #             out = out[~((out["Date"] == date) & (out["horizon"] > ix + 1))]
        #         if len(dates) - ix <= self.horizon:
        #             # Remove predictions after data available
        #             out = out[
        #                 ~(
        #                     (out["Date"] == date)
        #                     & (out["horizon"] <= self.horizon - len(dates) + ix)
        #                 )
        #             ]
        #     # Add to database
        #     tdf_out.append(self._merge_cases(df, out))

        return tdf_out
