import logging
from pathlib import Path
from typing import List
from typing import Optional

import pandas as pd
from darts import TimeSeries
from darts.models import Chronos2Model
from darts.utils.likelihood_models import QuantileRegression
from thucia.core.fs import DataFrame

from ._meta import ModelSpec
from .darts import DartsBase

SPEC = ModelSpec(
    name="chronos",
    family="darts",
    supports=frozenset({"train_end_date", "retrain", "multivariate"}),
    sampling="quantiles",
    extras=("chronos",),
)

try:  # Native chronos package (alternative available through Darts)
    from chronos import Chronos2Pipeline
except ImportError:
    Chronos2Pipeline = None

# Chronos (native) supports a fixed subset of quantile levels, so this is a
# documented subset of the canonical grid in thucia.core.quantiles.
CHRONOS_QUANTILES = [
    0.01,
    0.05,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    0.95,
    0.99,
]


# -------- Chronos (native implementation) --------
class ChronosQuantilesNative(DartsBase):
    def __init__(self, *args, **kwargs):
        if not Chronos2Pipeline:
            raise ImportError(
                "Chronos package not found. "
                "Please install Chronos to use ChronosQuantilesNative model."
            )

        super().__init__(*args, **kwargs)
        self.sampling_method = "quantiles"

    def build_model(self):
        return Chronos2Pipeline.from_pretrained(
            "amazon/chronos-2",
        )

    def pre_fit(self, target_gids=None, **kwargs):
        pass

    def historical_forecasts(
        self, ts, cov, start_date=None, retrain=True, horizon=1, **kwargs
    ):
        if not self.multivariate:
            ts = [ts]
            cov = [cov]

        output = [[] for _ in range(len(ts))]
        for gix in range(len(ts)):
            dates = ts[gix].time_index

            # Convert Darts to pandas DataFrame for Chronos
            df_ts = ts[gix].to_dataframe()
            df_cov = cov[gix].to_dataframe()
            df = df_ts.merge(df_cov, left_index=True, right_index=True)

            for tix, t in enumerate(dates):  # forecast target date at horizon
                if t < start_date:
                    continue
                t_display = pd.Period(t, freq=self.df["Date"].iloc[0].freq)

                logging.info(f"Forecasting for date: {t_display}...")
                if tix < horizon:
                    logging.info("Not enough data to forecast, skipping...")
                    continue

                # Temporal masks
                input_mask = dates <= dates[tix - horizon]
                covariate_mask = dates <= t

                context_df = df[input_mask].reset_index()
                future_df = (
                    df[covariate_mask]
                    .reset_index()
                    .iloc[-horizon:]
                    .reset_index(drop=True)
                )
                future_df = future_df.drop(columns=[self.case_col])

                # Forecast
                pred_df = self.model.predict_df(
                    context_df,
                    future_df=future_df,
                    prediction_length=horizon,
                    quantile_levels=self.quantiles,
                    id_column=f"{self.geo_col}_codes",
                    timestamp_column="Date",
                    target="Log_Cases",
                )
                # [
                #   f'{self.geo_col}_codes',
                #   'Date',
                #   'target_name', (target column, e.g. Log_Cases)
                #   'predictions',  (median forecast value, same as '0.5' quantile)
                #   '0.01', ..., '0.99', (quantiles)
                # ]

                times = ts[gix].time_index[covariate_mask][-horizon:]
                forecasts = (
                    pred_df[map(str, self.quantiles)]
                    .to_numpy()
                    .reshape(horizon, 1, len(self.quantiles))
                )
                output[gix].append(TimeSeries.from_times_and_values(times, forecasts))

        # output = [time]series[horizon][1][samples]
        if self.multivariate:
            return output
        else:
            if len(output) > 1:
                raise ValueError("Multiple GIDs processed, but multivariate=False.")
            return output[0]  # limit to one GID for now (extend this later)


# -------- Chronos (DARTS implementation) --------
class ChronosQuantilesDarts(DartsBase):
    def __init__(self, *args, **kwargs):
        # Model parameters
        self.input_chunk_length = 48  # how many past steps the model can see
        self.output_chunk_length = 1  # how many future steps the model predicts at once
        # Initialize model
        super().__init__(*args, **kwargs)
        self.sampling_method = "samples"

    def build_model(self):
        return Chronos2Model(
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.output_chunk_length,
            likelihood=QuantileRegression(quantiles=self.quantiles),
        )

    def pre_fit(self, target_gids=None, **kwargs):
        logging.info(
            "Fitting Chronos-2 model on historical data "
            f"({self.train_start_date} to {self.train_end_date})..."
        )
        # Train on historical data only
        target_list, covar_list, _ = self.get_cases(
            future=False,
            target_gids=target_gids,
            start_date=self.train_start_date,
            end_date=self.train_end_date,
        )
        # Fit
        self.model.fit(
            series=target_list,
            past_covariates=covar_list,
            verbose=True,
        )

    def historical_forecasts(
        self, ts, cov, start_date=None, retrain=True, horizon=1, **kwargs
    ):
        logging.info(
            "Generating Chronos-2 historical forecasts "
            f"from {start_date} with retrain={retrain}..."
        )
        bt = self.model.historical_forecasts(
            series=ts,
            past_covariates=cov,
            forecast_horizon=horizon,
            start=start_date,
            stride=1,
            retrain=retrain,
            last_points_only=False,  # this changes the output format
            verbose=False,
            num_samples=self.num_samples,
        )
        return bt


# -------- pipeline helper --------
def chronos(
    df: DataFrame | pd.DataFrame,
    start_date: str | pd.Timestamp = pd.Timestamp.min,
    end_date: str | pd.Timestamp = pd.Timestamp.max,
    train_start_date: str | pd.Timestamp = pd.Timestamp.min,
    train_end_date: str | pd.Timestamp = pd.Timestamp.max,
    geo_col: str = "GID_2",
    geo_parent: Optional[str] = "GID_1",
    geo_parent_filter: Optional[List[str]] = None,
    horizons: List[int] = [1],
    case_col: str = "Log_Cases",
    covariate_cols: Optional[List[str]] = None,
    retrain: bool = True,  # Only use False for a quick test
    db_file: str | Path | None = None,
    multivariate: bool = False,
    train_col: Optional[str] = None,
) -> DataFrame | pd.DataFrame:
    """Chronos forecasting pipeline.

    Returns a Thucia DataFrame if db_file is specified, otherwise a pandas DataFrame.
    """
    logging.info("Starting Chronos forecasting pipeline...")

    # Instantiate model
    if train_col is None and Chronos2Pipeline is not None:
        # Using native Chronos implementation (fast, but only supports whole-dataset fit)
        model = ChronosQuantilesNative(
            df=df,
            case_col=case_col,
            geo_col=geo_col,
            geo_parent=geo_parent,
            covariate_cols=covariate_cols,
            horizons=horizons,
            db_file=db_file,
            train_start_date=train_start_date,
            train_end_date=train_end_date,
            multivariate=False,
        )
    else:
        # Using Darts implementation of Chronos (can be surprisingly slow, due to
        # training, but supports per-region and whole-dataset training)
        model = ChronosQuantilesDarts(
            df=df,
            case_col=case_col,
            geo_col=geo_col,
            geo_parent=geo_parent,
            covariate_cols=covariate_cols,
            horizons=horizons,
            db_file=db_file,
            train_start_date=train_start_date,
            train_end_date=train_end_date,
            multivariate=multivariate,
            # DARTS quantiles must be a subset of Chronos native quantiles
            quantiles=CHRONOS_QUANTILES,
        )

    # Historical predictions
    tdf = model.historical_predictions(
        start_date=start_date,
        retrain=retrain,
        train_col=train_col,
    )
    logging.info("Completed Chronos forecasting pipeline.")

    return tdf
