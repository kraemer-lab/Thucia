import logging
from pathlib import Path
from typing import List
from typing import Optional

import pandas as pd
from darts import TimeSeries
from thucia.core.fs import DataFrame
from timesfm import TimesFm
from timesfm import TimesFmCheckpoint
from timesfm import TimesFmHparams

from ._meta import ModelSpec
from .darts import DartsBase

SPEC = ModelSpec(
    name="timesfm",
    family="darts",
    sampling="samples",
    extras=("timesfm[torch]",),
)


# -------- TimesFM --------
class TimesFMSamples(DartsBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sampling_method = "samples"

    def build_model(self, horizon):
        return TimesFm(
            hparams=TimesFmHparams(
                backend="cpu",
                context_len=32,
                horizon_len=horizon,
                num_layers=50,
                per_core_batch_size=32,
                use_positional_embedding=True,
            ),
            checkpoint=TimesFmCheckpoint(
                huggingface_repo_id="google/timesfm-2.0-500m-pytorch"
            ),
        )

    def pre_fit(self, target_gids=None, **kwargs):
        pass

    def historical_forecasts(
        self,
        ts,
        cov,
        horizon,
        start_date=None,
        retrain=True,
        **kwargs,
    ):
        dates = ts.time_index
        output = []
        for tix, t in enumerate(dates):  # forecast target date at horizon
            if t < start_date:
                continue
            t_display = pd.Period(t, freq=self.df["Date"].iloc[0].freq)

            logging.info(f"Forecasting for date: {t_display}...")
            if tix < horizon:
                logging.info("Not enough data to forecast, skipping...")
                continue

            # Temporal masks
            input_mask = ts.time_index <= dates[tix - horizon]
            covariate_mask = ts.time_index <= t

            # Inputs
            tsv = ts.values().squeeze()
            inputs = [tsv[input_mask].tolist()]

            # Covariates
            dynamic_numerical_covariates = {}
            dcov = cov.values().squeeze()
            for i in range(dcov.shape[1]):
                dynamic_numerical_covariates[f"covariate_{i}"] = [
                    dcov[covariate_mask, i].tolist()
                ]

            # Forecast
            forecasts, _ols = self.model.forecast_with_covariates(
                inputs=inputs,
                dynamic_numerical_covariates=dynamic_numerical_covariates,
                dynamic_categorical_covariates=None,
                static_numerical_covariates=None,  # No grouping
                static_categorical_covariates=None,
                freq=[1],  # 1 = monthly
                xreg_mode="xreg + timesfm",
            )

            times = ts.time_index[covariate_mask][-horizon:]
            output.append(
                pd.DataFrame({"Log_Cases_q0.500": forecasts[0][-1]}, index=[times[-1]])
            )

        # output = series[time][quantiles=1][1]
        output = pd.concat(output, axis=0)
        output = TimeSeries.from_dataframe(output)
        return output


# -------- pipeline helper --------
def timesfm(
    df: DataFrame | pd.DataFrame,
    start_date: str | pd.Timestamp = pd.Timestamp.min,
    end_date: str | pd.Timestamp = pd.Timestamp.max,
    geo_col: str = "GID_2",
    geo_parent: Optional[str] = "GID_1",
    geo_parent_filter: Optional[List[str]] = None,
    horizons: List[int] = [1],
    case_col: str = "Log_Cases",
    covariate_cols: Optional[List[str]] = None,
    retrain: bool = True,  # Only use False for a quick test
    db_file: str | Path | None = None,
    train_col: Optional[str] = None,
    *args,
    **kwargs,
) -> DataFrame | pd.DataFrame:
    """TimesFM forecasting pipeline.

    Returns a Thucia DataFrame if db_file is specified, otherwise a pandas DataFrame.
    """
    logging.info("Starting TimesFM forecasting pipeline...")

    if args:
        logging.warning(f"Unused positional arguments: {args}")
    if kwargs:
        logging.warning(f"Unused keyword arguments: {kwargs}")

    # Instantiate model
    model = TimesFMSamples(
        df=df,
        case_col=case_col,
        geo_col=geo_col,
        geo_parent=geo_parent,
        covariate_cols=covariate_cols,
        horizons=horizons,
        db_file=db_file,
    )

    # Historical predictions
    tdf = model.historical_predictions(
        start_date=start_date,
        retrain=retrain,
        train_col=train_col,
    )
    logging.info("Completed TimesFM forecasting pipeline.")

    return tdf
