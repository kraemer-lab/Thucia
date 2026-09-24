import logging
from pathlib import Path
from typing import List
from typing import Optional

import pandas as pd
from darts.models import TCNModel
from darts.utils.likelihood_models import QuantileRegression
from thucia.core.fs import DataFrame

from ._meta import ModelSpec
from .darts import DartsBase

SPEC = ModelSpec(
    name="tcn",
    family="darts",
    supports=frozenset({"train_end_date", "retrain", "multivariate", "samples"}),
    sampling="quantiles",
    extras=("torch",),
)


# -------- TCN --------
class TcnSamples(DartsBase):
    def __init__(self, *args, **kwargs):
        # Model parameters
        self.input_chunk_length = 48  # how many past steps the model can see
        self.output_chunk_length = 1  # how many future steps the model predicts at once
        self.kernel_size = 3
        self.num_filters = 4  # (increase to 16 or 32 for country-level data)
        self.dropout = 0.2  # enables MC dropout
        self.random_state = 42  # for reproducibility
        self.n_epochs = 150  # default 100
        self.batch_size = 64
        self.optimizer_kwargs = {
            "lr": 1e-4,
            "weight_decay": 1e-5,
        }

        # Initialize model
        super().__init__(*args, **kwargs)
        self.sampling_method = "quantiles"

    def build_model(self, horizon):
        return TCNModel(
            input_chunk_length=48,
            output_chunk_length=horizon,
            # kernel_size=self.kernel_size,
            # num_filters=self.num_filters,
            dropout=self.dropout,
            random_state=self.random_state,
            likelihood=QuantileRegression(self.quantiles),
            save_checkpoints=False,
            # force_reset=True,
            n_epochs=self.n_epochs,
            # batch_size=self.batch_size,
            # optimizer_kwargs=self.optimizer_kwargs,
        )

    def pre_fit(self, target_gids=None, **kwargs):
        logging.info(
            "Fitting TCN model on historical data "
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
        self,
        ts,
        cov,
        horizon,
        start_date=None,
        retrain=True,
        **kwargs,
    ):
        logging.info(
            "Generating TCN historical forecasts "
            f"from {start_date} with retrain={retrain}..."
        )
        bt = self.model.historical_forecasts(
            series=ts,
            past_covariates=cov,
            forecast_horizon=horizon,
            start=start_date,
            stride=1,
            retrain=retrain,
            last_points_only=True,
            verbose=False,
            num_samples=1,
            predict_likelihood_parameters=True,
        )
        return bt


# -------- pipeline helper --------
def tcn(
    df: pd.DataFrame,
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
    retrain: bool = True,  # Retrain after every step (accurate but slow)
    db_file: str | Path | None = None,
    train_col: Optional[str] = None,
    num_samples: int | None = None,
    multivariate: bool = True,
) -> DataFrame | pd.DataFrame:
    """Temporal Convolutional Network (TCN) forecasting pipeline.

    Returns a Thucia DataFrame if db_file is specified, otherwise a pandas DataFrame.
    """
    logging.info("Starting TCN forecasting pipeline...")

    # Instantiate model
    model = TcnSamples(
        df=df,
        case_col=case_col,
        geo_col=geo_col,
        geo_parent=geo_parent,
        covariate_cols=covariate_cols,
        horizons=horizons,
        num_samples=num_samples,
        db_file=db_file,
        train_start_date=train_start_date,
        train_end_date=train_end_date,
        multivariate=multivariate,
    )

    # Historical predictions
    tdf = model.historical_predictions(
        start_date=start_date,
        retrain=retrain,
        train_col=train_col,
    )
    logging.info("Completed TCN forecasting pipeline.")

    return tdf
