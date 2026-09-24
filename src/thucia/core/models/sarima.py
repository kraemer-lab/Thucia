import logging
from pathlib import Path
from typing import List
from typing import Optional

import pandas as pd
from darts.models import ARIMA
from darts.models import AutoARIMA
from thucia.core.cases import period_freq_str
from thucia.core.fs import DataFrame
from thucia.core.models.utils import season_length_for_freq

from ._meta import ModelSpec
from .darts import DartsBase

SPEC = ModelSpec(
    name="sarima",
    family="statistical",
    supports=frozenset({"season_length"}),
    sampling="samples",
)


# -------- SARIMA --------
class SarimaQuantiles(DartsBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sampling_method = "samples"
        self.sarima_retrain = False
        self.method = "powell"

    def set_retrain(self, retrain: bool):
        self.sarima_retrain = retrain

    def set_season_length(self, season_length: int):
        self.season_length = season_length

    def set_method(self, method: str):
        self.method = method

    def build_model(self, *args, **kwargs):
        # No global model
        return None

    def remove_gid_covariate(self, cov):
        try:
            cov = cov.drop_columns(f"{self.geo_col}_codes")
        except Exception:
            pass

    def pre_fit(self, target_gids=None, **kwargs):
        # Only determine model order if not retraining
        if self.sarima_retrain:
            return
        # Determine model structure from historical data
        self.fixed_order = {}
        logging.info("SARIMA pre-fitting: determining model orders for each GID...")
        for ix, target_gid in enumerate(target_gids):
            tic = pd.Timestamp.now()
            target_list, covar_list, _ = self.get_cases(
                future=False,
                target_gids=[target_gid],
                start_date=self.train_start_date,
                end_date=self.train_end_date,
            )
            cov = self.remove_gid_covariate(covar_list[0])
            model = AutoARIMA(
                season_length=self.season_length,
                quantiles=self.quantiles,
            )
            model.fit(
                target_list[0],
                future_covariates=cov,
            )
            p, q, P, Q, s, d, D = model.model.model_["arma"]
            s_info = f"{s}"
            if s != self.season_length:
                s = self.season_length
                s_info = f"{s_info} -> {self.season_length} [adjusted]"
            self.fixed_order[target_gid] = {
                "p": p,
                "q": q,
                "P": P,
                "Q": Q,
                "s": s,
                "d": d,
                "D": D,
            }
            toc = pd.Timestamp.now()
            logging.info(
                f"Determined SARIMA order for gid {target_gid}: "
                f"(p,d,q)=({p},{d},{q}), (P,D,Q,s)=({P},{D},{Q},{s_info}) "
                f"in {toc - tic}"
            )

    def historical_forecasts(
        self, ts, cov, gid=None, start_date=None, horizon=1, retrain=True, **kwargs
    ):
        if self.sarima_retrain:
            model = AutoARIMA(
                season_length=self.season_length,
                quantiles=self.quantiles,
            )
        else:
            if not gid:
                raise ValueError("gid must be provided when not retraining SARIMA.")
            model = ARIMA(
                p=self.fixed_order[gid]["p"],
                d=self.fixed_order[gid]["d"],
                q=self.fixed_order[gid]["q"],
                seasonal_order=(
                    self.fixed_order[gid]["P"],
                    self.fixed_order[gid]["D"],
                    self.fixed_order[gid]["Q"],
                    self.season_length,
                ),
            )
        # Remove GID covariate since SARIMA is univariate
        cov = self.remove_gid_covariate(cov)
        # AutoARIMA supports likelihood parameters directly; the fixed-order
        # ARIMA path does not, so sample from it instead (converted to
        # quantiles by DartsBase).
        if self.sarima_retrain:
            predict_kwargs = dict(num_samples=1, predict_likelihood_parameters=True)
        else:
            predict_kwargs = dict(num_samples=self.num_samples)
        # Historical forecasts
        bt = model.historical_forecasts(
            series=ts,
            future_covariates=cov,
            forecast_horizon=horizon,
            start=start_date,
            stride=1,
            retrain=retrain,
            last_points_only=True,
            verbose=False,
            **predict_kwargs,
        )
        return bt


# -------- pipeline helper --------
def sarima(
    df: pd.DataFrame,
    start_date: str | pd.Timestamp | pd.Period = pd.Timestamp.min,
    end_date: str | pd.Timestamp | pd.Period = pd.Timestamp.max,
    geo_col: str = "GID_2",
    geo_parent: Optional[str] = "GID_1",
    geo_parent_filter: Optional[List[str]] = None,
    horizons: List[int] = [1],
    case_col: str = "Log_Cases",
    covariate_cols: Optional[List[str]] = None,
    retrain: bool = True,  # AutoARIMA at every step
    db_file: str | Path | None = None,
    train_col: Optional[str] = None,
    num_samples: int | None = None,
    multivariate: bool = False,
    season_length: Optional[int] = None,  # None -> auto-detect from freq
    *args,
    **kwargs,
) -> DataFrame | pd.DataFrame:
    """SARIMA forecasting pipeline.

    Returns a Thucia DataFrame if db_file is specified, otherwise a pandas DataFrame.
    """
    logging.info("Starting SARIMA forecasting pipeline...")

    if args:
        logging.warning(f"Positional arguments {args} are ignored in sarima().")
    if kwargs:
        logging.warning(f"Keyword arguments {kwargs} are ignored in sarima().")

    if multivariate:
        logging.warning("SARIMA does not support multivariate forecasting.")

    if season_length is None:
        if isinstance(df["Date"].dtype, pd.PeriodDtype):
            season_length = season_length_for_freq(period_freq_str(df["Date"].dtype))
        else:
            season_length = 12

    # Instantiate model
    model = SarimaQuantiles(
        df=df,
        case_col=case_col,
        geo_col=geo_col,
        geo_parent=geo_parent,
        covariate_cols=covariate_cols,
        horizons=horizons,
        num_samples=num_samples,
        db_file=db_file,
        multivariate=False,
    )
    model.set_season_length(season_length=season_length)
    model.set_retrain(retrain)

    # Historical predictions
    tdf = model.historical_predictions(
        start_date=start_date,
        train_col=train_col,
    )
    logging.info("Completed SARIMA forecasting pipeline.")

    return tdf
