# Pipeline configuration.
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any
from typing import Optional

import pandas as pd


@dataclass
class PipelineConfig:
    """Parameters for the forecasting pipeline stages.

    The library stays generic; use-case-specific values (e.g. a country's
    covariate lag recipe or model list) are supplied via this config.
    """

    path: str | Path = "."
    iso3: Optional[str] = None
    adm1: Optional[list[str]] = None

    # Case aggregation
    cutoff_date: Optional[str | pd.Timestamp | pd.Period] = None
    future_periods: int = 12  # number of future rows per region (of the data's freq)

    # Covariate merging
    source_specs: list[str] = field(
        default_factory=lambda: [
            "worldclim.*",
            "edo.spi6",
            "noaa.oni",
            "worldpop.pop_count",
        ]
    )
    # Interpolation method for coarser-granularity sources onto finer case grids
    # ("linear" default; also "ffill"/"bfill").
    covariate_interpolation: str = "linear"

    # Model fitting
    start_date: Optional[str | pd.Period] = None
    train_start_date: Optional[str | pd.Period] = None
    train_end_date: Optional[str | pd.Period] = None
    horizons: list[int] = field(default_factory=lambda: [1, 3, 6, 12])
    # The column holding the finest geographic unit being modelled (defaults to
    # GADM admin-2 codes) and, when set, its coarser parent column (GADM
    # admin-1). Set `geo_parent=None` when the data has no parent column.
    geo_col: str = "GID_2"
    geo_parent: Optional[str] = "GID_1"
    # Optional column name to fit per-region at a *different* resolution than
    # `geo_col` (e.g. train per GID_1 but forecast per GID_2). None keeps the
    # default per-`geo_col` fitting.
    train_col: Optional[str] = None
    case_col: str = "Log_Cases"
    num_samples: int = 200
    retrain: bool = False
    multivariate: bool = False
    # SARIMA seasonal period; None auto-detects from the data frequency
    # (monthly -> 12, weekly -> 52, daily -> 365).
    season_length: Optional[int] = None

    # Model-input preparation: optional lag/roll feature recipe for
    # `build_features` (see thucia.core.models.utils.covariates). When None,
    # all non-base columns of the input are treated as covariates.
    lag_spec: Optional[list[dict[str, Any]]] = None

    def __post_init__(self) -> None:
        self.path = Path(self.path)
