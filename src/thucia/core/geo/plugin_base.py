from __future__ import annotations

from typing import Optional

import pandas as pd
from thucia.core.registry import Registry

#: Registry of covariate-source plugins, keyed by their ``ref`` string.
#: Sources self-register via ``@source_registry.register()`` at import time.
source_registry: Registry = Registry("covariate source")


class SourceBase:
    """Base class for covariate sources (e.g. WorldClim, EDO, NOAA, WorldPop).

    Subclasses set ``ref`` (the registry key) and ``name``, and implement
    ``merge(df, metrics, measures, use_cache, *, geo_col, iso3, polygons)``.
    """

    ref: str | None = None
    name: str = "Base"
    #: Native period granularity of the source's data ("M", "W", "D", "Y").
    #: When merged onto a finer case grid the values are interpolated.
    granularity: str = "M"

    def merge(
        self,
        df: pd.DataFrame,
        metrics: Optional[list[str]] = None,
        measures: Optional[list[str]] = None,
        use_cache: bool = False,
        *,
        geo_col: str = "GID_2",
        iso3: Optional[str] = None,
        polygons=None,
    ) -> pd.DataFrame:
        raise NotImplementedError("Source plugins must implement merge()")
