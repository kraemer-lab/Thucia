import logging
from functools import lru_cache
from pathlib import Path

import pandas as pd
import requests
from thucia.core.cases import align_date_types
from thucia.core.fs import cache_folder
from thucia.core.geo.plugin_base import source_registry
from thucia.core.geo.plugin_base import SourceBase
from thucia.core.geo.stats import raster_stats_gid2


def resolve_country_key(iso3, polygons, geo_codes, country_col: str = "COUNTRY"):
    """Resolve the single-country key WorldPop needs to pick its raster.

    Precedence: an explicit ``iso3``, else a single unique value in the regions
    map's ``country_col``, else the GADM code prefix (reproduces the historic
    ``GID_1``-derived country), else a ``ValueError``.
    """
    if iso3:
        return iso3
    if polygons is not None and country_col in polygons.columns:
        values = [str(v) for v in polygons[country_col].dropna().unique().tolist()]
        if len(values) != 1:
            raise ValueError(
                f"WorldPop needs a single country: the regions map's "
                f"'{country_col}' column has {len(values)} unique values. "
                "Pass iso3= to disambiguate."
            )
        return values[0]
    prefixes = {str(c).split(".")[0] for c in geo_codes}
    if len(prefixes) == 1:
        return prefixes.pop()
    raise ValueError(
        "WorldPop needs a single-country key: pass iso3=, add a "
        f"'{country_col}' column to the regions map, or use GADM-shaped codes."
    )


@source_registry.register()
class WorldPop(SourceBase):
    ref = "worldpop"
    name = "WorldPop"

    def get_filename(self, metric, gid_1, year):
        # Check for file in cache, download if not, and return as a DataFrame

        dirstem = Path(cache_folder) / "geo"
        dirstem.mkdir(parents=True, exist_ok=True)

        if metric == "pop_count":
            max_year = 2020
            if year > max_year:
                logging.warning(
                    f"Year {year} exceeds maximum available year {max_year} for "
                    "population count data. Switching to unconstrained estimates."
                )
                metric = "pop_estimate_unconstrained"

        match metric:
            case "pop_count":
                filestem = "{gid1}_ppp_{year}_1km_Aggregated.tif"
                url_template = (
                    "https://data.worldpop.org/GIS/Population/"
                    "Global_2000_2020_1km/{year}/{GID1}/"
                    "{gid1}_ppp_{year}_1km_Aggregated.tif"
                )
            case "pop_estimate_constrained":
                filestem = "{gid1}_pop_{year}_CN_1km_R2024B_UA_v1.tif"
                url_template = (
                    "https://data.worldpop.org/GIS/Population/Global_2015_2030/R2024B/"
                    "{year}/{GID1}/v1/1km_ua/constrained/"
                    "{gid1}_pop_{year}_CN_1km_R2024B_UA_v1.tif"
                )
            case "pop_estimate_unconstrained":
                filestem = "{gid1}_pop_{year}_UC_1km_R2024B_UA_v1.tif"
                url_template = (
                    "https://data.worldpop.org/GIS/Population/Global_2015_2030/R2024B/"
                    "{year}/{GID1}/v1/1km_ua/unconstrained/"
                    "{gid1}_pop_{year}_UC_1km_R2024B_UA_v1.tif"
                )
            case _:
                raise ValueError(f"Unsupported metric: {metric}")

        # Local file name
        tif_file = Path(dirstem) / filestem.format(gid1=gid_1.lower(), year=year)

        if not tif_file.exists():
            # Download file and place in the cache
            url = url_template.format(
                gid1=gid_1.lower(),
                GID1=gid_1.upper(),
                year=year,
            )
            logging.info(f"Downloading WorldPop data from {url}...")
            response = requests.get(url)
            if response.status_code != 200:
                raise FileNotFoundError(f"Failed to download {url}")
            with open(tif_file, "wb") as f:
                f.write(response.content)
            if not tif_file.exists():
                raise FileNotFoundError(
                    f"Raster file {tif_file} not found after extraction."
                )

        return tif_file

    @lru_cache(maxsize=500)
    def _get_cached_stats_gadm(self, tif_file, gid_2s, stats):
        return raster_stats_gid2(tif_file, list(gid_2s), stats=list(stats))

    def get_cached_stats(self, tif_file, geo_codes, stats, geo_col, iso3, polygons):
        if polygons is not None or geo_col != "GID_2":
            return raster_stats_gid2(
                tif_file,
                list(geo_codes),
                stats=list(stats),
                geo_col=geo_col,
                iso3=iso3,
                polygons=polygons,
            )
        return self._get_cached_stats_gadm(
            tif_file, tuple(geo_codes), stats=tuple(stats)
        )

    def merge(
        self,
        df: pd.DataFrame,
        metrics: list[str] | None = None,
        measures: list[str] | None = None,
        use_cache: bool = False,
        *,
        geo_col: str = "GID_2",
        iso3: str | None = None,
        polygons=None,
    ) -> pd.DataFrame:
        logging.info("Merging population data with case data...")

        if not metrics:
            metrics = ["pop_count"]
        if metrics == ["*"]:
            metrics = [
                "pop_count",
                "pop_estimate_constrained",
                "pop_estimate_unconstrained",
            ]
        if not measures:
            measures = ["sum"]

        # Get unique geo-code and Date combinations
        unique_gid2_dates = df[[geo_col, "Date"]].drop_duplicates()
        gid_1 = resolve_country_key(
            iso3, polygons, unique_gid2_dates[geo_col].unique().tolist()
        )

        for metric in metrics:
            logging.info(f"Merging population data for metric: {metric}")

            # Read and merge mean climate data per region for each Date
            stats = []
            for date in unique_gid2_dates["Date"].unique():
                date_df = unique_gid2_dates[unique_gid2_dates["Date"] == date]
                geo_codes = date_df[geo_col].tolist()

                # Read the corresponding raster file for the date
                try:
                    tif_file = self.get_filename(metric, gid_1, date.year)
                except FileNotFoundError as e:
                    logging.warning(f"Raster file for {date} not found: {e}")
                    continue

                if tif_file is None:
                    continue

                # Calculate zonal statistics for the regions
                stat = self.get_cached_stats(  # cached as pop is per year
                    tif_file,
                    geo_codes,
                    stats=("sum",),
                    geo_col=geo_col,
                    iso3=iso3,
                    polygons=polygons,
                ).copy()
                if len(stat) != len(geo_codes):
                    print(
                        f"Warning: Expected {len(geo_codes)} stats for {date}, "
                        f"got {len(stat)}"
                    )
                stat["sum"] = stat["sum"].fillna(0)  # Ensure no NaN values
                stat["Date"] = date
                stats.append(stat)

            if len(stats) != len(unique_gid2_dates["Date"].unique()):
                logging.warning(
                    f"Some dates did not have data for {metric}. "
                    "Check if the raster files are available."
                )

            stats = pd.concat(stats, ignore_index=True)
            col_map = {f"{measure}": f"{metric}_{measure}" for measure in measures}
            stats.rename(columns=col_map, inplace=True)

            # Align date formats for merging
            stats["Date"] = align_date_types(stats["Date"], df["Date"])

            # Merge with the original DataFrame
            df = df.merge(
                stats[[geo_col, "Date", *col_map.values()]],
                on=[geo_col, "Date"],
                how="left",
            )

            # Simpify 'sum' column names
            if "sum" in measures:
                df.rename(columns={f"{metric}_sum": f"{metric}"}, inplace=True)
            logging.info(f"Merged {metric} data with {len(stats)} records.")

        logging.info("Climate data merged.")
        return df
