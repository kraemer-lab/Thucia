import io
import logging
import zipfile
from pathlib import Path

import pandas as pd
import rasterio
import requests
from thucia.core.cache import Cache
from thucia.core.cases import align_date_types
from thucia.core.fs import cache_folder
from thucia.core.geo.plugin_base import source_registry
from thucia.core.geo.plugin_base import SourceBase
from thucia.core.geo.stats import raster_stats_gid2


@source_registry.register()
class WorldClim(SourceBase):
    ref = "worldclim"
    granularity = "M"
    name = "WorldClim"

    cache_file = Path(cache_folder) / "climate" / "worldclim_stats.sqlite"
    cache_columns = {
        "GID_2": "TEXT",
        "GID_0": "TEXT",
        "COUNTRY": "TEXT",
        "GID_1": "TEXT",
        "NAME_1": "TEXT",
        "NL_NAME_1": "TEXT",
        "NAME_2": "TEXT",
        "NL_NAME_2": "TEXT",
        "TYPE_2": "TEXT",
        "ENGTYPE_2": "TEXT",
        "CC_2": "TEXT",
        "HASC_2": "TEXT",
        "metric": "TEXT",
        "mean": "REAL",
        "Date": "TEXT",
        "source": "TEXT",
    }
    cache_keys = ["GID_2", "Date", "metric"]

    def __init__(self):
        self.cache = Cache(
            "sqlite",
            cache_file=self.cache_file,
            columntypes=self.cache_columns,
            keys=self.cache_keys,
            tablename="stats",
        )

    def _get_filename_cru(self, metric, year, month):
        # Check for file in cache, download if not, and return as a DataFrame

        dirstem = Path(cache_folder) / "climate"
        filestem = "wc2.1_cruts4.09_2.5m_{metric}_{year}-{month:02d}.tif"

        tif_file = Path(dirstem) / filestem.format(
            metric=metric, year=year, month=month
        )

        if not tif_file.exists():
            # CRU-TS version
            cru_ts_version = "4.09"
            max_year = 2024  # CRU-TS 4.09 supports records up to 2024

            if year > max_year:
                raise ValueError(
                    f"Year {year} is beyond the maximum supported year {max_year}."
                )

            # Download file and place in the cache
            url_template = (
                "https://geodata.ucdavis.edu/climate/worldclim/2_1/hist/"
                "cts{cts_version}/"
                "wc2.1_cruts4.09_2.5m_{metric}_{year_start}-{year_end}.zip"
            )
            # Year range is by decade
            year_start = year - (year % 10)
            year_end = min(year_start + 9, max_year)

            url = url_template.format(
                cts_version=cru_ts_version,
                metric=metric,
                year_start=year_start,
                year_end=year_end,
            )
            logging.info(f"Downloading WorldClim data from {url}...")
            response = requests.get(url)
            if response.status_code != 200:
                raise FileNotFoundError(f"Failed to download {url}")
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                z.extractall(dirstem)
            if not tif_file.exists():
                raise FileNotFoundError(
                    f"Raster file {tif_file} not found after extraction."
                )

        return tif_file

    def _get_filename_forecast(self, metric, year, month, source="ACCESS-CM2"):
        # Check for file in cache, download if not, and return as a DataFrame

        res = "2.5m"
        ssp = "ssp245"

        dirstem = Path(cache_folder) / "climate"
        filestem_month = "wc2.1_{res}_{metric}_{source}_{ssp}_{year}-{month:02d}.tif"
        filestem_year = "wc2.1_{res}_{metric}_{source}_{ssp}_{year}.tif"

        tif_file_month = Path(dirstem) / filestem_month.format(
            res=res, metric=metric, source=source, ssp=ssp, year=year, month=month
        )
        tif_file_year = Path(dirstem) / filestem_year.format(
            res=res, metric=metric, source=source, ssp=ssp, year=year
        )

        if not tif_file_year.exists():
            # Download file and place in the cache
            url_template = (
                "https://geodata.ucdavis.edu/cmip6/{res}/{source}/{ssp}/"
                "wc2.1_{res}_{metric}_{source}_{ssp}_{year_start}-{year_end}.tif"
            )
            # Year range is by 20 years (e.g. 2021-2040)
            year_start = year - (year % 20) + 1
            year_end = year_start + 19

            url = url_template.format(
                res=res,
                source=source,
                ssp=ssp,
                metric=metric,
                year_start=year_start,
                year_end=year_end,
            )
            logging.info(f"Downloading WorldClim data from {url}...")
            response = requests.get(url)
            if response.status_code != 200:
                raise FileNotFoundError(f"Failed to download {url}")
            with open(tif_file_year, "wb") as f:
                f.write(response.content)

        if not tif_file_month.exists():
            # Extract the month band from the year file
            with rasterio.open(tif_file_year) as src:
                band = src.read(int(month))
                profile = src.profile.copy()
                profile.update(count=1)
                with rasterio.open(tif_file_month, "w", **profile) as dst:
                    dst.write(band, 1)

        if not tif_file_month.exists():
            raise FileNotFoundError(
                f"Raster file {tif_file_month} not found after extraction."
            )

        return tif_file_month

    def get_filename(self, metric, year, month):
        try:
            return self._get_filename_cru(metric, year, month), "CRU-TS"
        except (FileNotFoundError, ValueError):
            pass
        logging.warning(
            f"CRU-TS data not available for {year}-{month:02d} ---"
            " using WorldClim forecast data."
        )
        try:
            return self._get_filename_forecast(metric, year, month), "forecast"
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Raster file for {metric} in {year}-{month:02d} not found."
            )

    def _get_cache_records(self, metric, dates, geo_codes, geo_col: str):
        df = self.cache.get_records(
            {
                "metric": [metric] * len(dates),
                "Date": pd.to_datetime(dates).dt.strftime("%Y-%m-%d").tolist(),
                "GID_2": geo_codes,
            }
        )
        if geo_col != "GID_2":
            df = df.rename(columns={"GID_2": geo_col})
        return df

    def _add_cache_records(self, metric, records, geo_col: str) -> None:
        records["metric"] = metric
        records["Date"] = records["Date"].dt.strftime("%Y-%m-%d")  # to string
        if geo_col != "GID_2":
            records = records.rename(columns={geo_col: "GID_2"})
        # Non-GADM polygon maps don't carry the GADM attribute columns; the cache
        # schema keeps them, so pad absent columns with NULL before inserting.
        for col in self.cache_columns:
            if col not in records.columns:
                records[col] = None
        self.cache.add_records(records)
        records.drop(columns=["metric"], inplace=True)

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
        logging.info("Merging climate data with case data...")

        if not metrics or metrics == ["*"]:
            metrics = ["tmin", "tmax", "prec"]
        if not measures:
            measures = ["mean"]

        for metric in metrics:
            logging.info(f"Merging climate data for metric: {metric}")

            # Get unique geo-code and Date combinations
            unique_gid2_dates = df[[geo_col, "Date"]].drop_duplicates()

            # Add cache hits and return cache misses for processing
            stats = []
            if use_cache:
                logging.info(
                    f"Checking cache for metric '{metric}' ("
                    f"{len(unique_gid2_dates)} records)."
                )
                stats = self._get_cache_records(
                    metric,
                    unique_gid2_dates["Date"],
                    unique_gid2_dates[geo_col],
                    geo_col,
                )
                stats["Date"] = pd.to_datetime(stats["Date"])
                logging.info(
                    f"Found {len(stats)} records in cache for metric '{metric}'."
                )
                unique_gid2_dates = unique_gid2_dates[
                    ~unique_gid2_dates.set_index([geo_col, "Date"]).index.isin(
                        stats.set_index([geo_col, "Date"]).index
                    )
                ]
                logging.info(f"Remaining records to process: {len(unique_gid2_dates)}.")
                stats = [stats]

            # Read and merge mean climate data per region for each Date
            n_unique_gid2_dates = unique_gid2_dates["Date"].nunique()
            for ix, date in enumerate(unique_gid2_dates["Date"].unique()):
                tic = pd.Timestamp.now()
                date_df = unique_gid2_dates[unique_gid2_dates["Date"] == date]
                geo_codes = date_df[geo_col].tolist()
                logging.info(
                    f"Processing metric '{metric}' for {date.strftime('%Y-%m-%d')}"
                    f" with {len(geo_codes)} {geo_col} regions."
                )

                # Read the corresponding raster file for the date
                try:
                    tif_file, source = self.get_filename(metric, date.year, date.month)
                except FileNotFoundError as e:
                    logging.warning(f"Raster file for {date} not found: {e}")
                    continue

                # Calculate zonal statistics for the regions
                stat = raster_stats_gid2(
                    tif_file,
                    geo_codes,
                    geo_col=geo_col,
                    polygons=polygons,
                    iso3=iso3,
                )
                stat = stat[stat["mean"].notna()]
                stat["Date"] = date
                stat["source"] = source

                self._add_cache_records(metric, stat, geo_col)

                stat["Date"] = pd.to_datetime(stat["Date"])  # ensure datetime
                stats.append(stat)

                # Estimate time remaining
                toc = pd.Timestamp.now()
                estimated_time_remaining = (toc - tic) * (n_unique_gid2_dates - ix - 1)
                logging.info(
                    f"Took {toc - tic}, "
                    f"estimated time remaining: {estimated_time_remaining}."
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

            # Simpify 'mean' column names
            if "mean" in measures:
                df.rename(columns={f"{metric}_mean": f"{metric}"}, inplace=True)
            logging.info(f"Merged {metric} data with {len(stats)} records.")

        logging.info("Climate data merged.")
        return df
