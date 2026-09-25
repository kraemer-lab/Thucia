import io
import logging
import os
import zipfile
from concurrent.futures import as_completed
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import requests
from thucia.core.cache import Cache
from thucia.core.cases import align_date_types
from thucia.core.fs import cache_folder
from thucia.core.geo.plugin_base import source_registry
from thucia.core.geo.plugin_base import SourceBase
from thucia.core.geo.stats import raster_stats_gid2
from tqdm import tqdm


cpu_count = os.cpu_count() or 1  # can return zero in some environments
max_workers = max(cpu_count - 1, 1)


@source_registry.register()
class EDO(SourceBase):
    ref = "edo"
    granularity = "M"
    name = "European Drought Observatory (EDO)"

    cache_file = Path(cache_folder) / "climate" / "edo_stats.sqlite"
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
        "mean": "REAL",
        "Date": "TEXT",
    }
    cache_keys = ["GID_2", "Date"]

    def __init__(self):
        self.cache = Cache(
            "sqlite",
            cache_file=self.cache_file,
            columntypes=self.cache_columns,
            keys=self.cache_keys,
            tablename="stats",
        )

    def get_filename(self, year: int, month: int) -> str:
        # Check for file in cache, download if not, and return as a DataFrame

        # Notes on file name conventions:
        #  spc06 - SPI6
        #  m - monthly
        #  gdo - Global Drought Observatory
        #  m_300 - spatial resolution, 300 arcsecs (approx 10km)
        #  z - Tile zone of the global raster (01=western[americas], 02=central, etc.)

        dirstem = Path(cache_folder) / "climate"
        filestem = "spc06_m_gdo_{year}{month:02}01_m_300_z01.tif"

        tif_file = Path(dirstem) / filestem.format(year=year, month=month)

        if not tif_file.exists():
            # Download file and place in the cache
            url_template = (
                "https://drought.emergency.copernicus.eu/data/"
                "Drought_Observatories_datasets/"
                "GDO_CHIRPS_Standardized_Precipitation_Index_SPI6/"
                "ver3-0-0/"
                "spc06_m_gdo_{year}0101_{year}{end_month}01_m.zip"
            )

            # Find the latest available month for the given year
            for end_month in range(12, month - 1, -1):
                url = url_template.format(year=year, end_month=end_month)

                # Mark zip file with todays date to prevent re-download
                zip_file = dirstem / url.split("/")[-1].replace(
                    ".zip",
                    f"_{pd.Timestamp.today().strftime('%Y%m%d')}.zip",
                )

                if zip_file.exists():
                    # We have todays zip file but not the tif, data is not available
                    raise FileNotFoundError(
                        f"Todays zip file {zip_file} exists but tif file {tif_file} "
                        "is missing."
                    )

                head_response = requests.head(url)
                if head_response.status_code == 200:
                    break
                if end_month == 1:
                    raise FileNotFoundError(f"No available data for {year}-{month:02}")
                logging.info(
                    f"EDO data for {year} up to month {end_month} not found. "
                    "Trying earlier month..."
                )

            logging.info(f"Downloading EDO data from {url}...")
            for attempt in range(5):
                try:
                    response = requests.get(url)
                except requests.RequestException as e:
                    logging.warning(f"Attempt {attempt + 1} failed: {e}")
                    continue

            if response.status_code != 200:
                raise FileNotFoundError(f"Failed to download {url}")

            # Save zip file to disk (prevent re-download)
            try:
                with open(zip_file, "wb") as f:
                    f.write(response.content)
                with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                    z.extractall(dirstem)
            except Exception:
                # Clean up partial files on failure
                if zip_file.exists():
                    zip_file.unlink()
            if not tif_file.exists():
                raise FileNotFoundError(
                    f"Raster file {tif_file} not found after extraction."
                )

        return tif_file

    def _get_cache_records(self, dates, geo_codes, geo_col: str):
        if isinstance(dates.dtype, pd.PeriodDtype):
            dates = dates.dt.to_timestamp(how="end")
        df = self.cache.get_records(
            {
                "Date": pd.to_datetime(dates).dt.strftime("%Y-%m-%d").tolist(),
                "GID_2": geo_codes,
            }
        )
        if geo_col != "GID_2":
            df = df.rename(columns={"GID_2": geo_col})
        return df

    def _add_cache_records(self, records, geo_col: str) -> None:
        records["Date"] = records["Date"].dt.strftime("%Y-%m-%d")  # to string
        if geo_col != "GID_2":
            records = records.rename(columns={geo_col: "GID_2"})
        # Non-GADM polygon maps don't carry the GADM attribute columns; the cache
        # schema keeps them, so pad absent columns with NULL before inserting.
        for col in self.cache_columns:
            if col not in records.columns:
                records[col] = None
        self.cache.add_records(records)

    def _process_date(self, args):
        """This runs inside worker processes."""
        date, geo_codes, tif_file, geo_col, iso3, polygons = args
        try:
            # Calculate zonal statistics for the regions
            stat = raster_stats_gid2(
                tif_file,
                geo_codes,
                geo_col=geo_col,
                iso3=iso3,
                polygons=polygons,
            )
            stat = stat[stat["mean"].notna()]
            stat["Date"] = date
            return stat
        except Exception as e:
            logging.error(f"Failed on date {date}: {e}")
            return None

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
        logging.info("Merging EDO data with case data...")

        # Get unique geo-code and Date combinations
        unique_gid2_dates = df[[geo_col, "Date"]].drop_duplicates()

        # Add cache hits and retun cache misses for processing
        stats = []
        if use_cache:
            logging.info(f"Checking cache for {len(unique_gid2_dates)} records.")
            stats = self._get_cache_records(
                dates=unique_gid2_dates["Date"],
                geo_codes=unique_gid2_dates[geo_col],
                geo_col=geo_col,
            )
            stats["Date"] = pd.to_datetime(stats["Date"])
            logging.info(f"Found {len(stats)} records in cache for EDO data.")
            if isinstance(unique_gid2_dates["Date"].dtype, pd.PeriodDtype):
                unique_gid2_dates = unique_gid2_dates.assign(
                    Date=unique_gid2_dates["Date"]
                    .dt.to_timestamp(how="end")
                    .dt.normalize()
                )
            unique_gid2_dates = unique_gid2_dates[
                ~unique_gid2_dates.set_index([geo_col, "Date"]).index.isin(
                    stats.set_index([geo_col, "Date"]).index
                )
            ]
            logging.info(f"Remaining records to process: {len(unique_gid2_dates)}.")
            stats = [stats]

        # Download datasets and prepare jobs
        jobs = []
        for date in unique_gid2_dates["Date"].unique():
            date_df = unique_gid2_dates[unique_gid2_dates["Date"] == date]
            geo_codes = date_df[geo_col].tolist()
            logging.info(
                f"Submitting {len(geo_codes)} {geo_col} regions for date {date}."
            )

            # Read the corresponding raster file for the date
            try:
                tif_file = self.get_filename(date.year, date.month)
            except FileNotFoundError as e:
                logging.warning(f"Raster file for {date} not found: {e}")
                continue

            jobs.append((date, geo_codes, tif_file, geo_col, iso3, polygons))

        # Run jobs in parallel
        results = []
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            future = {pool.submit(self._process_date, job): job for job in jobs}
            for fut in tqdm(as_completed(future), total=len(future)):
                result = fut.result()
                if result is not None:
                    results.append(result)

        # Post-process results (and add records to cache)
        for stat in results:
            self._add_cache_records(stat, geo_col)
            stat["Date"] = pd.to_datetime(stat["Date"])
            stats.append(stat)

        # Merge into dataframe
        frames = [s for s in stats if len(s)]
        stats = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        col_map = {"mean": "SPI6"}
        stats.rename(columns=col_map, inplace=True)

        # Align date formats for merging
        stats["Date"] = align_date_types(stats["Date"], df["Date"])

        # Merge with the original DataFrame
        df = df.merge(
            stats[[geo_col, "Date", *col_map.values()]],
            on=[geo_col, "Date"],
            how="left",
        )

        logging.info("EDO data merged.")
        return df
