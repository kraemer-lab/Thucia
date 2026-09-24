import logging
import unicodedata
import warnings
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from rapidfuzz import fuzz
from rapidfuzz import process
from thucia.core.cases import period_freq_str
from thucia.core.fs import cache_folder
from thucia.core.fs import DataFrame

from .plugin_base import source_registry
from .plugin_loader import load_plugins


def remove_accents(text):
    if not isinstance(text, str):
        return text
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )


def fuzzy_match_one(query, reference_list, threshold=90):
    if not isinstance(query, str):
        return ""
    match, score, _ = process.extractOne(
        query, reference_list, scorer=fuzz.token_sort_ratio
    )
    return match if score >= threshold else ""


def get_admin2_list(iso3: str) -> pd.DataFrame:
    """
    Get a list of administrative level 2 regions for the specified ISO3 country code.

    Parameters:
    iso3 (str): The ISO3 code of the country.

    Returns:
    GeoDataFrame: A GeoDataFrame containing the administrative regions.
    """

    file_path = Path(cache_folder) / "geo" / iso3 / f"gadm41_{iso3}.gpkg"
    if not file_path.exists():
        logging.info("GeoPackage file not found, downloading...")
        url = (
            f"https://geodata.ucdavis.edu/gadm/gadm4.1/gpkg/gadm41_{iso3.upper()}.gpkg"
        )
        response = requests.get(url)
        if response.status_code == 200:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "wb") as f:
                f.write(response.content)
            logging.info(f"Downloaded GeoPackage file to {file_path}")
        else:
            logging.error(f"Failed to download GeoPackage file from {url}")
    if not file_path.exists():
        raise FileNotFoundError(f"GeoPackage file for {iso3} not found at {file_path}")

    gdf = gpd.read_file(str(file_path), layer="ADM_ADM_2")
    gdf = gdf[["GID_0", "GID_1", "NAME_1", "GID_2", "NAME_2", "HASC_2"]]

    return gdf


def align_admin2_regions(
    df: pd.DataFrame,
    admin1_col: str | None = "ADM1",
    admin2_col: str | None = "ADM2",
    iso3: str | None = None,
) -> pd.DataFrame:
    """
    Aligns the administrative level 2 regions in the DataFrame with the standardized
    names.

    Parameters:
    df (pd.DataFrame): The DataFrame containing administrative regions.
    admin1_col (str): The column name for administrative level 1 regions.
                      Can be set to None if not available.
    admin2_col (str): The column name for administrative level 2 regions.
    iso3 (str): The ISO3 code of the country.

    Returns:
    GeoDataFrame: The DataFrame with aligned administrative regions.
    """

    use_admin1 = admin1_col is not None
    if admin2_col is None:
        raise ValueError("admin2_col must be specified.")
    if iso3 is None:
        raise ValueError("iso3 must be specified.")

    logging.info(f"Initial data contains {len(df)} records.")
    if use_admin1:
        logging.info(
            f"There are {df[admin1_col].nunique()} unique Admin-1 regions and "
            f"{df[admin2_col].nunique()} unique Admin-2 regions."
        )
    else:
        logging.info(f"There are {df[admin2_col].nunique()} unique Admin-2 regions.")
    logging.info(f"Aligning administrative regions for {iso3}...")

    df = df.copy()  # make local copy so that original is not modified

    def homogenise(x):
        return x.apply(remove_accents).str.replace(" ", "", regex=False).str.lower()

    logging.info("Loading administrative regions...")
    ref_names = get_admin2_list(iso3)
    logging.info("Homogenising administrative names...")
    adm1h_ref = homogenise(ref_names["NAME_1"]) if use_admin1 else None
    adm2h_ref = homogenise(ref_names["NAME_2"])
    adm1h_df = homogenise(df[admin1_col]) if use_admin1 else None
    adm2h_df = homogenise(df[admin2_col])

    # Build a reference mapping table
    ref_map = pd.DataFrame(
        {
            "adm1h": adm1h_ref,
            "adm2h": adm2h_ref,
            "NAME_1": ref_names["NAME_1"].values,
            "NAME_2": ref_names["NAME_2"].values,
        }
    )

    # Ensure main DataFrame has the matching homogenized keys
    if use_admin1:
        df["adm1h"] = adm1h_df
    df["adm2h"] = adm2h_df

    if use_admin1:
        exact = df.merge(ref_map, on=["adm1h", "adm2h"], how="left")
        unmatched = exact[exact["NAME_1"].isna() | exact["NAME_2"].isna()].copy()
    else:
        exact = df.merge(ref_map, on=["adm2h"], how="left")
        unmatched = exact[exact["NAME_2"].isna()].copy()

    if use_admin1:
        unique_unmatched_adm1 = unmatched[[admin1_col, "adm1h"]].drop_duplicates()
        unique_unmatched_adm1["fuzzy_match"] = unique_unmatched_adm1[admin1_col].apply(
            lambda x: fuzzy_match_one(x, ref_names["NAME_1"].tolist())
        )

    unique_unmatched_adm2 = unmatched[[admin2_col, "adm2h"]].drop_duplicates()
    unique_unmatched_adm2["fuzzy_match"] = unique_unmatched_adm2[admin2_col].apply(
        lambda x: fuzzy_match_one(x, ref_names["NAME_2"].tolist())
    )

    if use_admin1:
        adm1_fuzzy_map = dict(
            zip(unique_unmatched_adm1["adm1h"], unique_unmatched_adm1["fuzzy_match"])
        )
    adm2_fuzzy_map = dict(
        zip(unique_unmatched_adm2["adm2h"], unique_unmatched_adm2["fuzzy_match"])
    )

    def resolve_fuzzy_admin1(row):
        if pd.isna(row["NAME_1"]) and row["adm1h"] in adm1_fuzzy_map:
            return adm1_fuzzy_map[row["adm1h"]]
        return row["NAME_1"]

    def resolve_fuzzy_admin2(row):
        if pd.isna(row["NAME_2"]) and row["adm2h"] in adm2_fuzzy_map:
            return adm2_fuzzy_map[row["adm2h"]]
        return row["NAME_2"]

    if use_admin1:
        exact["NAME_1"] = exact.apply(resolve_fuzzy_admin1, axis=1)
    exact["NAME_2"] = exact.apply(resolve_fuzzy_admin2, axis=1)

    if use_admin1:
        df[admin1_col] = exact["NAME_1"]
    df[admin2_col] = exact["NAME_2"]

    if use_admin1:
        unmatched_regions = (
            df[df[admin1_col].isna()]
            .drop_duplicates(subset=["adm1h", "adm2h"])
            .sort_values(by=["adm1h", "adm2h"])
        )
        unmatched_regions = unmatched_regions.merge(
            ref_map[["adm1h", "adm2h", "NAME_1", "NAME_2"]],
            on=["adm1h", "adm2h"],
            how="left",
        )
        if not unmatched_regions.empty:
            logging.warning(
                "The following homogenised region names could not be matched: "
                f"{unmatched_regions}"
            )
        df.drop(columns=["adm1h", "adm2h"], inplace=True)
    else:
        unmatched_regions = (
            df[df[admin2_col].isna()]
            .drop_duplicates(subset=["adm2h"])
            .sort_values(by=["adm2h"])
        )
        unmatched_regions = unmatched_regions.merge(
            ref_map[["adm2h", "NAME_2"]],
            on=["adm2h"],
            how="left",
        )
        if not unmatched_regions.empty:
            logging.warning(
                "The following homogenised region names could not be matched: "
                f"{unmatched_regions}"
            )
        df.drop(columns=["adm2h"], inplace=True)

    # Add GID_1 and GID_2 columns
    logging.info("Merging with reference names...")
    if use_admin1:
        df = df.merge(
            # Need to include both donating columns, and pairing columns
            ref_names[["GID_1", "GID_2", "NAME_1", "NAME_2"]],
            left_on=[admin1_col, admin2_col],
            right_on=["NAME_1", "NAME_2"],
            how="left",
        )
        df = df.drop(columns=["NAME_1", "NAME_2"])
    else:
        df = df.merge(
            # Need to include both donating columns, and pairing columns
            ref_names[["GID_2", "NAME_2"]],
            left_on=[admin2_col],
            right_on=["NAME_2"],
            how="left",
        )
        df = df.drop(columns=["NAME_2"])

    df.dropna(inplace=True)

    logging.info("Alignment complete.")
    logging.info(f"After merging with admin2 regions, there are {len(df)} records.")
    if use_admin1:
        unique_pairs = df[[admin1_col, admin2_col]].drop_duplicates()
        logging.info(
            f"After alignment, there are {len(unique_pairs)} unique Admin-1/Admin-2 region pairs."
        )
    else:
        logging.info(
            f"After alignment, there are {df[admin2_col].nunique()} unique Admin-2 regions."
        )
    return df


def refresh_plugins(verbose: bool = False) -> dict[str, type]:
    """Import source modules so plugins self-register.

    Returns the ``{ref: class}`` registry mapping.
    """
    plugins = load_plugins()
    if verbose:
        print("Source plugins loaded:")
        for ref in source_registry.names():
            print(f" - [{ref}] {source_registry.get(ref).name}")
    logging.info("Plugins loaded: " + ", ".join(source_registry.names()))
    return plugins


def _ensure_plugins_loaded() -> None:
    if not source_registry.names():
        load_plugins()


def _freq_day_scale(freq: str) -> int:
    """Rough days-per-period for a pandas frequency (for granularity comparison)."""
    f = freq.upper()
    if f.startswith("D"):
        return 1
    if f.startswith("W"):
        return 7
    if f.startswith(("M", "B", "Q")):
        return 30
    return 365


def interpolate_covariates(
    df: pd.DataFrame,
    cols: list[str],
    gid_col: str = "GID_2",
    method: str = "linear",
) -> tuple[pd.DataFrame, int]:
    """Interpolate sparse covariate columns onto the full case grid, per GID.

    `df["Date"]` is expected to be a Period column. Each column's known points
    are reindexed onto the GID's period grid and interpolated; grid edges are
    filled with the nearest value so every row is populated.

    method: "linear"/"time" (default, smooth), "ffill"/"pad", "bfill"/"backfill",
    or any other method accepted by ``pandas.Series.interpolate``.

    Returns ``(df, n_filled)`` where n_filled is the number of rows filled.
    """
    out = df.copy()
    n_filled = 0
    for _, g in out.groupby(gid_col, observed=False):
        ts = pd.PeriodIndex(g["Date"]).to_timestamp(how="end")
        idx = g.index
        for col in cols:
            if col not in out.columns:
                continue
            s = pd.Series(g[col].to_numpy(), index=ts)
            known = s.notna()
            if not known.any():
                continue
            if known.sum() == 1:
                filled = s.ffill().bfill()
            elif method.lower() in ("ffill", "pad", "forward"):
                filled = s.ffill().bfill()
            elif method.lower() in ("bfill", "backfill", "back"):
                filled = s.bfill().ffill()
            else:
                filled = s.interpolate(method="time" if method == "linear" else method)
                filled = filled.ffill().bfill()
            n_filled += int((~known).sum())
            out.loc[idx, col] = filled.to_numpy()
    return out, n_filled


def merge_geo_sources(
    df: pd.DataFrame, sources: list[str], method: str = "linear"
) -> pd.DataFrame:
    """
    Add source information to the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame to which source information will be added.
    sources (list[str]): List of sources to be added. Format: ['origin.field']
                         where field may be '*', e.g. ['worldclim.*', 'edo.spi6'].
    method (str): Interpolation method used when a source's granularity is
                  coarser than the case-data frequency (e.g. monthly sources on
                  a weekly grid). Default "linear"; also "ffill"/"bfill".
    """
    _ensure_plugins_loaded()

    # Collate source information
    d_sources: dict[str, list[str]] = {}
    for source in sources:
        if "." not in source:
            raise ValueError(
                "Source format must be 'origin.field', "
                "e.g. 'worldclim.*' or 'edo.spi6'."
            )
        origin, field = source.split(".", 1)
        d_sources.setdefault(origin, []).append(field)

    for origin, fields in d_sources.items():
        plugin = source_registry.get(origin)()
        orig_cols = set(df.columns)
        merged = plugin.merge(df, metrics=fields)
        new_cols = [c for c in merged.columns if c not in orig_cols]
        if not new_cols or not isinstance(merged["Date"].dtype, pd.PeriodDtype):
            df = merged
            continue

        case_freq = period_freq_str(merged["Date"].dtype)
        granularity = getattr(plugin, "granularity", "M")
        if _freq_day_scale(case_freq) < _freq_day_scale(granularity):
            merged, n_filled = interpolate_covariates(merged, new_cols, method=method)
            if n_filled:
                warnings.warn(
                    f"Source '{origin}' is {granularity}-granular; interpolated "
                    f"{n_filled} covariate value(s) onto the {case_freq} case grid "
                    f"(method='{method}').",
                    UserWarning,
                    stacklevel=2,
                )
        df = merged

    return df


def add_incidence_rate(
    df: pd.DataFrame,
    target_col: str = "DIR",
    cases_col: str = "Cases",
    pop_col: str = "pop_count",
    cases_per: int = 1e5,
) -> pd.DataFrame:
    df[target_col] = cases_per * df[cases_col] / df[pop_col]
    return df


def convert_to_incidence_rate(
    df: pd.DataFrame,
    df_pop: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert the DataFrame to incidence rate format.

    Parameters:
    df (pd.DataFrame): The DataFrame containing cases.
    df_pop (pd.DataFrame): The DataFrame containing population counts with columns

    Returns:
    pd.DataFrame: The DataFrame with an additional column for incidence rate.
    """
    if "pop_count" in df.columns:
        df.drop(columns=["pop_count"], inplace=True)
    df_with_pop = df.merge(
        df_pop[["Date", "GID_2", "pop_count"]],
        on=["Date", "GID_2"],
        how="left",
    )
    cases_per = 1e5
    df_with_pop = add_incidence_rate(
        df_with_pop,
        target_col="Cases",
        cases_col="Cases",
        pop_col="pop_count",
        cases_per=cases_per,
    )
    df_with_pop = add_incidence_rate(
        df_with_pop,
        target_col="prediction",
        cases_col="prediction",
        pop_col="pop_count",
        cases_per=cases_per,
    )
    df_with_pop.drop(columns=["pop_count"], inplace=True)
    return df_with_pop


def pad_admin2(
    df: DataFrame | pd.DataFrame,
    *,
    geo_col: str = "GID_2",
    geo_parent: str = "GID_1",
    iso3: str | None = None,
) -> DataFrame:
    """
    Ensure all Admin-2 regions are included in the DataFrame, even those with zero
    cases.
    """

    if isinstance(df, DataFrame):
        df = df.df  # convert to pandas DataFrame (quick fix, consider function rewrite)

    if geo_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{geo_col}' column.")

    # Get unique Admin-2 regions
    if iso3 is None:
        # Fall back to deriving the ISO3 from the first geo code (GADM-native).
        iso3 = str(df[geo_col].iloc[0])[:3]
    unique_admin2 = df[geo_col].unique()
    all_admin2 = get_admin2_list(iso3)

    # Find missing Admin-2 regions
    missing_admin2 = set(all_admin2[geo_col].unique()) - set(unique_admin2)
    date_list = list(df["Date"].drop_duplicates().sort_values())
    n_dates = len(date_list)

    # Create a DataFrame for missing regions with zero cases
    missing_df = []
    for adm2 in missing_admin2:
        df_entry = pd.DataFrame(
            {
                "Date": date_list,
                geo_parent: [
                    all_admin2[geo_parent][all_admin2[geo_col] == adm2].values[0]
                ]
                * n_dates,
                geo_col: [adm2] * n_dates,
                "Cases": [0] * n_dates,
            }
        )
        if "ADM1" in df.columns:
            df_entry["ADM1"] = all_admin2["NAME_1"][all_admin2[geo_col] == adm2].values[
                0
            ]
        if "ADM2" in df.columns:
            df_entry["ADM2"] = all_admin2["NAME_2"][all_admin2[geo_col] == adm2].values[
                0
            ]
        missing_df.append(df_entry)

    # Concatenate the original DataFrame with the missing regions
    result = (
        pd.concat([df, *missing_df], ignore_index=True)
        .sort_values(["Date", geo_col])
        .reset_index(drop=True)
    )
    result.sort_values(by=["Date", geo_col], inplace=True)

    # Convert to Thucia DataFrame and clean up
    out = DataFrame(df=result)
    del result
    return out


def merge_sources(df, covars: list[str], method: str = "linear") -> pd.DataFrame:
    """
    Merge geographic and climatological covariates into the main DataFrame.

    `method` is the interpolation method used when a source's granularity is
    coarser than the case-data frequency (see merge_geo_sources).
    """
    categorical_covars = ["GID_1", "GID_2", "ADM1", "ADM2", "Status"]
    for covar in covars:
        df_covar = merge_geo_sources(df, [covar], method=method)
        for cat in categorical_covars:
            if cat in df_covar.columns:
                df_covar[cat] = df_covar[cat].astype("category")
        merge_vars = list(
            set(["GID_2", "Date"])
            | (set(df_covar.columns.tolist()) - set(df.columns.tolist()))
        )
        logging.info("Performing merge with variables: " + ", ".join(merge_vars))
        df = df.merge(df_covar[merge_vars], on=["GID_2", "Date"], how="left")
        logging.info(f"After merging {covar}, there are {len(df)} records.")
    return df
