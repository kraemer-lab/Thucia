import logging
import re
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


def _region_roster(
    regions: pd.DataFrame,
    *,
    geo_col: str,
    geo_parent: str | None,
    region_col: str | None,
    parent_col: str | None,
    adm1_col: str,
    adm2_col: str,
    name1_col: str | None,
    name2_col: str | None,
) -> pd.DataFrame:
    """Normalise a region roster onto the caller's geo column names.

    ``regions`` may key its rows under any names (e.g. a GADM frame using
    ``GID_2``/``GID_1``/``NAME_1``/``NAME_2``); they are aliased onto
    ``geo_col``/``geo_parent``/``adm1_col``/``adm2_col`` so the padding logic is
    column-name agnostic.
    """
    rename = {region_col or geo_col: geo_col}
    if geo_parent is not None:
        rename.setdefault(parent_col or geo_parent, geo_parent)
    if name1_col is not None:
        rename.setdefault(name1_col, adm1_col)
    if name2_col is not None:
        rename.setdefault(name2_col, adm2_col)
    roster = regions.rename(columns=rename)
    wanted = [geo_col]
    if geo_parent is not None:
        wanted.append(geo_parent)
    for col in wanted:
        if col not in roster.columns:
            raise ValueError(
                f"Region list has no '{col}' column (roster keyed by "
                f"'{region_col or geo_col}'); cannot ensure all regions."
            )
    keep = wanted + [c for c in (adm1_col, adm2_col) if c in roster.columns]
    return roster[keep].drop_duplicates(geo_col)


def ensure_all_regions(
    df: DataFrame | pd.DataFrame,
    *,
    geo_col: str = "GID_2",
    geo_parent: str | None = "GID_1",
    iso3: str | None = None,
    regions: pd.DataFrame | None = None,
    region_col: str | None = None,
    parent_col: str | None = None,
    adm1_col: str = "ADM1",
    adm2_col: str = "ADM2",
    name1_col: str | None = "NAME_1",
    name2_col: str | None = "NAME_2",
) -> DataFrame:
    """
    Ensure every region in ``geo_col`` appears in the DataFrame, even those with
    zero cases in every period.

    The authoritative region list is resolved as:

    - ``regions``: a roster DataFrame such as a shapefile attribute table,
      keyed by ``region_col``/``parent_col`` (and the name columns when you want
      ADM1/ADM2 names carried into the padded rows);
    - the GADM admin-2 list for ``iso3`` (or the geo-code prefix) when
      ``regions`` is omitted and the geo codes are GADM-shaped — this applies to
      categorical codes too, since an aggregated frame's categories only ever
      reflect observed regions;
    - if ``df[geo_col]`` is a categorical with non-GADM codes, its
      ``.cat.categories`` form an implicit region list; or
    - otherwise a ``ValueError`` is raised — callers that want a subset simply do
      not run this function.
    """

    if isinstance(df, DataFrame):
        df = df.df

    if geo_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{geo_col}' column.")

    observed_code = str(df[geo_col].dropna().iloc[0])
    gadm_shaped = bool(re.match(r"^[A-Za-z0-9]{1,3}\.\d+(\.\d+)?_\d+$", observed_code))

    if regions is not None:
        roster = _region_roster(
            regions,
            geo_col=geo_col,
            geo_parent=geo_parent,
            region_col=region_col,
            parent_col=parent_col,
            adm1_col=adm1_col,
            adm2_col=adm2_col,
            name1_col=name1_col,
            name2_col=name2_col,
        )
    elif gadm_shaped:
        # GADM-shaped codes take the GADM admin-2 list even when the column is
        # categorical: aggregation re-derives categories from observed data, so
        # never-seen regions are invisible to `.cat.categories`.
        if iso3 is None:
            iso3 = observed_code[:3]
        roster = _region_roster(
            get_admin2_list(iso3),
            geo_col=geo_col,
            geo_parent=geo_parent,
            region_col="GID_2",
            parent_col="GID_1",
            adm1_col=adm1_col,
            adm2_col=adm2_col,
            name1_col="NAME_1",
            name2_col="NAME_2",
        )
    elif isinstance(df[geo_col].dtype, pd.CategoricalDtype):
        # Non-GADM categorical: the categories are the implicit region list.
        roster = pd.DataFrame({geo_col: list(df[geo_col].cat.categories)})
        for col in (geo_parent, adm1_col, adm2_col):
            if col is None or col not in df.columns:
                continue
            roster = roster.merge(
                df.drop_duplicates(geo_col)[[geo_col, col]].dropna(subset=[geo_col]),
                on=geo_col,
                how="left",
            )
        if geo_parent is not None and geo_parent not in roster.columns:
            roster[geo_parent] = None
    else:
        # No roster, not GADM-shaped, not categorical: no region list can be
        # resolved. Callers that want a subset simply do not run this function.
        raise ValueError(
            f"Cannot resolve the region list: '{geo_col}' has no regions= roster "
            f"and code {observed_code!r} does not look GADM-shaped, nor is the "
            "column categorical. Pass regions=... (e.g. a shapefile attribute "
            "table) or make the geo column a categorical to pad to its "
            "categories."
        )

    # Observed set comes from the *present* values: on a categorical, `.unique()`
    # drops unused categories, so the roster codes stay authoritative for the
    # no-miss guarantee.
    roster_codes = roster[geo_col].dropna().astype("object").unique()
    observed = set(df[geo_col].dropna().astype("object").unique())
    missing = [c for c in roster_codes if c not in observed]

    missing_frames = []
    if missing:
        dates = list(df["Date"].drop_duplicates().sort_values())
        n_dates = len(dates)
        roster_by_zone = roster.set_index(geo_col)

        def _value(zone, col):
            val = roster_by_zone.loc[zone, col]
            return val.iloc[0] if isinstance(val, pd.Series) else val

        for zone in missing:
            row = {
                "Date": dates,
                geo_col: [zone] * n_dates,
                "Cases": [0] * n_dates,
            }
            if geo_parent is not None:
                parent = _value(zone, geo_parent)
                # Omit the column when the parent is unknown: leaving it out makes
                # the concat fill NaN for those rows instead of passing an
                # all-NA column through concat (which pandas deprecates).
                if not pd.isna(parent):
                    row[geo_parent] = [parent] * n_dates
            for col in (adm1_col, adm2_col):
                if col in df.columns and col in roster_by_zone.columns:
                    val = _value(zone, col)
                    if not pd.isna(val):
                        row[col] = [val] * n_dates
            missing_frames.append(pd.DataFrame(row))

    result = (
        pd.concat([df, *missing_frames], ignore_index=True)
        if missing_frames
        else df.copy()
    )
    if isinstance(df[geo_col].dtype, pd.CategoricalDtype):
        categories = list(df[geo_col].cat.categories) + [
            c for c in roster_codes if c not in set(df[geo_col].cat.categories)
        ]
        result[geo_col] = pd.Categorical(
            result[geo_col].astype("object"), categories=categories
        )
    result = result.sort_values(["Date", geo_col]).reset_index(drop=True)

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
