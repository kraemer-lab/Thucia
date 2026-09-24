import logging
import re
import subprocess
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from thucia.core.fs import DataFrame
from thucia.core.fs import read_db  # noqa: F401
from thucia.core.fs import read_nc  # noqa: F401
from thucia.core.fs import read_zarr  # noqa: F401
from thucia.core.fs import write_db  # noqa: F401
from thucia.core.fs import write_nc  # noqa: F401
from thucia.core.fs import write_zarr  # noqa: F401
from thucia.core.quantiles import quantiles as default_quantiles

from .wis import wis_bracher


def period_freq_str(dtype: pd.PeriodDtype) -> str:
    """Period-valid frequency string (e.g. 'M', not 'ME') from a Period dtype."""
    return re.search(r"period\[(.+)\]", str(dtype.name)).group(1)


def cases_per_month(*args, **kwargs) -> pd.DataFrame:
    return aggregate_cases(*args, **kwargs, freq="M")


def aggregate_cases(
    df: DataFrame | pd.DataFrame,
    statuses: list[str] | None = None,
    cases_col: str = "Cases",
    cutoff_date: pd.Timestamp | str | None = None,
    geo_col: str | None = "GID_2",
    freq: str = "M",
) -> pd.DataFrame:
    """
    Count how many times each exact row appears, grouped by (period) Date

    Parameters
    ----------
    df : DataFrame | pd.DataFrame
        DataFrame with a 'Date' column and optionally a 'Status' column.
    statuses : list[str], optional
        Subset of Status values to include.
    cases_col : str, optional
    cutoff_date : str, optional
    geo_col : str, optional
        If provided, will fill missing Dates, grouped by 'geo_col'.
    freq: str
        Resampling period for dates. Default is 'M' (month end). Typical options:
        - 'M': Month end
        - 'W-SUN': Week ending Sunday (Mon-Sun) [ISO week]
        - 'W-SAT': Week ending Saturday (Sun-Sat) [CDC epidemiological week]

    Returns
    -------
    DataFrame
        Reference to temporary Thucia DataFrame, with deduplicated rows + 'Cases' count.
    """

    # Convert to pandas DataFrame (for now)
    if isinstance(df, DataFrame):
        df = df.df  # load full pandas DataFrame
    else:
        df = df.copy()  # copy of input DataFrame

    if "Date" not in df.columns:
        raise ValueError("DataFrame must contain a 'Date' column.")

    if statuses is not None:
        if "Status" not in df.columns:
            raise ValueError(
                "DataFrame must contain a 'Status' column to use `statuses`."
            )
        df = df[df["Status"].isin(statuses)].copy()

    cutoff_date = pd.to_datetime(cutoff_date) if cutoff_date is not None else None
    if cutoff_date is not None:
        mask = pd.to_datetime(df["Date"]) <= cutoff_date
        if cases_col in df.columns:
            logging.info(
                f"Removing n={df[~mask][cases_col].sum()} cases occurring "
                f"after cutoff date {cutoff_date.date()}, remaining cases: "
                f"{df[mask][cases_col].sum()}"
            )
        else:
            logging.info(
                f"Removing n={len(df[~mask])} records occurring after "
                f"cutoff date {cutoff_date.date()}, remaining records: "
                f"{len(df[mask])}"
            )
        df = df[mask]

    if cases_col in df.columns:
        incoming_case_count = df[cases_col].sum()
    else:
        incoming_case_count = len(df)

    if not isinstance(df["Date"].dtype, pd.PeriodDtype):
        df["Date"] = pd.to_datetime(df["Date"]).dt.to_period(freq)

    # Prepare columns to group by (exclude Status)
    group_cols = ["Date", geo_col]

    # Group and count cases
    if cases_col not in df.columns:
        df[cases_col] = 1
    grouped = df.groupby(group_cols, as_index=False, observed=False)[cases_col].sum()

    if geo_col is None:
        return grouped.sort_values(by="Date").reset_index(drop=True)

    # Build full date range
    full_periods = pd.period_range(
        df["Date"].min(),
        df["Date"].max(),
        freq=df["Date"].dtype.freq,
    )

    if geo_col not in group_cols:
        raise ValueError(f"Expected '{geo_col}' column in DataFrame for full coverage.")

    unique_fill_values = df[geo_col].unique()

    # Full grid of all geo_col x Date combos
    full_grid = pd.DataFrame(
        list(product(unique_fill_values, full_periods)), columns=[geo_col, "Date"]
    )

    # Merge grouped counts onto full grid
    result = pd.merge(full_grid, grouped, on=[geo_col, "Date"], how="left")

    # Fill cases_col with zeros where missing
    result[cases_col] = result[cases_col].fillna(0).astype(int)

    # Identify descriptive columns to fill (all except geo_col, Date, cases_col, and Status)
    descriptive_cols = [
        col for col in df.columns if col not in [geo_col, "Date", cases_col, "Status"]
    ]

    # For each descriptive column, build a mapping from geo_col to the unique value,
    # then map/fill in the result
    for col in descriptive_cols:
        # Get unique mapping from geo_col to col value (assumes 1 unique value per geo_col)
        mapping = df.drop_duplicates(subset=[geo_col])[[geo_col, col]].set_index(
            geo_col
        )[col]
        result[col] = result[geo_col].map(mapping)

    # Sort results
    result = result.sort_values(by=["Date", geo_col]).reset_index(drop=True)

    # Check case counts match
    outgoing_case_count = result[cases_col].sum()
    if incoming_case_count != outgoing_case_count:
        logging.warning(
            f"Case count mismatch: incoming {incoming_case_count}, outgoing {outgoing_case_count}"
        )

    # Convert to Thucia DataFrame and clean up
    return DataFrame(df=result)


def _filter_and_separate(df, pred_col, true_col, transform=None, df_filter: dict = {}):
    if df_filter is None:
        df_filter = {}
    for key, value in df_filter.items():
        df = df[df[key] == value]

    y_true = df[true_col]
    y_pred = df[pred_col]
    kk = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[kk]
    y_pred = y_pred[kk]

    if transform is not None:
        y_true = transform(y_true)
        y_pred = transform(y_pred)
    return y_true, y_pred


def r2_score(y_true, y_pred):
    kk = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[kk]
    y_pred = y_pred[kk]

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        # Constant target: R^2 is undefined; follow sklearn's convention (0.0)
        # so downstream aggregates don't propagate inf/nan.
        return 0.0
    r2x = 1 - (ss_res / ss_tot)
    return r2x


def r2(df, pred_col, true_col, group_col=None, transform=None, df_filter: dict = {}):
    if not group_col:
        logging.warning(
            "No group_col provided for r2 calculation, returning overall r2 score."
        )
        y_true, y_pred = _filter_and_separate(
            df, pred_col, true_col, transform, df_filter
        )
        return r2_score(y_true, y_pred)

    if "quantile" in df.columns:
        df = df[df["quantile"] == 0.5]
    if df_filter is None:
        df_filter = {}
    for key, value in df_filter.items():
        df = df[df[key] == value]

    r2_gid = pd.DataFrame(columns=[group_col, "R2"])
    groups = df[group_col].unique()
    parts = []
    for group in groups:
        dfg = df[df[group_col] == group].copy()
        parts.append(
            pd.DataFrame(
                {
                    group_col: [group],
                    "R2": [
                        r2_score(
                            dfg[true_col],
                            dfg[pred_col],
                        )
                    ],
                }
            )
        )
    if parts:
        r2_gid = pd.concat(parts, ignore_index=True)
    return r2_gid


def rmse_score(y_true, y_pred):
    kk = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[kk]
    y_pred = y_pred[kk]
    mse = np.mean((y_true - y_pred) ** 2)
    return np.sqrt(mse)


def rmse(df, pred_col, true_col, group_col=None, transform=None, df_filter: dict = {}):
    if not group_col:
        logging.warning(
            "No group_col provided for rmse calculation, returning overall rmse score."
        )
        y_true, y_pred = _filter_and_separate(
            df, pred_col, true_col, transform, df_filter
        )
        return rmse_score(y_true, y_pred)

    if "quantile" in df.columns:
        df = df[df["quantile"] == 0.5]
    if df_filter is None:
        df_filter = {}
    for key, value in df_filter.items():
        df = df[df[key] == value]

    rmse_gid = pd.DataFrame(columns=[group_col, "RMSE"])
    groups = df[group_col].unique()
    for group in groups:
        dfg = df[df[group_col] == group].copy()
        rmse_gid = pd.concat(
            [
                rmse_gid,
                pd.DataFrame(
                    {
                        group_col: [group],
                        "RMSE": [
                            rmse_score(
                                dfg[true_col],
                                dfg[pred_col],
                            )
                        ],
                    }
                ),
            ],
            ignore_index=True,
        )
    return rmse_gid["RMSE"]


def wis(
    df,
    pred_col,
    true_col,
    geo_col="GID_2",
    group_col=None,
    transform=None,
    df_filter: dict = {},
):
    if group_col is not None:
        logging.warning(
            "Group_col provided for wis calculation, but WIS is computed over all groups. Ignoring group_col."
        )

    df = df[[geo_col, "Date", "quantile", pred_col, true_col]]
    if transform is not None:
        df.loc[:, true_col] = transform(df[true_col])
        df.loc[:, pred_col] = transform(df[pred_col])
    wis = wis_bracher(
        df=df,
        group_cols=(geo_col, "Date"),
        quantile_col="quantile",
        pred_col=pred_col,
        obs_col=true_col,
        log1p_scale=False,
        clamp_negative_to_zero=(not transform),
        monotonic_fix=True,
    )
    # Average over Date
    # wis_gid = wis.groupby(geo_col).mean().reset_index()
    # wis_gid = wis_gid.drop(columns=["Date"])
    return wis


def run_job(cmd: list[str], cwd: str | None = None) -> None:
    """
    Run a command in a subprocess and wait for it to finish.
    """

    result = subprocess.run(cmd, cwd=cwd, check=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with return code {result.returncode}")


def prepare_pdfm_embeddings(
    pdfm_filename: str | Path,
    provinces: list[str] | None = None,
    geo_col: str = "GID_2",
) -> pd.DataFrame:
    """Load user-supplied PDFM embeddings from a NetCDF file.

    PDFM embeddings are **not publicly distributed**; the file must be provided
    by the user. Expected schema: one row per admin region with a geo code
    column (default ``GID_2``) plus ``feature0``..``feature329`` embedding
    columns. The embedded dimensions are batched:

        0-127    Aggregated Search Trends
        128-255  Maps and Busyness
        256-329  Weather & Air Quality

    When ``provinces`` is given, only rows whose geo code is listed are kept.
    Duplicate geo codes are an encoding error: a warning is emitted and the
    first occurrence of each geo code is kept.
    """
    df = read_nc(pdfm_filename)
    if geo_col not in df.columns:
        raise ValueError(f"Embeddings file must contain a '{geo_col}' column.")
    if provinces is not None:
        df = df[df[geo_col].isin(provinces)]
    n_dupes = int(df[geo_col].duplicated().sum())
    if n_dupes:
        warnings.warn(
            f"Embeddings contain {n_dupes} duplicate '{geo_col}' row(s); "
            "this indicates an encoding error. Keeping the first occurrence "
            "of each geo code.",
            UserWarning,
            stacklevel=2,
        )
    return df.drop_duplicates(subset=[geo_col])


def prepare_embeddings(filename: str, embedding_type="pdfm") -> pd.DataFrame:
    if embedding_type == "pdfm":
        # Filter embeddings
        # Dimensions:   0-127 are used to reconstruct the Aggregated Search Trends
        #             128-255 are used to reconstruct the Maps and Busyness
        #             256-329 are used for Weather & Air Quality
        return prepare_pdfm_embeddings(filename)
    else:
        raise ValueError(f"Unknown embedding type: {embedding_type}")


def align_date_types(
    source_dates: pd.Series | pd.Timestamp,
    target_dates: pd.Series,
) -> pd.Series:
    """
    Align the date types of source_dates to match target_dates.
    If target_dates is a PeriodIndex, convert source_dates to PeriodIndex with same freq.
    If target_dates is a DatetimeIndex, convert source_dates to DatetimeIndex.

    Parameters
    ----------
    source_dates : pd.Series | pd.Timestamp
        Series of dates to be aligned.
    target_dates : pd.Series
        Series of dates to align to.

    Returns
    -------
    pd.Series | pd.Timestamp
        Foramt aligned dates.
    """
    if isinstance(source_dates, pd.Series):
        if isinstance(target_dates.dtype, pd.PeriodDtype):
            freq = period_freq_str(target_dates.dtype)
            if isinstance(source_dates.dtype, pd.PeriodDtype):
                # Source is already Period, just ensure same freq
                source_dates = source_dates.dt.asfreq(freq)
            else:
                # Convert to datetime, then period
                source_dates = pd.to_datetime(source_dates).dt.to_period(freq)
        else:
            source_dates = pd.to_datetime(source_dates)
    elif isinstance(source_dates, pd.Timestamp):
        if isinstance(target_dates.dtype, pd.PeriodDtype):
            freq = period_freq_str(target_dates.dtype)
            source_dates = source_dates.to_period(freq)
        else:
            source_dates = pd.to_datetime(source_dates)
    return source_dates


def check_index_combinations(df: pd.DataFrame, group_cols):
    """Check that all combinations of group_cols are present in df."""
    product = []
    for c in group_cols:
        product.append(df[c].unique())
    all_idx = pd.MultiIndex.from_product(product, names=group_cols)
    present_idx = pd.MultiIndex.from_frame(df[group_cols].drop_duplicates())
    missing = all_idx.difference(present_idx)
    if not missing.empty:
        raise ValueError(f"Missing index combinations in data:\n{missing}")


def check_covars_for_nans(df, cols):
    if df[cols].isnull().any().any():
        missing = df[df[cols].isnull().any(axis=1)]
        raise ValueError(f"Missing covariate values in data:\n{missing}")


# --- helper to prepare shared draws (S, E, and optional U for independent) ---
def prepare_shared_draws(k_max, N, seed=3, dtype=np.float32, mode="copula"):
    """
    Prepare reusable base draws for Common Random Numbers (CRN).
    Returns a dict with keys:
      - 'N': N
      - 'S': shared standard normals of shape (N,) for copula factor (float32)
      - 'E': idiosyncratic normals of shape (k_max, N) for copula (float32)
      - 'U': independent uniforms of shape (k_max, N) if mode == "independent" (float32)
    """
    rng = np.random.default_rng(seed)
    out = {"N": int(N)}
    if mode in ("copula", "both"):
        out["S"] = rng.standard_normal(size=(N,)).astype(dtype)
        out["E"] = rng.standard_normal(size=(k_max, N)).astype(dtype)
    if mode in ("independent", "both"):
        out["U"] = rng.random((k_max, N)).astype(dtype)
    return out


# --- optimized quantile sum with optional shared draws ---
def quantile_sum_fast(
    df,
    date,
    gids,
    horizon,
    geo_col="GID_2",
    N=50000,
    mode="independent",
    rho=0.3,
    seed=3,
    probabilities=None,
    chunk_size=5000,
    dtype=np.float32,
    shared_draws=None,
):
    """
    Memory- and speed-optimized version of quantile_sum using:
      - chunked sampling (reduces peak memory)
      - equicorrelation trick for Gaussian copula
      - optional reuse of shared_draws (dict from prepare_shared_draws)

    Parameters:
      - df: quantiles table (columns: Date, geo_col, horizon, quantile, prediction)
      - date, gids, horizon: filters
      - mode: "independent" | "comonotonic" | "copula"
      - shared_draws: optional dict with keys 'N', 'S', 'E', 'U' prepared for a large k_max.
                      If provided, will slice E/U to the required k and reuse S.
    """
    if probabilities is None:
        probabilities = list(default_quantiles)

    k = len(gids)
    rng = np.random.default_rng(seed)

    # prepare inverse-CDF info for each gid (list of probs, vals)
    probs_list = []
    vals_list = []
    for gid in gids:
        g = df[
            (df.Date == date) & (df[geo_col] == gid) & (df.horizon == horizon)
        ].sort_values("quantile")
        if g.shape[0] == 0:
            raise ValueError(
                f"Missing quantiles for {gid} on {date}, horizon {horizon}"
            )
        probs = g["quantile"].to_numpy()
        vals = g["prediction"].to_numpy()
        # extend flat tails
        if probs[0] > 0.0:
            probs = np.insert(probs, 0, 0.0)
            vals = np.insert(vals, 0, vals[0])
        if probs[-1] < 1.0:
            probs = np.append(probs, 1.0)
            vals = np.append(vals, vals[-1])
        probs_list.append(probs)
        vals_list.append(vals)

    # If shared_draws provided, sanity-check
    if shared_draws is not None:
        if shared_draws.get("N", None) != N:
            # mismatch: either regenerate or raise; here we regenerate to be tolerant
            logging.warning(
                "shared_draws.N != N; ignoring shared_draws and regenerating."
            )
            shared_draws = None

    totals = np.empty(N, dtype=dtype)
    n_done = 0

    # Optionally pre-extract shared S/E/U slices for this k to avoid repeated indexing cost
    if shared_draws is not None:
        S_global = shared_draws.get("S", None)
        E_global = shared_draws.get("E", None)
        U_global = shared_draws.get("U", None)
        if E_global is not None:
            # E_global shape (k_max, N) -> slice first k rows
            E_slice = E_global[:k, :]
        else:
            E_slice = None
        if U_global is not None:
            U_slice = U_global[:k, :]
        else:
            U_slice = None
    else:
        S_global = E_slice = U_slice = None

    while n_done < N:
        this_chunk = min(chunk_size, N - n_done)

        # Build U_chunk depending on mode
        if mode == "comonotonic":
            start = n_done + 1
            end = n_done + this_chunk
            u_chunk = np.linspace(
                start / (N + 1), end / (N + 1), this_chunk, dtype=dtype
            )
            # We'll broadcast u_chunk to each marginal
        elif mode == "independent":
            if U_slice is not None:
                # reuse precomputed independent uniforms (slice columns)
                U_chunk = U_slice[:, n_done : n_done + this_chunk]
            else:
                U_chunk = rng.random((k, this_chunk)).astype(dtype)
        elif mode == "copula":
            if not (-0.999 < rho < 0.999):
                raise ValueError("rho must be between -0.999 and 0.999")
            if abs(rho) < 1e-12:
                # near zero -> independent uniforms
                if U_slice is not None:
                    U_chunk = U_slice[:, n_done : n_done + this_chunk]
                else:
                    U_chunk = rng.random((k, this_chunk)).astype(dtype)
            else:
                sqrt_rho = np.sqrt(max(0.0, rho))
                sqrt_1_rho = np.sqrt(max(0.0, 1.0 - rho))
                # use shared S/E if provided, else generate new
                if S_global is not None and E_slice is not None:
                    S = S_global[n_done : n_done + this_chunk]  # (this_chunk,)
                    E = E_slice[:, n_done : n_done + this_chunk]  # (k,this_chunk)
                else:
                    S = rng.standard_normal(size=(this_chunk,)).astype(dtype)
                    E = rng.standard_normal(size=(k, this_chunk)).astype(dtype)
                # build correlated normals Z = sqrt_rho * S + sqrt_1_rho * E
                Z_chunk = (sqrt_rho * S[np.newaxis, :]) + (sqrt_1_rho * E)
                U_chunk = norm.cdf(Z_chunk).astype(dtype)
        else:
            raise ValueError("mode must be 'independent', 'comonotonic' or 'copula'")

        # inverse-CDF mapping and accumulation for this chunk
        chunk_totals = np.zeros(this_chunk, dtype=dtype)
        for i in range(k):
            if mode == "comonotonic":
                u_i = u_chunk
            else:
                u_i = U_chunk[i, :]
            # np.interp: convert to float64 for numerical stability then cast back
            mapped = np.interp(
                u_i.astype(np.float64),
                probs_list[i].astype(np.float64),
                vals_list[i].astype(np.float64),
            )
            chunk_totals += mapped.astype(dtype)

        totals[n_done : n_done + this_chunk] = chunk_totals
        n_done += this_chunk

    qvals = np.quantile(totals.astype(np.float64), probabilities)

    return pd.DataFrame(
        {
            "Date": [date] * len(probabilities),
            "horizon": [horizon] * len(probabilities),
            "quantile": probabilities,
            "prediction": qvals,
        }
    )


# --- top-level function that iterates over geo_parent groups and reuses shared draws ---
def quantile_sum_gid(
    df,
    db_file: str,
    new_file: bool = True,
    samples: int = 10000,
    chunk_size: int = 5000,
    dtype=np.float32,
    quantiles=None,
    rho=0.3,
    seed_base: int = 3,
    geo_col: str = "GID_2",
    geo_parent: str = "GID_1",
):
    """
    Process all geo_parent groups in df and append aggregated quantiles to a database
    backing store (DataFrame(db_file=...)).

      - Prepares shared draws (S,E and U) once using the maximum k across groups,
        then reuses slices for each group (massive speed/memory benefit).
      - Allows chunking and float32.
      - Keeps identical API for output and merges Cases/Log_Cases as before.

    Parameters:
      - df: input long quantiles dataframe with columns Date, geo_parent, geo_col,
            horizon, quantile, prediction, Cases
      - db_file, new_file: storage for results (DataFrame wrapper assumed)
      - samples: number of Monte Carlo samples to use (N)
      - chunk_size: chunk size for sampling
      - dtype: np.float32 by default
      - quantiles: list of probabilities to return (default set used earlier)
      - rho: copula rho used for all groups here (scalar)
      - seed_base: RNG seed base for reproducibility
    """
    if not db_file:
        raise ValueError("db_file must be provided to store quantile sum results.")
    if quantiles is None:
        quantiles = df["quantile"].unique()

    tdf = DataFrame(db_file=db_file, new_file=new_file)

    # Determine max number of geo_col across all geo_parent (k_max) to prepare shared draws
    parents = df[geo_parent].unique()
    max_k = 0
    parent_counts = {}
    for parent in parents:
        cnt = df[df[geo_parent] == parent][geo_col].nunique()
        parent_counts[parent] = int(cnt)
        if cnt > max_k:
            max_k = int(cnt)
    if max_k == 0:
        logging.warning(f"No {geo_col} groups found in df; nothing to process.")
        return tdf

    # Prepare shared draws for reuse: both copula and independent available
    shared_draws = prepare_shared_draws(
        k_max=max_k, N=samples, seed=seed_base, dtype=dtype, mode="both"
    )

    # Process each geo_parent
    for parent in parents:
        logging.info(f"Processing {geo_parent}={parent}")
        df_parent = df[df[geo_parent] == parent]
        gids = df_parent[geo_col].unique()
        # use slice of shared_draws for this group
        # (quantile_sum_fast will slice shared_draws internally)
        horizons = df_parent["horizon"].unique()
        for horizon in horizons:
            dates = df_parent["Date"].unique()
            for date in dates:
                logging.info(
                    f"Processing {geo_parent}={parent}, horizon={horizon}, date={date}"
                )
                try:
                    entry = quantile_sum_fast(
                        df=df_parent,
                        date=date,
                        gids=gids,
                        horizon=horizon,
                        geo_col=geo_col,
                        N=samples,
                        mode="copula",
                        rho=rho,
                        seed=seed_base,  # deterministic: shared_draws already seeded
                        probabilities=quantiles,
                        chunk_size=chunk_size,
                        dtype=dtype,
                        shared_draws=shared_draws,
                    )

                    # add geo_parent column (categorical with consistent categories)
                    entry[geo_parent] = parent
                    entry[geo_parent] = entry[geo_parent].astype("category")
                    entry[geo_parent] = entry[geo_parent].cat.set_categories(parents)

                    # Add Sum of Cases over geo_col for each Date (like original code)
                    cases = df_parent.groupby(
                        ["Date", geo_col],
                        observed=True,
                    ).aggregate({"Cases": "first"})
                    cases = (
                        cases.groupby(
                            ["Date"],
                            observed=True,
                        )
                        .aggregate({"Cases": "sum"})
                        .reset_index()
                    )
                    entry = entry.merge(cases, on=["Date"], how="left")
                    entry["Log_Cases"] = np.log1p(entry["Cases"].fillna(0.0))

                    tdf.append(entry)
                    # free memory if needed
                    del entry

                except ValueError as e:
                    logging.warning(
                        f"Skipping quantile sum for {geo_parent}={parent}, date={date}, horizon={horizon}: {e}"
                    )

    return tdf
