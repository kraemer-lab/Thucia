import math
from pathlib import Path

import geopandas as gpd
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon
from thucia.core.fs import cache_folder


def _admin_level_col(admin_level: str | int | None, default: str = "GID_1") -> str:
    """Normalize an admin_level specifier to a GADM geo column name."""
    if isinstance(admin_level, int):
        return f"GID_{admin_level}"
    if admin_level is None:
        return default
    return admin_level


def boundary(
    country="BRA",
    ax=None,
    admin_level=1,
):
    admin_level = _admin_level_col(admin_level)
    if admin_level == "GID_1":
        layer = "ADM_ADM_1"
    elif admin_level == "GID_2":
        layer = "ADM_ADM_2"
    else:
        raise ValueError(f"Unsupported admin_level: {admin_level}")

    geo_filename = Path(cache_folder) / "geo" / country / f"gadm41_{country}.gpkg"
    gdf = gpd.read_file(geo_filename, layer=layer)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    gdf.boundary.plot(ax=ax, edgecolor="0.6", linewidth=0.5)
    if ax is None:
        plt.show()


def choropleth(
    df,
    ax=None,
    admin_level: str | int | None = 1,
    value_col="Cases",
    value_transform=None,
    cmap="viridis",
    aggregation="sum",
    edgecolor="0.6",
    linewidth=0.2,
    legend=True,
    symmetric_cmap=False,
    colorbar=True,
    vmin=None,
    vmax=None,
    colorbar_format="%.0f",
):
    admin_level = _admin_level_col(admin_level)
    if admin_level == "GID_1":
        layer = "ADM_ADM_1"
    elif admin_level == "GID_2":
        layer = "ADM_ADM_2"
    else:
        raise ValueError(f"Unsupported admin_level: {admin_level}")

    def passthrough(x):
        return x

    value_transform = passthrough if value_transform is None else value_transform

    countries = set(map(lambda x: x.split(".")[0], df[admin_level].unique()))
    if len(countries) > 1:
        raise ValueError(f"DataFrame contains multiple countries: {countries}")
    country = countries.pop()
    df = df.groupby(admin_level)[value_col].agg(aggregation).reset_index()
    df[value_col] = value_transform(df[value_col])
    geo_filename = Path(cache_folder) / "geo" / country / f"gadm41_{country}.gpkg"
    gdf = gpd.read_file(geo_filename, layer=layer)
    merged = gdf.merge(df, left_on=admin_level, right_on=admin_level, how="right")
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    if symmetric_cmap:
        lim = max(abs(merged[value_col].min()), merged[value_col].max())
        merged.plot(
            column=value_col,
            ax=ax,
            legend=False,
            cmap=cmap,
            edgecolor=edgecolor,
            linewidth=linewidth,
            aspect="auto",
            vmin=-lim,
            vmax=lim,
        )
    else:
        merged.plot(
            column=value_col,
            ax=ax,
            legend=False,
            cmap=cmap,
            edgecolor=edgecolor,
            linewidth=linewidth,
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
        )

    values = merged[value_col]

    if symmetric_cmap:
        lim = max(abs(values.min()), values.max())
        norm = mcolors.Normalize(vmin=-lim, vmax=lim)
    else:
        norm = mcolors.Normalize(vmin=values.min(), vmax=values.max())

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    if colorbar:
        cax = inset_axes(
            ax,
            width="3%",  # thickness
            height="50%",
            loc="lower left",
            bbox_to_anchor=(0, 0, 1, 1),  # 30% up
            bbox_transform=ax.transAxes,
            borderpad=0,
        )

        cbar = plt.colorbar(
            sm,
            cax=cax,
        )
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter(colorbar_format))

    if ax is None:
        plt.show()


def hexmap(
    df,
    ax=None,
    admin_level: str | int | None = 1,
    value_col="Cases",
    side_length_km: float = 20.0,
    method: str = "area",  # 'area' or 'centroid'
    cmap="viridis",
    aggregation="sum",
):
    """
    Create a contiguous hexmap from regional data.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain GID_1 (and GID_2 if admin_level=GID_2) and the value_col.
    admin_level : int|str|None
        1 or 2 (or 'GID_1'/'GID_2')
    side_length_km : float
        hex side length in kilometers (uses metric projection)
    method : str
        'area' for area-weighted aggregation (slower, recommended), or 'centroid' for centroid assignment (faster)
    Returns
    -------
    hexmap_gdf : geopandas.GeoDataFrame
        GeoDataFrame of hexes with aggregated `value` column.
    """
    admin_level = _admin_level_col(admin_level)
    if admin_level == "GID_1":
        layer = "ADM_ADM_1"
    elif admin_level == "GID_2":
        layer = "ADM_ADM_2"
    else:
        raise ValueError(f"Unsupported admin_level: {admin_level}")

    # Basic validations
    if "GID_1" not in df.columns:
        raise ValueError("Input df must contain 'GID_1' column to deduce country.")
    countries = set(map(lambda x: x.split(".")[0], df["GID_1"].unique()))
    if len(countries) > 1:
        raise ValueError(f"DataFrame contains multiple countries: {countries}")
    country = countries.pop()

    # aggregate df by admin_level
    df_agg = df.groupby(admin_level)[value_col].agg(aggregation).reset_index()

    # load geometry
    geo_filename = Path(cache_folder) / "geo" / country / f"gadm41_{country}.gpkg"
    regions_gdf = gpd.read_file(geo_filename, layer=layer)

    # join aggregated values to regions (keep only regions with data)
    regions_gdf = regions_gdf.merge(
        df_agg, left_on=admin_level, right_on=admin_level, how="right"
    )

    # project to metric CRS for consistent hex sizing
    regions_metric = regions_gdf.to_crs(epsg=3857)

    # helper: build flat-top hexagon centered at (cx,cy) with side length s (meters)
    def make_hex(cx, cy, s):
        pts = []
        # flat-top: start angle at 0
        for i in range(6):
            angle = math.radians(60 * i)
            x = cx + s * math.cos(angle)
            y = cy + s * math.sin(angle)
            pts.append((x, y))
        return Polygon(pts)

    # build hex grid covering bounds
    xmin, ymin, xmax, ymax = regions_metric.total_bounds
    s = side_length_km * 1000.0
    dx = 1.5 * s
    dy = math.sqrt(3) * s

    cols = int((xmax - xmin) / dx) + 3
    rows = int((ymax - ymin) / dy) + 3

    hex_geoms = []
    hex_ids = []
    for col in range(-1, cols):
        cx = xmin + col * dx
        y_offset = (col % 2) * (dy / 2.0)
        for row in range(-1, rows):
            cy = ymin + row * dy + y_offset
            hex_geoms.append(make_hex(cx, cy, s))
            hex_ids.append(f"h_{col}_{row}")

    hex_gdf = gpd.GeoDataFrame(
        {"hex_id": hex_ids, "geometry": hex_geoms}, crs="EPSG:3857"
    )

    if method == "centroid":
        # assign each region to the hex containing its centroid (faster)
        centroids = regions_metric.copy()
        centroids["geometry"] = centroids.geometry.centroid
        assigned = gpd.sjoin(
            hex_gdf,
            centroids[[admin_level, value_col, "geometry"]],
            how="left",
            predicate="contains",
        )
        # group by hex_id and aggregate
        hex_values = (
            assigned.groupby("hex_id")[value_col].agg("mean").reset_index()
        )  # mean by default; could use sum
        hex_values = hex_values.rename(columns={value_col: "value"})
        hexmap = hex_gdf.merge(hex_values, on="hex_id", how="left")
    else:
        # area-weighted aggregation (more accurate)
        # intersect hexes with regions
        # ensure unique hex_id column for overlay
        hex_gdf_reset = hex_gdf.reset_index(drop=True)
        # overlay intersection
        inter = gpd.overlay(
            regions_metric[[admin_level, value_col, "geometry"]],
            hex_gdf_reset[["hex_id", "geometry"]],
            how="intersection",
        )
        if inter.empty:
            raise RuntimeError(
                "Intersection produced no geometries — try larger side_length_km or check CRS/bounds."
            )
        inter["area"] = inter.geometry.area
        # weighted sum per hex: sum(value * area) / sum(area)
        weighted = (
            inter.groupby("hex_id")
            .apply(
                lambda d: pd.Series(
                    {
                        "value": (d[value_col] * d["area"]).sum() / d["area"].sum(),
                    }
                ),
                include_groups=False,
            )
            .reset_index()
        )
        hexmap = hex_gdf.merge(weighted, on="hex_id", how="left")

    # plot
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    hexmap.plot(
        column="value", ax=ax, legend=True, cmap=cmap, edgecolor="0.6", linewidth=0.2
    )
    ax.set_axis_off()
    if ax is None:
        plt.show()

    return hexmap


def hex_cartogram(
    df,
    ax=None,
    admin_level: str | int | None = 2,
    value_col="Cases",
    side_length_km: float = 20.0,
    aggregation="sum",
    cmap="viridis",
):
    """
    Create a pseudo-geographic contiguous hexmap with exactly one hex per admin2 (GID_2).
    Matches region centroids to a regular hex lattice via optimal assignment.

    Returns hex_gdf (GeoDataFrame with hex geometry and columns from regions).
    """
    # normalize admin_level
    admin_level = _admin_level_col(admin_level, default="GID_2")
    if admin_level != "GID_2":
        raise ValueError(
            "This function is intended for admin_level=2 (one hex per GID_2)."
        )

    # country detection (same as your choropleth)
    if "GID_1" not in df.columns:
        raise ValueError("df must contain GID_1 to detect country")
    countries = set(map(lambda x: x.split(".")[0], df["GID_1"].unique()))
    if len(countries) > 1:
        raise ValueError(f"DataFrame contains multiple countries: {countries}")
    country = countries.pop()

    # aggregate values per GID_2
    df_agg = df.groupby(admin_level)[value_col].agg(aggregation).reset_index()

    # load region geometries (GID_2)
    geo_filename = Path(cache_folder) / "geo" / country / f"gadm41_{country}.gpkg"
    regions_gdf = gpd.read_file(geo_filename, layer="ADM_ADM_2")

    # keep only regions we have values for
    regions_gdf = regions_gdf.merge(
        df_agg, left_on=admin_level, right_on=admin_level, how="inner"
    )

    # project to metric CRS for distances and hex sizing
    regions_metric = regions_gdf.to_crs(epsg=3857)
    centroids = regions_metric.geometry.centroid.reset_index(drop=True)
    n = len(centroids)
    if n == 0:
        raise ValueError("No GID_2 regions after merging; check your data.")

    # bounding box + margin
    xmin, ymin, xmax, ymax = regions_metric.total_bounds
    margin = 0.05  # 5% margin
    xpad = (xmax - xmin) * margin
    ypad = (ymax - ymin) * margin
    xmin -= xpad
    xmax += xpad
    ymin -= ypad
    ymax += ypad

    # hex parameters (meters)
    s = side_length_km * 1000.0
    dx = 1.5 * s
    dy = math.sqrt(3) * s

    # compute a grid large enough to cover extent and contain >= n hexes
    cols = int((xmax - xmin) / dx) + 6
    rows = int((ymax - ymin) / dy) + 6

    centers = []
    hex_polys = []
    ids = []
    for col in range(-3, cols):
        cx = xmin + col * dx
        y_offset = (col % 2) * (dy / 2.0)
        for row in range(-3, rows):
            cy = ymin + row * dy + y_offset
            # create center, polygon
            centers.append((cx, cy))
            pts = []
            for i in range(6):
                angle = math.radians(60 * i)
                x = cx + s * math.cos(angle)
                y = cy + s * math.sin(angle)
                pts.append((x, y))
            hex_polys.append(Polygon(pts))
            ids.append(f"h_{col}_{row}")

    centers = np.array(centers)  # M x 2
    hex_gdf_full = gpd.GeoDataFrame(
        {"hex_id": ids, "geometry": hex_polys}, crs="EPSG:3857"
    )

    # pick the best subset of hex centers to match the n centroids.
    # Strategy: pick the k nearest hex centers to the centroid cloud center,
    # then solve assignment between region centroids and those hex centers.
    cloud_center = np.mean(np.vstack([c.coords[0] for c in centroids]), axis=0)
    # compute distances of hex centers to cloud center
    hex_centers_arr = np.array(
        [poly.centroid.coords[0] for poly in hex_gdf_full.geometry]
    )
    d_to_cloud = np.linalg.norm(hex_centers_arr - cloud_center, axis=1)
    # sort hex indices by closeness to cloud center and select top K (K >= n)
    K = max(int(n * 1.5), n + 50)  # a buffer; tweak if needed
    idx_sorted = np.argsort(d_to_cloud)
    candidate_idx = idx_sorted[:K]
    candidate_centers = hex_centers_arr[candidate_idx]

    # construct cost matrix (n regions x K hexes)
    cent_arr = np.array([c.coords[0] for c in centroids])
    cost = np.linalg.norm(cent_arr[:, None, :] - candidate_centers[None, :, :], axis=2)

    # solve rectangular assignment using Hungarian: we need square cost for linear_sum_assignment;
    # so pad cost matrix to square with large values OR use K>=n and assign n rows to n columns by selecting best columns.
    # We'll use the classic approach: run linear_sum_assignment on the n x K matrix by turning each region as a "row" and selecting unique column assignment.
    # SciPy's linear_sum_assignment works with non-square matrices and returns min assignment with length = min(dimensions).
    row_ind, col_ind = linear_sum_assignment(cost)
    # row_ind size == n, col_ind size == n (assigned columns indices in candidate set)
    chosen_hex_idx = candidate_idx[col_ind]  # indices into hex_gdf_full

    # build final hex_gdf matched 1:1 to regions (preserve region order)
    chosen_hexes = hex_gdf_full.iloc[chosen_hex_idx].reset_index(drop=True)
    # attach region attributes in the same order as centroids (centroids are in regions_metric order)
    result = chosen_hexes.copy()
    # ensure matching order: centroids correspond to regions_metric rows
    result = pd.concat(
        [result, regions_metric.reset_index(drop=True)[[admin_level, value_col]]],
        axis=1,
    )
    result = gpd.GeoDataFrame(result, geometry="geometry", crs="EPSG:3857")

    # optional: set an attribute for original centroid distance (diagnostic)
    hex_centers_chosen = np.array([geom.centroid.coords[0] for geom in result.geometry])
    dists = np.linalg.norm(cent_arr - hex_centers_chosen, axis=1)
    result["centroid_dist_m"] = dists

    # project back to WGS84 for plotting if you wish
    result = result.to_crs(epsg=4326)

    # plot
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    result.plot(
        column=value_col, ax=ax, legend=True, cmap=cmap, edgecolor="grey", linewidth=0.4
    )
    ax.set_axis_off()
    if ax is None:
        plt.show()

    return result


def subset_regions(
    gdf,
    admin_level,
    region_filter=None,
):
    if region_filter is None:
        return gdf

    if callable(region_filter):
        return gdf[gdf[admin_level].apply(region_filter)]

    if isinstance(region_filter, str):
        region_filter = [region_filter]

    region_filter = set(region_filter)

    # direct match
    mask = gdf[admin_level].isin(region_filter)

    # hierarchical match:
    # selecting BRA.25_1 at admin2 keeps all BRA.25.XXX_1
    if admin_level == "GID_2":
        prefixes = {r.split("_")[0] for r in region_filter if r.count(".") == 1}
        if prefixes:
            mask |= gdf["GID_1"].isin(region_filter)
    return gdf[mask]


def adjacency_matrix(
    df=None,
    country=None,
    admin_level: str | int | None = 1,
    region_filter=None,
    contiguity: str = "queen",  # "queen" or "rook"
    weight: str = "binary",  # "binary" or "shared_border_km"
    include_self: bool = False,
    fillna: float = 0.0,
):
    """
    Build an adjacency matrix for GADM admin regions.

    Parameters
    ----------
    df : pandas.DataFrame | None
        Optional. If provided, the function will infer the country from GID_1
        and restrict the matrix to regions present in df.
    country : str | None
        ISO3 country code, required if df is None.
    admin_level : int | str | None
        1 or 2 (or 'GID_1'/'GID_2').
    reion_filter : list|str|callable | None
        Optional filter to select a subset of regions:
    contiguity : str
        "queen" -> share a border or a vertex
        "rook"   -> share a border segment only
    weight : str
        "binary" -> 1 if adjacent, else 0
        "shared_border_km" -> length of shared border in kilometers
    include_self : bool
        If True, diagonal entries are set to 1 (binary) or 0 (weighted).
    fillna : float
        Value used to fill missing entries in the matrix.

    Returns
    -------
    adj : pandas.DataFrame
        Square adjacency matrix indexed and columned by admin codes.
    gdf : geopandas.GeoDataFrame
        The filtered region geometries used to build the matrix.
    """

    admin_level = _admin_level_col(admin_level)
    if admin_level == "GID_1":
        layer = "ADM_ADM_1"
    elif admin_level == "GID_2":
        layer = "ADM_ADM_2"
    else:
        raise ValueError(f"Unsupported admin_level: {admin_level}")

    if contiguity not in {"queen", "rook"}:
        raise ValueError("contiguity must be 'queen' or 'rook'")

    if weight not in {"binary", "shared_border_km"}:
        raise ValueError("weight must be 'binary' or 'shared_border_km'")

    # Infer country / filter regions
    if df is not None:
        if "GID_1" not in df.columns:
            raise ValueError("df must contain GID_1 to infer country.")
        countries = set(map(lambda x: x.split(".")[0], df["GID_1"].unique()))
        if len(countries) > 1:
            raise ValueError(f"DataFrame contains multiple countries: {countries}")
        inferred_country = countries.pop()
        if country is None:
            country = inferred_country

        if admin_level not in df.columns:
            raise ValueError(f"df must contain {admin_level} column.")
        keep_ids = pd.Index(df[admin_level].dropna().unique())
    else:
        if country is None:
            raise ValueError("country must be provided when df is None.")
        keep_ids = None

    geo_filename = Path(cache_folder) / "geo" / country / f"gadm41_{country}.gpkg"
    gdf = gpd.read_file(geo_filename, layer=layer)

    gdf = subset_regions(
        gdf,
        admin_level,
        region_filter,
    )

    if keep_ids is not None:
        gdf = gdf[gdf[admin_level].isin(keep_ids)].copy()

    gdf = gdf[[admin_level, "geometry"]].dropna(subset=["geometry"]).copy()

    if gdf.empty:
        raise ValueError("No geometries available after filtering.")

    # Repair obvious invalid geometries if needed
    gdf["geometry"] = gdf.geometry.apply(
        lambda geom: geom.buffer(0) if not geom.is_valid else geom
    )

    # Use a metric CRS for shared-border lengths
    gdf_m = gdf.to_crs(epsg=3857).reset_index(drop=True)

    labels = gdf_m[admin_level].tolist()

    # Initialize matrix
    adj = pd.DataFrame(fillna, index=labels, columns=labels, dtype=float)

    # Spatial index candidate search
    sindex = gdf_m.sindex
    tol = 1e-9  # meters, for numerical noise in rook test

    for i, geom_i in enumerate(gdf_m.geometry):
        # query possible neighbors
        candidates = list(sindex.query(geom_i, predicate="intersects"))
        for j in candidates:
            if j <= i:
                continue

            geom_j = gdf_m.geometry.iloc[j]

            # Exclude self
            if geom_i.equals(geom_j):
                continue

            is_adjacent = False
            shared_len_km = 0.0

            if contiguity == "queen":
                is_adjacent = geom_i.touches(geom_j)
            else:
                # rook: shared boundary segment only
                border = geom_i.boundary.intersection(geom_j.boundary)
                shared_len_km = border.length / 1000.0
                is_adjacent = shared_len_km > tol

            if is_adjacent:
                if weight == "binary":
                    val = 1.0
                else:
                    # Length of shared border in km (works for rook and queen;
                    # for queen, point-only touches contribute 0)
                    if contiguity == "queen":
                        border = geom_i.boundary.intersection(geom_j.boundary)
                        shared_len_km = border.length / 1000.0
                    val = shared_len_km

                adj.iat[i, j] = val
                adj.iat[j, i] = val

    if include_self:
        if weight == "binary":
            np.fill_diagonal(adj.values, 1.0)
        else:
            np.fill_diagonal(adj.values, 0.0)

    # Clean up formatting
    if weight == "binary":
        adj = adj.astype(int)

    return adj, gdf_m
