from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio.mask
from thucia.core.fs import cache_folder

_STAT_REDUCERS = {
    "min": np.nanmin,
    "max": np.nanmax,
    "mean": np.nanmean,
    "sum": np.nansum,
    "median": np.nanmedian,
    "stdev": np.nanstd,
    "count": lambda values: np.count_nonzero(~np.isnan(values)),
}


def _feature_stats(geom, src, stat_names):
    """Zonal statistics for a single geometry against an open raster."""
    out_image, _ = rasterio.mask.mask(src, [geom], crop=True, nodata=src.nodata)
    data = np.ma.filled(out_image[0], np.nan).astype("float64")
    if src.nodata is not None:
        data = np.where(data == src.nodata, np.nan, data)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return {name: None for name in stat_names}
    return {name: _STAT_REDUCERS[name](data).item() for name in stat_names}


def zonal_stats(polygons, tif_file, stats=None):
    """Compute per-geometry zonal statistics of a raster.

    A self-contained ``rasterio`` implementation (no rasterstats dependency):
    each geometry is masked with ``rasterio.mask`` and the requested reductions
    run over the finite, non-nodata pixels covered. Geometries in a different
    CRS are reprojected to the raster's CRS first.
    """
    stats = list(stats) if stats is not None else ["mean"]
    unknown = [s for s in stats if s not in _STAT_REDUCERS]
    if unknown:
        raise ValueError(f"Unsupported zonal statistic(s): {unknown}")
    with rasterio.open(tif_file) as src:
        polys = polygons
        if polygons.crs is not None and not polygons.crs.equals(src.crs):
            polys = polygons.to_crs(src.crs)
        return [_feature_stats(geom, src, stats) for geom in polys.geometry]


def raster_stats_gid2(
    tif_file,
    geo_codes: list[str],
    stats=None,
    *,
    geo_col: str = "GID_2",
    polygons=None,
    iso3: str | None = None,
):
    """
    Calculate zonal statistics for a given set of region polygons against a raster
    file.

    The region polygons are either supplied explicitly (``polygons``, a
    GeoDataFrame keyed by ``geo_col``) or, for GADM-shaped codes, loaded from the
    cached GADM GeoPackage for ``iso3`` (derived from the code prefix when not
    given).

    Returns:
    gpd.GeoDataFrame: The input GeoDataFrame with additional columns for the
    calculated statistics.
    """
    stats = stats or ["mean"]
    if polygons is None:
        if isinstance(iso3, str):
            pass
        elif iso3 is None:
            # Fall back to deriving the ISO3 from the geo codes (GADM-native).
            iso3 = set(map(lambda x: x.split(".")[0], geo_codes))
            if len(iso3) != 1:
                raise ValueError("All filters must be for the same ISO3 country code.")
            iso3 = iso3.pop()
        else:
            raise ValueError("Cannot derive ISO3 from a non-GADM code scheme.")

        file_path = Path(cache_folder) / "geo" / iso3 / f"gadm41_{iso3}.gpkg"
        if not file_path.exists():
            raise FileNotFoundError(
                f"GeoPackage file for {iso3} not found at {file_path}"
            )

        polygons = gpd.read_file(str(file_path), layer="ADM_ADM_2")
        keep_columns = [
            "GID_0",
            "COUNTRY",
            "GID_1",
            "NL_NAME_1",
            "NAME_1",
            "GID_2",
            "NL_NAME_2",
            "NAME_2",
            "TYPE_2",
            "ENGTYPE_2",
            "CC_2",
            "HASC_2",
        ]
        key = "GID_2"
    else:
        # Explicit polygon map: the caller's geo column is authoritative. Keep the
        # map's attribute columns (excluding geometry) so names etc. flow through.
        key = geo_col
        keep_columns = [c for c in polygons.columns if c not in ("geometry", geo_col)]

    polygons = polygons[polygons[key].isin(geo_codes)].copy()
    stats_data = zonal_stats(polygons, tif_file, stats=stats)
    if not stats_data:
        return polygons[keep_columns + [geo_col]].drop(columns=["geometry"])
    for measure in stats_data[0].keys():
        polygons[measure] = [stat[measure] for stat in stats_data]
    if key != geo_col:
        polygons = polygons.rename(columns={key: geo_col})
    # Drop geometry
    keep_columns = keep_columns + [geo_col, *stats]
    polygons = polygons[keep_columns]
    return polygons
