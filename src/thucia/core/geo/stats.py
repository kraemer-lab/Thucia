from pathlib import Path

import geopandas as gpd
from rasterstats import zonal_stats
from thucia.core.fs import cache_folder


def raster_stats_gid2(
    tif_file,
    geo_codes: list[str],
    stats=["mean"],
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
    for measure in stats_data[0].keys():
        polygons[measure] = [stat[measure] for stat in stats_data]
    if key != geo_col:
        polygons = polygons.rename(columns={key: geo_col})
    # Drop geometry
    keep_columns = keep_columns + [geo_col, *stats]
    polygons = polygons[keep_columns]
    return polygons
