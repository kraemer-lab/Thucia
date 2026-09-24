from pathlib import Path

import geopandas as gpd
from rasterstats import zonal_stats
from thucia.core.fs import cache_folder


def raster_stats_gid2(
    tif_file, geo_codes: list[str], stats=["mean"], *, iso3: str | None = None
):
    """
    Calculate zonal statistics for a given GeoDataFrame of polygons against a raster
    file.

    Parameters:

    Returns:
    gpd.GeoDataFrame: The input GeoDataFrame with additional columns for the
    calculated statistics.
    """

    if iso3 is None:
        # Fall back to deriving the ISO3 from the geo codes (GADM-native).
        iso3 = set(map(lambda x: x.split(".")[0], geo_codes))
        if len(iso3) != 1:
            raise ValueError("All filters must be for the same ISO3 country code.")
        iso3 = iso3.pop()

    file_path = Path(cache_folder) / "geo" / iso3 / f"gadm41_{iso3}.gpkg"
    if not file_path.exists():
        raise FileNotFoundError(f"GeoPackage file for {iso3} not found at {file_path}")

    polygons = gpd.read_file(str(file_path), layer="ADM_ADM_2")
    polygons = polygons[polygons["GID_2"].isin(geo_codes)]
    stats_data = zonal_stats(polygons, tif_file, stats=stats)
    for measure in stats_data[0].keys():
        polygons[measure] = [stat[measure] for stat in stats_data]
    # Drop geometry
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
        *stats,
    ]
    polygons = polygons[keep_columns]
    return polygons
