# Probing tests for raster zonal-statistics helpers (core/geo/stats.py).
# GADM GeoPackage and raster access are mocked (no network, no real raster).
import numpy as np
import pandas as pd
import pytest
import rasterio
import thucia.core.geo.stats as stats
from geopandas import GeoDataFrame
from rasterio.transform import from_origin
from shapely.geometry import box


@pytest.fixture
def polygons():
    return pd.DataFrame(
        {
            "GID_0": ["X"] * 2,
            "COUNTRY": ["X"] * 2,
            "GID_1": ["X.1_1"] * 2,
            "NL_NAME_1": [None, None],
            "NAME_1": ["A", "A"],
            "GID_2": ["X.1.1_2", "X.1.2_2"],
            "NL_NAME_2": [None, None],
            "NAME_2": ["One", "Two"],
            "TYPE_2": ["Type"] * 2,
            "ENGTYPE_2": ["Type"] * 2,
            "CC_2": [None, None],
            "HASC_2": ["XA", "XB"],
            "geometry": [None, None],
        }
    )


def test_raster_stats_gid2_returns_kept_columns_with_stats(
    polygons, monkeypatch, tmp_path
):
    gid_2s = ["X.1.1_2", "X.1.2_2"]
    monkeypatch.setattr(stats, "cache_folder", str(tmp_path))
    # create the expected gpkg file so the existence check passes
    gpkg = tmp_path / "geo" / "X" / "gadm41_X.gpkg"
    gpkg.parent.mkdir(parents=True)
    gpkg.touch()

    def fake_read_file(path, layer):
        assert layer == "ADM_ADM_2"
        return polygons

    def fake_zonal_stats(polys, tif, stats=None):
        assert tif == "some.tif"
        assert set(polys["GID_2"]) == set(gid_2s)
        assert stats == ["mean"]
        return [{"mean": 1.0}, {"mean": 2.0}]

    monkeypatch.setattr(stats.gpd, "read_file", fake_read_file)
    monkeypatch.setattr(stats, "zonal_stats", fake_zonal_stats)

    out = stats.raster_stats_gid2("some.tif", gid_2s, stats=["mean"])

    assert "mean" in out.columns
    assert out["mean"].tolist() == [1.0, 2.0]
    # geometry and internal columns are dropped; only keep_columns + stats remain
    assert "geometry" not in out.columns
    assert set(out.columns) == {
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
        "mean",
    }


def test_raster_stats_gid2_multi_stats(polygons, monkeypatch, tmp_path):
    monkeypatch.setattr(stats, "cache_folder", str(tmp_path))
    gpkg = tmp_path / "geo" / "X" / "gadm41_X.gpkg"
    gpkg.parent.mkdir(parents=True)
    gpkg.touch()
    monkeypatch.setattr(stats.gpd, "read_file", lambda *a, **k: polygons)

    def fake_zonal_stats(polys, tif, stats=None):
        assert stats == ["mean", "count"]
        return [{"mean": 1.0, "count": 3}, {"mean": 2.0, "count": 5}]

    monkeypatch.setattr(stats, "zonal_stats", fake_zonal_stats)

    out = stats.raster_stats_gid2(
        "some.tif", ["X.1.1_2", "X.1.2_2"], stats=["mean", "count"]
    )

    assert out["mean"].tolist() == [1.0, 2.0]
    assert out["count"].tolist() == [3, 5]


def test_raster_stats_gid2_mixed_iso3_raises(polygons, monkeypatch, tmp_path):
    monkeypatch.setattr(stats, "cache_folder", str(tmp_path))
    with pytest.raises(ValueError, match="same ISO3"):
        stats.raster_stats_gid2("some.tif", ["X.1.1_2", "Y.1.1_2"])


def test_raster_stats_gid2_missing_gpkg_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(stats, "cache_folder", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="not found"):
        stats.raster_stats_gid2("some.tif", ["X.1.1_2"])


def test_raster_stats_gid2_explicit_polygons(monkeypatch):
    # A caller-supplied polygon map (no GADM involvement): non-GADM codes, list
    # of codes keyed by the map's own geo column, no iso3 required.
    regions = pd.DataFrame(
        {
            "region": ["north", "south"],
            "ADM1": ["ProvA", "ProvB"],
            "geometry": [None, None],
        }
    )

    def fake_zonal_stats(polys, tif, stats=None):
        assert set(polys["region"]) == {"north", "south"}
        assert tif == "some.tif"
        assert stats == ["mean"]
        return [{"mean": 1.0}, {"mean": 2.0}]

    monkeypatch.setattr(stats, "zonal_stats", fake_zonal_stats)
    monkeypatch.setattr(
        stats.gpd, "read_file", lambda *a, **k: pytest.fail("no GADM read")
    )

    out = stats.raster_stats_gid2(
        "some.tif",
        ["north", "south"],
        geo_col="region",
        polygons=regions,
    )

    assert out["mean"].tolist() == [1.0, 2.0]
    # The map's own attribute columns flow through; no GADM columns appear.
    assert set(out.columns) == {"region", "ADM1", "mean"}
    assert "geometry" not in out.columns


def test_stats_region_only_intersection(monkeypatch):
    # Explicit polygons are filtered down to the requested codes before zonal
    # stats, so a map can cover more regions than the merge needs.
    regions = pd.DataFrame(
        {
            "region": ["north", "south", "west"],
            "geometry": [None] * 3,
        }
    )

    def fake_zonal_stats(polys, tif, stats=None):
        assert set(polys["region"]) == {"north"}
        return [{"mean": 5.0}]

    monkeypatch.setattr(stats, "zonal_stats", fake_zonal_stats)
    out = stats.raster_stats_gid2(
        "some.tif", ["north"], geo_col="region", polygons=regions
    )
    assert out["mean"].tolist() == [5.0]
    assert out["region"].tolist() == ["north"]


def _write_ones_raster(tmp_path, nodata=-9999.0, sz=6):
    """A 6x6 unit-CRS raster: ones except the top-left cell set to nodata."""
    path = tmp_path / "ones.tif"
    array = np.ones((sz, sz), dtype="float32")
    array[0, 0] = nodata
    profile = {
        "driver": "GTiff",
        "height": sz,
        "width": sz,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": from_origin(0.0, 6.0, 1.0, 1.0),
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array, 1)
    return path


def test_zonal_stats_rasterio(tmp_path):
    # Unweighted reductions over finite, non-nodata pixels: a 2x2 all-ones
    # block -> mean 1, sum 4, count 4.
    tif = _write_ones_raster(tmp_path)
    row_2 = box(1, 4, 3, 6)  # cols 1-2, rows 0-1 (lat 4-6)
    row_3 = box(3, 4, 5, 6)  # cols 3-4, rows 0-1
    polys = GeoDataFrame(
        {"region": ["a", "b"], "geometry": [row_2, row_3]}, crs="EPSG:4326"
    )
    out = stats.zonal_stats(polys, tif, stats=["mean", "sum", "count"])
    for stat in out:
        assert stat["mean"] == pytest.approx(1.0)
        assert stat["sum"] == pytest.approx(4.0)
        assert stat["count"] == 4


def test_zonal_stats_rasterio_excludes_nodata(tmp_path):
    # The polygon includes the nodata cell: only the finite pixel is counted.
    tif = _write_ones_raster(tmp_path)
    covering_nodata = box(0, 5, 2, 6)  # cols 0-1, row 0; cell (0,0) is nodata
    polys = GeoDataFrame(
        {"region": ["a"], "geometry": [covering_nodata]}, crs="EPSG:4326"
    )
    out = stats.zonal_stats(polys, tif, stats=["mean", "sum", "count"])
    assert out[0]["mean"] == pytest.approx(1.0)
    assert out[0]["sum"] == pytest.approx(1.0)
    assert out[0]["count"] == 1


def test_zonal_stats_unsupported_stat_raises(tmp_path):
    tif = _write_ones_raster(tmp_path)
    assert tmp_path.exists()
    polys = GeoDataFrame(
        {"region": ["a"], "geometry": [box(1, 5, 2, 6)]}, crs="EPSG:4326"
    )
    with pytest.raises(ValueError, match="Unsupported zonal"):
        stats.zonal_stats(polys, tif, stats=["nope"])
