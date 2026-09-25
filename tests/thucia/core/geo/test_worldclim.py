# Probing tests for the WorldClim covariate source (geo/sources/worldclim.py).
# No network and no real raster: downloads, zonal stats, and cache all run
# against mocked/fake artifacts in a tmp cache folder.
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import thucia.core.geo.sources.worldclim as worldclim


@pytest.fixture
def wc(tmp_path, monkeypatch):
    monkeypatch.setattr(worldclim, "cache_folder", str(tmp_path))
    climate = Path(tmp_path) / "climate"
    climate.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        worldclim.WorldClim,
        "cache_file",
        climate / "worldclim_stats.sqlite",
    )
    return worldclim.WorldClim()


def _zip_with(filename):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(filename, b"raster-bytes")
    return buf.getvalue()


def _fake_stat(gid_2s, mean_value=25.0):
    """A raster_stats_gid2-shaped frame with all the WorldClim cache columns."""
    return pd.DataFrame(
        {
            "GID_2": gid_2s,
            "GID_0": ["X"] * len(gid_2s),
            "COUNTRY": ["X"] * len(gid_2s),
            "GID_1": ["X.1_1"] * len(gid_2s),
            "NAME_1": ["State"] * len(gid_2s),
            "NL_NAME_1": [None] * len(gid_2s),
            "NAME_2": [f"R{i}" for i in range(len(gid_2s))],
            "NL_NAME_2": [None] * len(gid_2s),
            "TYPE_2": ["T"] * len(gid_2s),
            "ENGTYPE_2": ["T"] * len(gid_2s),
            "CC_2": [None] * len(gid_2s),
            "HASC_2": [f"X{i}" for i in range(len(gid_2s))],
            "mean": [mean_value] * len(gid_2s),
        }
    )


class _FakeSrc:
    def __init__(self, band_values, profile):
        self._band = band_values
        self.profile = profile

    def read(self, idx):
        return self._band

    def write(self, band, idx):
        return len(band)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def fake_rasterio_open(create_on_write=True):
    """Stand-in for rasterio.open: read yields a fake src, write creates the file."""

    def open_impl(path, mode="r", **profile):
        if "w" in mode:
            if create_on_write:
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                Path(path).touch()
            return _FakeSrc(np.zeros((2, 2), dtype="uint8"), profile)
        return _FakeSrc(np.array([[1, 2], [3, 4]], dtype="uint8"), {"count": 12})

    return open_impl


# --- _get_filename_cru ---


def test_get_filename_cru_downloads_and_extracts(wc, monkeypatch, tmp_path):
    tif_name = "wc2.1_cruts4.09_2.5m_tmin_2020-01.tif"

    class FakeResp:
        status_code = 200
        content = _zip_with(tif_name)

    calls = []
    monkeypatch.setattr(
        worldclim.requests, "get", lambda url: calls.append(url) or FakeResp()
    )

    tif = wc._get_filename_cru("tmin", 2020, 1)
    assert tif.name == tif_name
    assert tif.exists()
    assert len(calls) == 1
    assert "2020-2024" in calls[0]  # decade range


def test_get_filename_cru_uses_cached_file(wc, monkeypatch):
    tif = wc.cache_file.parent / "wc2.1_cruts4.09_2.5m_tmin_2020-01.tif"
    tif.parent.mkdir(parents=True, exist_ok=True)
    tif.touch()

    monkeypatch.setattr(
        worldclim.requests, "get", lambda url: pytest.fail("no download")
    )
    assert wc._get_filename_cru("tmin", 2020, 1) == tif


def test_get_filename_cru_year_beyond_max_raises(wc, monkeypatch):
    monkeypatch.setattr(
        worldclim.requests, "get", lambda url: pytest.fail("no download")
    )
    with pytest.raises(ValueError, match="beyond the maximum"):
        wc._get_filename_cru("tmin", 2025, 1)


def test_get_filename_cru_http_error_raises(wc, monkeypatch):
    class FakeResp:
        status_code = 404
        content = b""

    monkeypatch.setattr(worldclim.requests, "get", lambda url: FakeResp())
    with pytest.raises(FileNotFoundError, match="Failed to download"):
        wc._get_filename_cru("tmin", 2020, 1)


def test_get_filename_cru_missing_tif_after_extract_raises(wc, monkeypatch):
    class FakeResp:
        status_code = 200
        content = _zip_with("something_else.tif")

    monkeypatch.setattr(worldclim.requests, "get", lambda url: FakeResp())
    with pytest.raises(FileNotFoundError, match="not found after extraction"):
        wc._get_filename_cru("tmin", 2020, 1)


# --- _get_filename_forecast ---


def test_get_filename_forecast_downloads_and_splits_band(wc, monkeypatch, tmp_path):
    year_file = wc.cache_file.parent / "wc2.1_2.5m_tmin_ACCESS-CM2_ssp245_2031.tif"
    month_file = wc.cache_file.parent / "wc2.1_2.5m_tmin_ACCESS-CM2_ssp245_2031-03.tif"

    class FakeResp:
        status_code = 200
        content = b"raster-bytes"

    monkeypatch.setattr(worldclim.requests, "get", lambda url: FakeResp())
    monkeypatch.setattr(worldclim.rasterio, "open", fake_rasterio_open())

    tif = wc._get_filename_forecast("tmin", 2031, 3)
    assert tif == month_file
    assert year_file.exists()
    assert month_file.exists()


def test_get_filename_forecast_year_download_failure(wc, monkeypatch):
    class FakeResp:
        status_code = 500
        content = b""

    monkeypatch.setattr(worldclim.requests, "get", lambda url: FakeResp())
    with pytest.raises(FileNotFoundError, match="Failed to download"):
        wc._get_filename_forecast("tmin", 2031, 3)


def test_get_filename_forecast_month_missing_after_split_raises(wc, monkeypatch):
    class FakeResp:
        status_code = 200
        content = b"raster-bytes"

    monkeypatch.setattr(worldclim.requests, "get", lambda url: FakeResp())
    # The band-split write does NOT create the month file -> the guard fires.
    monkeypatch.setattr(
        worldclim.rasterio, "open", fake_rasterio_open(create_on_write=False)
    )
    with pytest.raises(FileNotFoundError, match="not found after extraction"):
        wc._get_filename_forecast("tmin", 2031, 3)


# --- get_filename (CRU -> forecast fallback) ---


def test_get_filename_cru_success(wc, monkeypatch):
    monkeypatch.setattr(wc, "_get_filename_cru", lambda m, y, mo: "cru.tif")
    path, source = wc.get_filename("tmin", 2020, 1)
    assert path == "cru.tif" and source == "CRU-TS"


def test_get_filename_falls_back_to_forecast(wc, monkeypatch):
    def fail(*a, **k):
        raise FileNotFoundError("nope")

    monkeypatch.setattr(wc, "_get_filename_cru", fail)
    monkeypatch.setattr(wc, "_get_filename_forecast", lambda *a, **k: "forecast.tif")
    path, source = wc.get_filename("tmin", 2025, 1)
    assert path == "forecast.tif" and source == "forecast"


def test_get_filename_total_failure_raises(wc, monkeypatch):
    def fail(*a, **k):
        raise FileNotFoundError("nope")

    monkeypatch.setattr(wc, "_get_filename_cru", fail)
    monkeypatch.setattr(wc, "_get_filename_forecast", fail)
    with pytest.raises(FileNotFoundError, match="not found"):
        wc.get_filename("tmin", 2025, 1)


# --- merge ---


@pytest.fixture
def case_df():
    dates = pd.to_datetime(["2020-01-31", "2020-02-29"])
    return pd.DataFrame(
        {
            "Date": np.repeat(dates, 2),
            "GID_2": ["X.1.1_2", "X.1.2_2"] * 2,
            "Cases": [5.0, 7.0, 6.0, 8.0],
        }
    )


def test_merge_expands_default_metrics(wc, case_df, monkeypatch):
    monkeypatch.setattr(
        wc, "get_filename", lambda metric, year, month: ("fake.tif", "CRU-TS")
    )
    calls = []

    def fake_stats(tif, gid_2s, geo_col="GID_2", iso3=None, polygons=None):
        calls.append(gid_2s)
        return _fake_stat(gid_2s, mean_value=25.0)

    monkeypatch.setattr(worldclim, "raster_stats_gid2", fake_stats)

    out = wc.merge(case_df, metrics=["*"])
    # default metrics tmin/tmax/prec each merged (mean renamed to bare metric)
    for col in ["tmin", "tmax", "prec"]:
        assert col in out.columns
        assert out[col].notna().all()
    # incidence-relevant columns untouched
    assert "Cases" in out.columns
    # zonal stats computed for each GID_2 subset per date
    assert all(len(g) == 2 for g in calls)


def test_merge_specific_metric_and_measures(wc, case_df, monkeypatch):
    monkeypatch.setattr(
        wc, "get_filename", lambda metric, year, month: ("fake.tif", "CRU-TS")
    )
    monkeypatch.setattr(
        worldclim,
        "raster_stats_gid2",
        lambda tif, gids, geo_col="GID_2", iso3=None, polygons=None: _fake_stat(
            gids, mean_value=10.0
        ),
    )
    out = wc.merge(case_df, metrics=["tmin"], measures=["mean"])
    assert "tmin" in out.columns
    assert "tmax" not in out.columns
    assert out["tmin"].notna().all()


def test_merge_use_cache_serves_cached_records(wc, case_df, monkeypatch):
    # Pre-populate the cache for all (GID_2, Date) pairs.
    for date in pd.to_datetime(["2020-01-31", "2020-02-29"]):
        wc._add_cache_records(
            "tmin",
            _fake_stat(["X.1.1_2", "X.1.2_2"], mean_value=7.5).assign(
                Date=date, source="CRU-TS"
            ),
            geo_col="GID_2",
        )

    # If the cache is hit, neither downloads nor zonal stats should run.
    monkeypatch.setattr(wc, "get_filename", lambda *a, **k: pytest.fail("download"))
    monkeypatch.setattr(
        worldclim, "raster_stats_gid2", lambda *a, **k: pytest.fail("stats")
    )

    out = wc.merge(case_df, metrics=["tmin"], use_cache=True)
    assert "tmin" in out.columns
    assert out["tmin"].tolist() == [7.5, 7.5, 7.5, 7.5]


def test_merge_use_cache_serves_cached_records_period_dates(wc, monkeypatch, case_df):
    # Pipeline frames carry Period dates; the cache-read path must normalise
    # them to timestamps (regression: pd.to_datetime on PeriodDtype raised).
    case_df_period = case_df.assign(Date=case_df["Date"].dt.to_period("M"))
    for date in pd.to_datetime(["2020-01-31", "2020-02-29"]):
        wc._add_cache_records(
            "tmin",
            _fake_stat(["X.1.1_2", "X.1.2_2"], mean_value=7.5).assign(
                Date=date, source="CRU-TS"
            ),
            geo_col="GID_2",
        )

    monkeypatch.setattr(wc, "get_filename", lambda *a, **k: pytest.fail("download"))
    monkeypatch.setattr(
        worldclim, "raster_stats_gid2", lambda *a, **k: pytest.fail("stats")
    )

    out = wc.merge(case_df_period, metrics=["tmin"], use_cache=True)
    assert out["tmin"].tolist() == [7.5, 7.5, 7.5, 7.5]


def test_merge_skips_missing_raster_dates(wc, case_df, monkeypatch):
    def get_filename(metric, year, month):
        if month == 2:
            raise FileNotFoundError("missing")
        return ("fake.tif", "CRU-TS")

    monkeypatch.setattr(wc, "get_filename", get_filename)
    monkeypatch.setattr(
        worldclim,
        "raster_stats_gid2",
        lambda tif, gids, geo_col="GID_2", iso3=None, polygons=None: _fake_stat(
            gids, mean_value=3.0
        ),
    )
    out = wc.merge(case_df, metrics=["tmin"])
    # Only the January raster exists -> only those rows are filled.
    assert out["tmin"].notna().sum() == 2
    assert out.loc[out["Date"] == pd.Timestamp("2020-02-29"), "tmin"].isna().all()


def test_merge_generic_geo_col(wc, case_df, monkeypatch):
    # A non-GADM geo column: the same merge, columns keyed by `region`, with a
    # caller-supplied polygon map threaded into raster_stats_gid2.
    calls = {}

    grid = case_df.rename(columns={"GID_2": "region"})
    regions = pd.DataFrame({"region": ["X.1.1_2", "X.1.2_2"], "geometry": [None, None]})

    def fake_stats(tif, gids, geo_col="GID_2", iso3=None, polygons=None):
        calls["geo_col"] = geo_col
        calls["iso3"] = iso3
        calls["polygons"] = polygons
        stat = _fake_stat(gids, mean_value=4.0)
        return stat.rename(columns={"GID_2": geo_col})

    monkeypatch.setattr(
        wc, "get_filename", lambda metric, year, month: ("fake.tif", "CRU-TS")
    )
    monkeypatch.setattr(worldclim, "raster_stats_gid2", fake_stats)

    out = wc.merge(grid, metrics=["tmin"], geo_col="region", iso3="X", polygons=regions)
    assert "tmin" in out.columns
    assert out["tmin"].notna().all()
    assert "GID_2" not in out.columns
    assert calls["geo_col"] == "region"
    assert calls["iso3"] == "X"
    assert calls["polygons"] is regions


def test_merge_dedups_raster_per_month(wc, monkeypatch):
    # Weekly dates within one month share the monthly raster: zonal stats are
    # computed once per (metric, month), not once per date.
    dates = pd.to_datetime(["2020-01-04", "2020-01-11", "2020-01-18", "2020-02-01"])
    grid = pd.DataFrame(
        {
            "Date": np.repeat(dates, 2),
            "GID_2": ["X.1.1_2", "X.1.2_2"] * 4,
            "Cases": 1.0,
        }
    )
    calls = []

    def fake_stats(tif, gids, geo_col="GID_2", iso3=None, polygons=None):
        calls.append((tif, sorted(gids)))
        return _fake_stat(gids, mean_value=22.0)

    monkeypatch.setattr(
        wc, "get_filename", lambda metric, year, month: (f"wf{month}.tif", "CRU-TS")
    )
    monkeypatch.setattr(worldclim, "raster_stats_gid2", fake_stats)

    out = wc.merge(grid, metrics=["tmin"])
    # One extraction per month over the month's union, not one per week.
    assert calls == [
        ("wf1.tif", ["X.1.1_2", "X.1.2_2"]),
        ("wf2.tif", ["X.1.1_2", "X.1.2_2"]),
    ]
    # Every row filled and constant within the month.
    assert out["tmin"].notna().all()
    jan = out.loc[out["Date"] == pd.Timestamp("2020-01-04"), "tmin"].tolist()
    jan_mid = out.loc[out["Date"] == pd.Timestamp("2020-01-11"), "tmin"].tolist()
    assert jan == jan_mid == [22.0, 22.0]


def test_merge_primes_month_cache(wc, monkeypatch):
    # A merge over weekly dates caches the month's full union for every date,
    # so a later run is served entirely from the cache (no download/reraster).
    dates = pd.to_datetime(["2020-01-04", "2020-01-11"])
    grid = pd.DataFrame(
        {
            "Date": np.repeat(dates, 2),
            "GID_2": ["X.1.1_2", "X.1.2_2"] * 2,
            "Cases": 1.0,
        }
    )

    def fake_stats(tif, gids, geo_col="GID_2", iso3=None, polygons=None):
        return _fake_stat(gids, mean_value=5.0)

    monkeypatch.setattr(
        wc, "get_filename", lambda metric, year, month: ("fake.tif", "CRU-TS")
    )
    monkeypatch.setattr(worldclim, "raster_stats_gid2", fake_stats)

    wc.merge(grid, metrics=["tmin"], use_cache=True)

    # Second run: fully served from the primed cache.
    monkeypatch.setattr(wc, "get_filename", lambda *a, **k: pytest.fail("download"))
    monkeypatch.setattr(
        worldclim, "raster_stats_gid2", lambda *a, **k: pytest.fail("stats")
    )
    out = wc.merge(grid, metrics=["tmin"], use_cache=True)
    assert out["tmin"].tolist() == [5.0, 5.0, 5.0, 5.0]


def test__add_cache_records_pads_missing_gadm_columns(wc, tmp_path):
    # Non-GADM maps don't carry the GADM attribute columns; they must be stored
    # as NULL rather than crashing the cache insert.
    stat = pd.DataFrame(
        {"region": ["north"], "mean": [3.0], "Date": pd.to_datetime(["2020-01-31"])}
    )
    wc._add_cache_records("tmin", stat.copy(), geo_col="region")
    hits = wc._get_cache_records(
        "tmin",
        pd.Series(pd.to_datetime(["2020-01-31"])),
        ["north"],
        geo_col="region",
    )
    assert len(hits) == 1
    assert hits["mean"].tolist() == [3.0]
    assert hits["region"].tolist() == ["north"]
    assert hits["GID_0"].isna().all()
