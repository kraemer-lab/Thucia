# Probing tests for the EDO covariate source (geo/sources/edo.py).
# No network, no real rasters, and no real worker processes: downloads, zonal
# stats, and the ProcessPoolExecutor are all stubbed; the pool runs inline.
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import thucia.core.geo.sources.edo as edo_mod


@pytest.fixture
def edo(tmp_path, monkeypatch):
    monkeypatch.setattr(edo_mod, "cache_folder", str(tmp_path))
    climate = Path(tmp_path) / "climate"
    climate.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        edo_mod.EDO,
        "cache_file",
        climate / "edo_stats.sqlite",
    )
    return edo_mod.EDO()


def _zip_with(filename):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(filename, b"raster-bytes")
    return buf.getvalue()


def _fake_stat(gid_2s, mean_value=-0.5):
    """A raster_stats_gid2-shaped frame with the EDO 'mean'/'SPI6' statistic."""
    return pd.DataFrame({"GID_2": gid_2s, "mean": [mean_value] * len(gid_2s)})


class _SyncPool:
    """ProcessPoolExecutor stand-in that runs jobs inline (no process spawn)."""

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def submit(self, fn, arg):
        return _DoneFuture(fn(arg))


class _DoneFuture:
    def __init__(self, result):
        self._result = result

    def result(self):
        return self._result


@pytest.fixture
def sync_pool(monkeypatch):
    def fake_as_completed(futures, **kwargs):
        for future in futures:
            yield future

    monkeypatch.setattr(edo_mod, "ProcessPoolExecutor", _SyncPool)
    monkeypatch.setattr(edo_mod, "as_completed", fake_as_completed)


def _weekly_grid(dates):
    return pd.DataFrame(
        {
            "Date": np.repeat(pd.to_datetime(dates), 2),
            "GID_2": ["X.1.1_2", "X.1.2_2"] * len(dates),
            "Cases": 1.0,
        }
    )


def test_get_filename_downloads_and_extracts(edo, monkeypatch, tmp_path):
    tif_name = "spc06_m_gdo_20200101_m_300_z01.tif"

    class FakeResp:
        status_code = 200
        content = _zip_with(tif_name)

    monkeypatch.setattr(edo_mod.requests, "head", lambda url: FakeResp())
    calls = []
    monkeypatch.setattr(
        edo_mod.requests, "get", lambda url: calls.append(url) or FakeResp()
    )

    tif = edo.get_filename(2020, 1)
    assert Path(tif).exists()
    assert any("spc06" in c for c in calls)
    # No re-download when the tif is already present.
    monkeypatch.setattr(edo_mod.requests, "get", lambda url: pytest.fail("download"))
    assert edo.get_filename(2020, 1) == tif


def test_merge_dedups_raster_per_month(edo, monkeypatch, sync_pool):
    # Weekly dates within one month share the monthly raster: one pool job per
    # month over the month's union, not one job per week.
    calls = []

    def fake_stats(tif, gids, geo_col="GID_2", iso3=None, polygons=None):
        calls.append((tif, sorted(gids)))
        return _fake_stat(gids)

    monkeypatch.setattr(
        edo, "get_filename", lambda year, month: f"spc_{year}{month:02d}.tif"
    )
    monkeypatch.setattr(edo_mod, "raster_stats_gid2", fake_stats)

    grid = _weekly_grid(
        ["2020-01-04", "2020-01-11", "2020-01-18", "2020-02-01"]
    )
    out = edo.merge(grid)
    assert calls == [
        ("spc_202001.tif", ["X.1.1_2", "X.1.2_2"]),
        ("spc_202002.tif", ["X.1.1_2", "X.1.2_2"]),
    ]
    # Every row filled and constant within the month.
    assert out["SPI6"].notna().all()
    jan = out.loc[out["Date"] == pd.Timestamp("2020-01-04"), "SPI6"].tolist()
    jan_mid = out.loc[out["Date"] == pd.Timestamp("2020-01-11"), "SPI6"].tolist()
    assert jan == jan_mid == [-0.5, -0.5]


def test_merge_primes_month_cache(edo, monkeypatch, sync_pool):
    # A merge over weekly dates caches the month's full union for every date,
    # so a later run is served entirely from the cache.
    def fake_stats(tif, gids, geo_col="GID_2", iso3=None, polygons=None):
        return _fake_stat(gids)

    monkeypatch.setattr(
        edo, "get_filename", lambda year, month: f"spc_{year}{month:02d}.tif"
    )
    monkeypatch.setattr(edo_mod, "raster_stats_gid2", fake_stats)

    grid = _weekly_grid(["2020-01-04", "2020-01-11"])
    edo.merge(grid, use_cache=True)

    # Second run: fully served from the primed cache (no download / no pool).
    monkeypatch.setattr(edo, "get_filename", lambda *a, **k: pytest.fail("download"))
    monkeypatch.setattr(
        edo_mod, "raster_stats_gid2", lambda *a, **k: pytest.fail("stats")
    )
    out = edo.merge(grid, use_cache=True)
    assert out["SPI6"].tolist() == [-0.5, -0.5, -0.5, -0.5]