import numpy as np
import pandas as pd
import pytest
from thucia.core.geo import interpolate_covariates
from thucia.core.geo import merge_geo_sources
from thucia.core.geo import refresh_plugins
from thucia.core.geo.plugin_base import source_registry
from thucia.core.geo.plugin_loader import load_plugins
from thucia.core.registry import PluginNotFoundError
from thucia.core.registry import Registry


def test_real_plugins_self_register():
    refresh_plugins()
    names = source_registry.names()
    assert {"worldclim", "edo", "noaa", "worldpop"} <= set(names)
    for name in names:
        cls = source_registry.get(name)
        assert issubclass(cls, object)


def test_merge_geo_sources_unknown_origin_raises():
    with pytest.raises(PluginNotFoundError):
        merge_geo_sources(pd.DataFrame(), ["nope.metric"])


def test_merge_geo_sources_malformed_source_raises():
    with pytest.raises(ValueError, match="origin.field"):
        merge_geo_sources(pd.DataFrame(), ["no-dot-here"])


def test_load_plugins_isolates_broken_module(tmp_path, monkeypatch):
    # A module that raises on import must not stop the other plugins loading.
    pkg = tmp_path / "sources"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "broken.py").write_text("raise RuntimeError('boom')")
    (pkg / "fine.py").write_text(
        "from thucia.core.geo.plugin_base import SourceBase, source_registry\n"
        "@source_registry.register()\n"
        "class Fine(SourceBase):\n"
        "    ref = 'fine'\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        plugins = load_plugins(plugin_dir=pkg, module_stem="sources")
    finally:
        source_registry.unregister("fine")
    assert "broken" not in plugins
    assert "fine" in plugins


def _weekly_grid(n=14):
    return pd.DataFrame(
        {
            "Date": pd.period_range("2020-01-04", periods=n, freq="W-SAT"),
            "GID_2": ["A"] * n,
        }
    )


class MonthlyPlugin:
    ref = "fake"
    granularity = "M"
    name = "fake monthly"

    def merge(
        self,
        df,
        metrics=None,
        measures=None,
        use_cache=False,
        *,
        geo_col="GID_2",
        iso3=None,
        polygons=None,
    ):
        # One value per (geo_col, month), placed on the last week of each month.
        out = df.copy()
        end = pd.PeriodIndex(out["Date"]).to_timestamp(how="end")
        month = end.to_period("M")
        last_of_month = (
            out.groupby([geo_col, month], observed=False)["Date"].transform("max")
            == out["Date"]
        )
        out["tmax"] = np.nan
        out.loc[last_of_month, "tmax"] = np.arange(last_of_month.sum()) % 10 + 20.0
        return out


class FullPlugin(MonthlyPlugin):
    ref = "full"
    granularity = "M"

    def merge(
        self,
        df,
        metrics=None,
        measures=None,
        use_cache=False,
        *,
        geo_col="GID_2",
        iso3=None,
        polygons=None,
    ):
        out = df.copy()
        out["tmax"] = 25.0  # gap-free
        return out


@pytest.fixture
def fake_registry():
    reg = Registry("covariate source")
    reg.register()(MonthlyPlugin)
    reg.register()(FullPlugin)
    return reg


def test_interpolate_covariates_linear():
    df = _weekly_grid(10)
    df["tmax"] = np.nan
    df.loc[[0, 5, 9], "tmax"] = [0.0, 10.0, 20.0]
    out, n_filled = interpolate_covariates(df, ["tmax"])
    assert n_filled == 7
    assert out["tmax"].notna().all()
    # intermediate weeks are linearly interpolated between the known points
    assert out.loc[2, "tmax"] == pytest.approx(4.0)
    assert out.loc[7, "tmax"] == pytest.approx(15.0)


def test_interpolate_covariates_ffill_and_bfill():
    df = _weekly_grid(10)
    df["tmax"] = np.nan
    df.loc[[2, 7], "tmax"] = [10.0, 20.0]
    ff, _ = interpolate_covariates(df.copy(), ["tmax"], method="ffill")
    # forward-fill: carries the last known value forward; edges get nearest
    assert ff.loc[4, "tmax"] == 10.0
    assert ff.loc[0, "tmax"] == 10.0  # bfill edge
    bb, _ = interpolate_covariates(df.copy(), ["tmax"], method="bfill")
    assert bb.loc[4, "tmax"] == 20.0
    assert bb.loc[9, "tmax"] == 20.0  # ffill edge


def test_merge_geo_sources_interpolates_monthly_to_weekly(fake_registry, monkeypatch):
    import thucia.core.geo as geo

    monkeypatch.setattr(geo, "source_registry", fake_registry)
    with pytest.warns(UserWarning, match="interpolated"):
        out = merge_geo_sources(_weekly_grid(), ["fake.tmax"])
    assert out["tmax"].notna().all()
    assert len(out) == 14


def test_merge_geo_sources_monthly_case_no_warning(fake_registry, monkeypatch):
    import warnings

    import thucia.core.geo as geo

    monkeypatch.setattr(geo, "source_registry", fake_registry)
    monthly = pd.DataFrame(
        {"Date": pd.period_range("2020-01", periods=3, freq="M"), "GID_2": ["A"] * 3}
    )
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        out = merge_geo_sources(monthly, ["fake.tmax"])
    assert not [w for w in record if issubclass(w.category, UserWarning)]
    assert out["tmax"].notna().all()


def test_merge_geo_sources_gap_free_no_warning(fake_registry, monkeypatch):
    import warnings

    import thucia.core.geo as geo

    monkeypatch.setattr(geo, "source_registry", fake_registry)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        out = merge_geo_sources(_weekly_grid(), ["full.tmax"])
    assert not [w for w in record if issubclass(w.category, UserWarning)]
    assert (out["tmax"] == 25.0).all()


def test_merge_geo_sources_non_gadm_geo_col(fake_registry, monkeypatch):
    import thucia.core.geo as geo

    monkeypatch.setattr(geo, "source_registry", fake_registry)
    grid = pd.DataFrame(
        {
            "Date": pd.period_range("2020-01-04", periods=14, freq="W-SAT"),
            "region": ["north", "south"] * 7,
        }
    )
    # The source is M-granular on a W-SAT grid: interpolated onto every week.
    with pytest.warns(UserWarning, match="interpolated"):
        out = merge_geo_sources(grid, ["fake.tmax"], geo_col="region")
    assert set(out.columns) == {"Date", "region", "tmax"}
    assert out["tmax"].notna().all()
    assert "GID_2" not in out.columns
    assert "GID_1" not in out.columns
