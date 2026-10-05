from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from thucia.core.cases import ensure_complete_grid
from thucia.core.geo import attach_geo_attributes
from thucia.core.panels import panel_to_frame
from thucia.core.panels import read_rds
from thucia.core.pipeline import PipelineConfig
from thucia.core.pipeline import prepare_model_inputs

DATA = Path(__file__).resolve().parent / "fixtures"


def test_read_rds_returns_dataframe():
    # Tiny data.frame fixture shipped by the rdata test suite (MIT).
    out = read_rds(DATA / "panel.rds")
    assert isinstance(out, pd.DataFrame)
    assert out.shape == (3, 2)
    assert set(out.columns) == {"class", "value"}
    assert out["value"].tolist() == [1, 2, 3]


def test_panel_to_frame_from_date_col():
    df = pd.DataFrame(
        {
            "Date": pd.PeriodIndex(["2020-01", "2020-02"], freq="M"),
            "g": ["A", "B"],
            "cases": [1.0, np.nan],
            "temp": [10.0, 11.0],
            "pop": [100.0, 200.0],
        }
    )
    out = panel_to_frame(
        df,
        cases_col="cases",
        geo_col="g",
        covariate_cols=["temp"],
        pop_col="pop",
    )
    assert list(out.columns) == ["Date", "g", "Cases", "future", "temp", "pop_count"]
    assert out["Cases"].isna().iloc[1]  # NA outcome preserved
    assert (out["future"] == False).all()  # noqa: E712
    assert out["pop_count"].tolist() == [100.0, 200.0]


def test_panel_to_frame_from_index_col_and_anchor():
    df = pd.DataFrame({"week": [1, 2, 3], "g": ["A", "A", "A"], "cases": [1, 2, 3]})
    out = panel_to_frame(
        df,
        cases_col="cases",
        geo_col="g",
        geo_parent=None,
        index_col="week",
        anchor="2020-01-01",
        freq="W-SAT",
    )
    assert str(out["Date"].dtype) == "period[W-SAT]"
    assert out["Date"].is_monotonic_increasing
    assert len(out) == 3


def test_panel_to_frame_requires_single_time_axis():
    df = pd.DataFrame({"g": ["A"], "cases": [1]})
    with pytest.raises(ValueError, match="Pass date_col="):
        panel_to_frame(df, cases_col="cases", geo_col="g")
    with pytest.raises(ValueError, match="only one"):
        panel_to_frame(
            df,
            cases_col="cases",
            geo_col="g",
            date_col="d",
            index_col="week",
        )


def test_panel_to_frame_duplicate_key_raises():
    df = pd.DataFrame(
        {
            "Date": pd.PeriodIndex(["2020-01", "2020-01"], freq="M"),
            "g": ["A", "A"],
            "cases": [1, 2],
        }
    )
    with pytest.raises(ValueError, match="not unique"):
        panel_to_frame(df, cases_col="cases", geo_col="g")


def test_panel_to_frame_integer_index_validation():
    df = pd.DataFrame({"week": [0, 1], "g": ["A", "A"], "cases": [1, 2]})
    with pytest.raises(ValueError, match="positive integers"):
        panel_to_frame(
            df,
            cases_col="cases",
            geo_col="g",
            index_col="week",
            anchor="2020-01-01",
        )


def test_panel_recipe_feeds_prepare_model_inputs(tmp_path):
    # Panel (10 rows) -> frame -> attach parents -> complete grid -> prepare.
    df = pd.DataFrame(
        {
            "district_id": ["A"] * 6 + ["B"] * 5,
            "week_index": [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5],
            "weekly_cases": [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5],
            "temp": [20.0] * 6 + [21.0] * 5,
            "population_total": [1000.0] * 6 + [2000.0] * 5,
        }
    )
    frame = panel_to_frame(
        df,
        cases_col="weekly_cases",
        geo_col="district_id",
        index_col="week_index",
        anchor="2020-01-01",
        freq="W-SAT",
        covariate_cols=["temp"],
        pop_col="population_total",
    )
    regions = pd.DataFrame(
        {
            "analysis_district_id": ["A", "B"],
            "province_ubigeo": ["P1", "P1"],
        }
    )
    frame = attach_geo_attributes(
        frame,
        regions,
        geo_col="district_id",
        region_col="analysis_district_id",
        geo_parent="province_ubigeo",
    )
    frame = ensure_complete_grid(frame, geo_col="district_id", case_col="Cases")
    assert set(frame["district_id"]) == {"A"}  # B lacks week 6

    config = PipelineConfig(
        path=tmp_path,
        source_specs=[],
        geo_col="district_id",
        geo_parent="province_ubigeo",
        horizons=[1],
        season_length=52,
    )
    prepared, covariates = prepare_model_inputs(frame, config)
    assert "Log_Cases" in prepared.columns
    assert "temp" in covariates
    assert prepared["Log_Cases"].notna().all()
