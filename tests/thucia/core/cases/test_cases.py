import pandas as pd
import pytest
from thucia.core.cases import aggregate_cases
from thucia.core.cases import align_date_types
from thucia.core.cases import cases_per_month
from thucia.core.cases import ensure_complete_grid


def test_cases_per_month_fills_zeros():
    # Sample data with missing province-date combos
    data = {
        "GID_1": ["A"] * 3,
        "GID_2": ["A_A", "A_A", "A_B"],
        "Date": ["2025-01-15", "2025-02-20", "2025-01-10"],
        "Status": ["active"] * 3,
    }
    df = pd.DataFrame(data)

    result = cases_per_month(df).df  # default: monthly
    expected_dates = pd.Series(
        [
            pd.Period("2025-01", freq="M"),
            pd.Period("2025-02", freq="M"),
        ]
    )
    expected_index = pd.MultiIndex.from_product(
        [["A_A", "A_B"], expected_dates], names=["GID_2", "Date"]
    )
    result = result.set_index(["GID_2", "Date"]).sort_index()

    # Assert that Dates are periods, with month end frequency
    assert result.index.dtypes["Date"] == "period[M]"

    # Assert all expected combinations exist
    for combo in expected_index:
        assert combo in result.index, f"Missing combination {combo}"

    # Check case counts
    assert result.loc[("A_A", pd.Period("2025-01", freq="M")), "Cases"] == 1
    assert result.loc[("A_A", pd.Period("2025-02", freq="M")), "Cases"] == 1
    assert result.loc[("A_B", pd.Period("2025-01", freq="M")), "Cases"] == 1
    assert result.loc[("A_B", pd.Period("2025-02", freq="M")), "Cases"] == 0


def test_aggregate_cases_epiweek_monday():
    # Sample data with missing province-date combos
    data = {
        "GID_1": ["A"] * 3,
        "GID_2": ["A_A", "A_A", "A_B"],
        "Date": ["2025-01-15", "2025-02-20", "2025-01-10"],
        "Status": ["active"] * 3,
    }
    df = pd.DataFrame(data)

    result = aggregate_cases(df, geo_col="GID_2", freq="W-SAT").df

    # Assert that Dates are periods, with month end frequency
    assert result["Date"].dtype == "period[W-SAT]"  # week end Saturday
    # Assert that weeks all start on Monday
    assert all(result["Date"].dt.start_time.dt.weekday == 6)  # 6 = Sunday


def test_aggregate_cases_epiweek_sunday():
    # Sample data with missing province-date combos
    data = {
        "GID_1": ["A"] * 3,
        "GID_2": ["A_A", "A_A", "A_B"],
        "Date": ["2025-01-15", "2025-02-20", "2025-01-10"],
        "Status": ["active"] * 3,
    }
    df = pd.DataFrame(data)

    result = aggregate_cases(df, geo_col="GID_2", freq="W-SUN").df

    # Assert that Dates are periods, with month end frequency
    assert result["Date"].dtype == "period[W-SUN]"
    # Assert that weeks all start on Sunday
    assert all(result["Date"].dt.start_time.dt.weekday == 0)  # 0 = Monday


def test_align_date_types_series():
    # Timestamp Series -> Period
    s = pd.to_datetime(
        pd.Series(["2025-01-15", "2025-02-20", "2025-01-10"], dtype="string")
    )
    df = pd.DataFrame(
        {
            "Date": [
                pd.Period("2025-01", freq="M"),
                pd.Period("2025-02", freq="M"),
                pd.Period("2025-01", freq="M"),
            ]
        }
    )
    aligned = align_date_types(s, df["Date"])
    assert aligned.dtype.name == "period[M]"


def test_align_date_types_datetime():
    # Timestamp scalar -> Period
    s = pd.to_datetime("2025-01-15")
    df = pd.DataFrame(
        {
            "Date": [
                pd.Period("2025-01", freq="M"),
                pd.Period("2025-02", freq="M"),
                pd.Period("2025-01", freq="M"),
            ]
        }
    )
    aligned = align_date_types(s, df["Date"])
    assert isinstance(aligned, pd.Period)
    assert not isinstance(aligned, pd.Timestamp)
    assert not isinstance(aligned, pd.Series)


def _grid_df():
    return pd.DataFrame(
        {
            "GID_2": ["A", "A", "A", "B", "B"],
            "Date": pd.PeriodIndex(
                ["2020-01", "2020-02", "2020-03", "2020-01", "2020-03"], freq="M"
            ),
            "Cases": [1, 2, 3, 4, 5],
        }
    )


def test_ensure_complete_grid_zero_fill():
    out = ensure_complete_grid(_grid_df(), drop_incomplete=False)
    assert len(out) == 6  # A x 3 months + B x 3 months
    b_feb = out[(out["GID_2"] == "B") & (out["Date"] == pd.Period("2020-02", "M"))]
    assert b_feb["Cases"].iloc[0] == 0
    # Observed values preserved
    a_jan = out[(out["GID_2"] == "A") & (out["Date"] == pd.Period("2020-01", "M"))]
    assert a_jan["Cases"].iloc[0] == 1


def test_ensure_complete_grid_drops_incomplete_by_default():
    out = ensure_complete_grid(_grid_df())  # drop_incomplete=True
    assert set(out["GID_2"]) == {"A"}  # B is missing 2020-02
    assert len(out) == 3
    assert str(out["Date"].dtype) == "period[M]"


def test_ensure_complete_grid_drops_na_outcome_units():
    df = _grid_df()
    # Give B all three months but a NA outcome -> still incomplete.
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                {
                    "GID_2": ["B"],
                    "Date": pd.PeriodIndex(["2020-02"], freq="M"),
                    "Cases": [float("nan")],
                }
            ),
        ],
        ignore_index=True,
    )
    out = ensure_complete_grid(df)
    assert set(out["GID_2"]) == {"A"}
    assert out["Cases"].notna().all()


def test_ensure_complete_grid_requires_period_or_freq():
    df = _grid_df()
    df["Date"] = df["Date"].dt.to_timestamp()
    with pytest.raises(ValueError, match="pass freq="):
        ensure_complete_grid(df)
    out = ensure_complete_grid(df, freq="M")
    assert set(out["GID_2"]) == {"A"}
