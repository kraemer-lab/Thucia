import pandas as pd
from thucia.core.cases import aggregate_cases
from thucia.core.cases import align_date_types
from thucia.core.cases import cases_per_month


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
