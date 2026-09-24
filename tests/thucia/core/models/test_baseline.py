import pandas as pd
from thucia.core.models.baseline import baseline
from thucia.core.models.baseline import BaselineSamples


def test_BaselineSamples_asymmetric():
    inc_diffs = [2, -1]  # Time series diffs (to be interpolated across nsim quantiles)
    model = BaselineSamples(inc_diffs, symmetrize=False)
    predictions = model.predict(last_level=10, horizon=1, nsim=5)
    assert predictions.shape == (5, 1)
    # All predictions should be in the interval [10 - 1, 10 + 2],
    # and more specifically at nsim(5) evenly sampled quantiles
    assert set(predictions.flatten()) == {9, 9.75, 10.5, 11.25, 12}


def test_BaselineSamples_symmetric():
    inc_diffs = [2]  # Time series diffs (to be interpolated across nsim quantiles)
    model = BaselineSamples(inc_diffs, symmetrize=True)
    predictions = model.predict(last_level=10, horizon=1, nsim=5)
    assert predictions.shape == (5, 1)
    # All predictions should be in the interval [10 - 2, 10 + 2],
    # and more specifically at nsim(5) evenly sampled quantiles
    assert set(predictions.flatten()) == {8, 9, 10, 11, 12}


def test_BaselineSamples_asymmetric_horizon():
    inc_diffs = [2, -1]  # Time series diffs (to be interpolated across nsim quantiles)
    model = BaselineSamples(inc_diffs, symmetrize=False)
    predictions = model.predict(last_level=10, horizon=3, nsim=5)
    assert predictions.shape == (5, 3)
    # First step should match single step prediction
    assert set(predictions[:, 0]) == {9.0, 9.75, 10.5, 11.25, 12.0}
    # Subsequent steps should be centred around the previous median which increment by
    #  0.5 in this case.
    assert set(predictions[:, 1]) == {9.5, 10.25, 11.0, 11.75, 12.5}
    assert set(predictions[:, 2]) == {10.0, 10.75, 11.5, 12.25, 13.0}


def test_BaselineSamples_symmetric_horizon():
    inc_diffs = [2]  # Time series diffs (to be interpolated across nsim quantiles)
    model = BaselineSamples(inc_diffs, symmetrize=True)
    predictions = model.predict(last_level=10, horizon=3, nsim=5)
    assert predictions.shape == (5, 3)
    # Symmetric predictions have zero median increments, so no drift / change compared
    #  to the one-step case (horizon of 1).
    assert set(predictions.flatten()) == {8, 9, 10, 11, 12}


# utility function
def quantiles_are_bounded(df: pd.DataFrame, bounds: list[float]) -> bool:
    lower, median, upper = sorted(bounds)
    return (
        df["prediction"].apply(lambda x: lower <= x <= upper).all()
        and df[df["quantile"] == 0.5]["prediction"].eq(median).all()
    )


# Ensure utility function works as expected
def test_quantiles_are_bounded():
    assert quantiles_are_bounded(
        pd.DataFrame(
            {
                "quantile": [0.1, 0.5, 0.9],
                "prediction": [8, 10, 12],
            }
        ),
        [8, 10, 12],
    )
    assert not quantiles_are_bounded(
        pd.DataFrame(
            {
                "quantile": [0.1, 0.5, 0.9],
                "prediction": [7, 10, 12],
            }
        ),
        [8, 10, 12],
    )


def test_baseline_symmetric():
    df = pd.DataFrame(
        {
            "Date": pd.period_range(start="2023-01", periods=8, freq="M"),
            "GID_1": ["A", "A", "A", "A", "A", "A", "A", "A"],
            "GID_2": ["X", "X", "X", "X", "X", "X", "X", "X"],
            "Cases": [10, 20, 30, 40, 50, 60, 70, 80],
            "future": [False, False, False, False, False, True, True, True],
        }
    )
    df_pred = baseline(
        df,
        start_date="2023-01",
        end_date="2023-08",
        geo_parent_filter=["A"],
        num_samples=3,  # quantiles are sampled without replacement
        # symmetrize should be True by default
    )
    # First three dates should have NA predictions
    assert df_pred[df_pred["Date"] == pd.Period("2023-01")]["prediction"].isna().all()
    assert df_pred[df_pred["Date"] == pd.Period("2023-02")]["prediction"].isna().all()
    assert df_pred[df_pred["Date"] == pd.Period("2023-03")]["prediction"].isna().all()
    # Next two dates should have predictions of Cases(k-1) + [-10, 0, 10] since there
    # is a constant gradient of 10, the baseline is symmetrised e.g. [-10, 10] and we
    # take three quantiles evenly sampled without replacement e.g. [-10, 0, 10]
    assert quantiles_are_bounded(
        df_pred[df_pred["Date"] == pd.Period("2023-04")],
        [20, 30, 40],
    )
    assert quantiles_are_bounded(
        df_pred[df_pred["Date"] == pd.Period("2023-05")],
        [30, 40, 50],
    )
    # Last three should be are future forecasts (this requires the model to substitute
    # predictions for observed cases for the one-step ahead prediction, internally)
    assert quantiles_are_bounded(
        df_pred[df_pred["Date"] == pd.Period("2023-06")],
        [40, 50, 60],
    )
    # Note that because we symmatrize, the predictions are the same
    assert quantiles_are_bounded(
        df_pred[df_pred["Date"] == pd.Period("2023-07")],
        [40, 50, 60],
    )
    # Note that because we symmatrize, the predictions are the same
    assert quantiles_are_bounded(
        df_pred[df_pred["Date"] == pd.Period("2023-08")],
        [40, 50, 60],
    )


def test_baseline_asymmetric():
    df = pd.DataFrame(
        {
            "Date": pd.period_range(start="2023-01", periods=8, freq="M"),
            "GID_1": ["A", "A", "A", "A", "A", "A", "A", "A"],
            "GID_2": ["X", "X", "X", "X", "X", "X", "X", "X"],
            "Cases": [10, 20, 30, 40, 50, 60, 70, 80],
            "future": [False, False, False, False, False, True, True, True],
        }
    )
    df_pred = baseline(
        df,
        start_date="2023-01",
        end_date="2023-08",
        geo_parent_filter=["A"],
        num_samples=3,  # quantiles are sampled without replacement
        symmetrize=False,
    )
    # First three dates should have NA predictions
    assert df_pred[df_pred["Date"] == pd.Period("2023-01")]["prediction"].isna().all()
    assert df_pred[df_pred["Date"] == pd.Period("2023-02")]["prediction"].isna().all()
    assert df_pred[df_pred["Date"] == pd.Period("2023-03")]["prediction"].isna().all()
    # Next two dates should have predictions of Cases(k-1) + [-10, 0, 10] since there
    # is a constant gradient of 10, the baseline is symmetrised e.g. [-10, 10] and we
    # take three quantiles evenly sampled without replacement e.g. [-10, 0, 10]
    assert set(
        df_pred[df_pred["Date"] == pd.Period("2023-04")]["prediction"].values.tolist()
    ) == {40}
    assert set(
        df_pred[df_pred["Date"] == pd.Period("2023-05")]["prediction"].values.tolist()
    ) == {50}
    # Last three should be are future forecasts (this requires the model to substitute
    # predictions for observed cases for the one-step ahead prediction, internally)
    assert set(
        df_pred[df_pred["Date"] == pd.Period("2023-06")]["prediction"].values.tolist()
    ) == {60}
    assert set(
        df_pred[df_pred["Date"] == pd.Period("2023-07")]["prediction"].values.tolist()
    ) == {70}
    assert set(
        df_pred[df_pred["Date"] == pd.Period("2023-08")]["prediction"].values.tolist()
    ) == {80}
