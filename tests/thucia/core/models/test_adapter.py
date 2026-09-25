import numpy as np
import pandas as pd
import pytest
from thucia.core.models.utils.adapter import _align_embeddings
from thucia.core.models.utils.adapter import _ensure_geo_index
from thucia.core.models.utils.adapter import _prepare_fit_table
from thucia.core.models.utils.adapter import _residual_regression_fit_and_apply
from thucia.core.models.utils.adapter import HGBQuantileAdapter
from thucia.core.models.utils.adapter import MLPAdapter
from thucia.core.models.utils.adapter import QuantileAdapter
from thucia.core.models.utils.adapter import residual_regression
from thucia.core.models.utils.adapter import RidgeAdapter


def _embedding_df(n=2):
    return pd.DataFrame(
        {
            "GID_2": [f"G{i}" for i in range(n)],
            "feature0": [float(i) for i in range(n)],
            "feature1": [float(n - 1 - i) for i in range(n)],
        }
    )


def _predictions_df(n_dates=6, n=2, seed=0):
    dates = pd.period_range("2020-01", periods=n_dates, freq="M")
    rows = []
    for d in dates:
        for i in range(n):
            rows.append(
                {
                    "Date": d,
                    "GID_2": f"G{i}",
                    "prediction": 10.0 + i,
                    "Cases": 12.0 + i,  # constant +2 residual for every GID
                    "horizon": 1,
                    "quantile": 0.5,
                }
            )
    return pd.DataFrame(rows)


def test_ridge_adapter_learns_bias():
    ad = RidgeAdapter(_embedding_df().set_index("GID_2"), alpha=1.0)
    df = _predictions_df()
    ad.fit(df)
    # A per-GID bias is applied to predictions
    out = ad.apply(df.copy())
    # residuals (Cases - prediction) = +2 for all rows; bias should shrink them
    assert np.abs((out["prediction"] - df["Cases"]).mean()) < 2.0


def test_quantile_adapter_fit_and_apply():
    ad = QuantileAdapter(_embedding_df().set_index("GID_2"), quantile=0.5)
    ad.fit(_predictions_df())
    out = ad.apply(_predictions_df().copy())
    assert out["prediction"].notna().all()


def test_hgb_adapter_raises_without_data():
    ad = HGBQuantileAdapter(_embedding_df().set_index("GID_2"), quantile=0.5)
    with pytest.raises(ValueError):
        ad.fit(_predictions_df()[0:0])


def test_residual_regression_ridge_reduces_error():
    preds = _embedding_df()
    out = residual_regression(_predictions_df(), preds, method="ridge", geo_col="GID_2")
    assert len(out) == len(_predictions_df())
    rmse_before = np.sqrt(
        np.mean((_predictions_df()["prediction"] - _predictions_df()["Cases"]) ** 2)
    )
    rmse_after = np.sqrt(np.mean((out["prediction"] - out["Cases"]) ** 2))
    assert rmse_after < rmse_before


def test_residual_regression_unknown_method():
    with pytest.raises(Exception):
        residual_regression(
            _predictions_df(), _embedding_df(), method="nope", geo_col="GID_2"
        )


def test_residual_regression_requires_feature_columns():
    bad = _embedding_df().rename(columns={"feature0": "x0", "feature1": "x1"})
    # No feature* columns -> nothing to regress on, but must not crash badly
    out = residual_regression(_predictions_df(), bad, method="ridge", geo_col="GID_2")
    assert len(out) == len(_predictions_df())


# --- helper functions ---


def test_ensure_geo_index_column_to_index():
    df = pd.DataFrame({"GID_2": ["A", "B"], "x": [1.0, 2.0]})
    out = _ensure_geo_index(df)
    assert out.index.name == "GID_2"
    assert "GID_2" not in out.columns


def test_ensure_geo_index_passthrough_and_raise():
    df = _embedding_df().set_index("GID_2")
    assert _ensure_geo_index(df) is df
    with pytest.raises(ValueError, match="index or column"):
        _ensure_geo_index(pd.DataFrame({"x": [1.0]}))


def test_align_embeddings_missing_gid_raises():
    preds = _embedding_df().set_index("GID_2")
    with pytest.raises(ValueError, match="missing for"):
        _align_embeddings(preds, ["G0", "GHOST"])


def test_align_embeddings_orders_by_gids():
    preds = _embedding_df().set_index("GID_2")
    X, cols = _align_embeddings(preds, ["G1", "G0"])
    assert X[0, 0] == 1.0  # G1 first
    assert X[1, 0] == 0.0  # G0 second
    assert cols == ["feature0", "feature1"]


def test_prepare_fit_table_train_mask_and_cutoff():
    df = _predictions_df()
    mask = pd.Series([True, False] * (len(df) // 2))
    out = _prepare_fit_table(df, train_mask=mask)
    assert len(out) == mask.sum()
    cutoff = pd.Timestamp("2020-02-01")
    out2 = _prepare_fit_table(df, cutoff_date=cutoff)
    assert (out2["Date"].astype(str) <= str(pd.to_datetime(cutoff).date())).all()
    assert (out2["residual"] == 2.0).all()  # Cases - prediction = +2


def test_prepare_fit_table_empty_raises():
    with pytest.raises(ValueError, match="No rows"):
        _prepare_fit_table(_predictions_df()[0:0])


# --- AdapterBase behaviors ---


def test_bias_for_gid_unfitted_returns_zero():
    ad = RidgeAdapter(_embedding_df().set_index("GID_2"))
    assert ad.bias_for_gid("G0") == 0.0


def test_bias_for_gid_unknown_raises():
    ad = RidgeAdapter(_embedding_df().set_index("GID_2"))
    ad.fit(_predictions_df())
    with pytest.raises(ValueError, match="unknown to adapter"):
        ad.bias_for_gid("NOPE")


def test_apply_unfitted_warns_and_copies():
    ad = RidgeAdapter(_embedding_df().set_index("GID_2"))
    df = _predictions_df().copy()
    out = ad.apply(df, out_col="prediction")
    assert (out["prediction"] == df["prediction"]).all()


def test_adapter_fit_with_transform():
    # transform is applied to y and prediction before fitting; the learned bias
    # must still shrink the residual on the transformed scale.
    ad = RidgeAdapter(_embedding_df().set_index("GID_2"))
    df = _predictions_df().copy()
    ad.fit(df, transform=np.log1p)
    out = ad.apply(df.copy())
    # residual = +2 on raw scale -> log-residual > 0 -> prediction should rise
    assert (out["prediction"] > df["prediction"]).all()


def test_adapter_standardize_y_roundtrip():
    # With standardize_y=True the bias is de-standardized back onto the raw
    # residual scale, so apply() still reduces the error.
    ad = RidgeAdapter(_embedding_df().set_index("GID_2"), standardize_y=True)
    df = _predictions_df()
    ad.fit(df)
    out = ad.apply(df.copy())
    assert np.abs((out["prediction"] - df["Cases"]).mean()) < 2.0


def test_hgb_quantile_adapter_fit_and_predict():
    ad = HGBQuantileAdapter(
        _embedding_df().set_index("GID_2"), quantile=0.5, max_iter=50
    )
    df = _predictions_df()
    ad.fit(df)
    out = ad.apply(df.copy())
    assert out["prediction"].notna().all()


def test_mlp_adapter_fit_and_predict():
    pytest.importorskip("torch")
    ad = MLPAdapter(_embedding_df().set_index("GID_2"), epochs=2)
    df = _predictions_df()
    ad.fit(df)
    out = ad.apply(df.copy())
    assert out["prediction"].notna().all()


def test_lightgbm_adapter_fit_and_predict():
    pytest.importorskip("lightgbm")
    ad = _lightgbm_adapter()
    df = _predictions_df()
    ad.fit(df)
    out = ad.apply(df.copy())
    assert out["prediction"].notna().all()


def _lightgbm_adapter():
    from thucia.core.models.utils.adapter import LightGBMQuantileAdapter

    return LightGBMQuantileAdapter(
        _embedding_df().set_index("GID_2"), quantile=0.5, n_estimators=20
    )


# --- residual-regression machinery ---


def test_residual_regression_fit_apply_first_date_passthrough():
    # On the first target date there is no history to fit -> unchanged.
    df = _predictions_df(n_dates=3)
    dfh = df[df["quantile"] == 0.5]
    out = _residual_regression_fit_and_apply(
        dfh,
        horizon=1,
        q=0.5,
        df_predictors=_embedding_df().set_index("GID_2"),
        method="ridge",
        window=None,
        adapter=RidgeAdapter(_embedding_df().set_index("GID_2")),
    )
    # first date: no fit data (origin = d - 1 < first date) -> kept as-is
    first = out[out["Date"] == dfh["Date"].unique()[0]]
    assert (
        first["prediction"] == dfh[dfh["Date"] == dfh["Date"].unique()[0]]["prediction"]
    ).all()


def test_residual_regression_sliding_window_uses_bounded_history():
    df = _predictions_df(n_dates=8)
    dfh = df[df["quantile"] == 0.5]
    out = _residual_regression_fit_and_apply(
        dfh,
        horizon=1,
        q=0.5,
        df_predictors=_embedding_df().set_index("GID_2"),
        method="ridge",
        window=2,
        adapter=RidgeAdapter(_embedding_df().set_index("GID_2")),
    )
    assert len(out) == len(dfh)


def test_residual_regression_ridge_multihorizon_offset():
    # Non-0.5 quantiles get the median offset applied; the median keeps the fit.
    df = _predictions_df(n_dates=8)
    df2 = pd.concat(
        [df.assign(quantile=q, horizon=2) for q in [0.1, 0.5, 0.9]],
        ignore_index=True,
    )
    out = residual_regression(
        df2, _embedding_df(), method="ridge", geo_col="GID_2", horizons=[2]
    )
    assert set(out["quantile"].unique()) == {0.1, 0.5, 0.9}
    assert len(out) == len(df2)


def test_residual_regression_pinball_method():
    df = _predictions_df(n_dates=6)
    out = residual_regression(
        df, _embedding_df(), method="pinball", geo_col="GID_2", horizons=[1]
    )
    assert len(out) == len(df)


def test_residual_regression_mlp_method():
    pytest.importorskip("torch")
    df = _predictions_df(n_dates=6)
    out = residual_regression(
        df, _embedding_df(), method="mlp", geo_col="GID_2", horizons=[1]
    )
    assert len(out) == len(df)


def test_residual_regression_none_predictors_returns_unchanged():
    df = _predictions_df()
    out = residual_regression(df, None, method="ridge", geo_col="GID_2")
    assert out.equals(df)


def test_residual_regression_log_transform_roundtrip():
    # residual_regression applies log1p internally and expm1 on output, so
    # Case/prediction scales are preserved for an untransformed adapter.
    preds = _embedding_df()
    out = residual_regression(_predictions_df(), preds, method="ridge", geo_col="GID_2")
    assert out["Cases"].max() < 100  # back on raw count scale, not log
    assert np.allclose(out["Cases"], _predictions_df()["Cases"])
