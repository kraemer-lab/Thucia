import logging
import warnings
from dataclasses import dataclass
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import QuantileRegressor
from sklearn.linear_model import Ridge


def _ensure_geo_index(
    predictors_df: pd.DataFrame, geo_col: str = "GID_2"
) -> pd.DataFrame:
    if predictors_df.index.name == geo_col:
        return predictors_df
    if geo_col in predictors_df.columns:
        return predictors_df.set_index(geo_col)
    raise ValueError(f"predictors_df must have index or column named '{geo_col}'.")


def _align_embeddings(
    predictors_df: pd.DataFrame, gids: List[str]
) -> Tuple[np.ndarray, List[str]]:
    missing = [g for g in gids if g not in predictors_df.index]
    if missing:
        raise ValueError(
            f"Embeddings missing for {len(missing)} gid(s). First few: {missing[:5]}"
        )
    M = predictors_df.reindex(gids)
    return M.to_numpy(dtype=np.float32), list(M.columns)


def _prepare_fit_table(
    dfm: pd.DataFrame,
    use_cols: Tuple[str, str, str] = ("Date", "GID_2", "prediction"),
    y_col: str = "Cases",
    train_mask: Optional[pd.Series] = None,
    cutoff_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    date_col, geo_col, yhat_col = use_cols
    if train_mask is not None:
        dfm = dfm[train_mask.values].copy()
    if cutoff_date is not None:
        dates = dfm[date_col]
        if isinstance(dates.dtype, pd.PeriodDtype):
            # Compare on the same Period grid as the model data
            cutoff = pd.Period(pd.to_datetime(cutoff_date), freq=dates.dt.freq)
        else:
            cutoff = pd.to_datetime(cutoff_date)
        dfm = dfm[dates <= cutoff].copy()
    if dfm.empty:
        raise ValueError("No rows available to fit adapter after masking/cutoff.")
    dfm["residual"] = dfm[y_col].astype(np.float32) - dfm[yhat_col].astype(np.float32)
    return dfm


# ---------- Base ----------


@dataclass
class _EmbeddingSpace:
    X: np.ndarray  # (N, D) embeddings aligned to gid_order
    gid_order: List[str]  # length N


class AdapterBase:
    """
    Base class. Subclasses implement _fit_impl(X, y) and _predict_impl(X) for residuals.
    """

    def __init__(
        self,
        predictors_df: pd.DataFrame,
        geo_col: str = "GID_2",
        standardize_y: bool = False,
    ):
        self.predictors_raw = _ensure_geo_index(predictors_df, geo_col=geo_col)
        self.geo_col = geo_col
        self.standardize_y = standardize_y

        self._space: Optional[_EmbeddingSpace] = None
        self._y_mean: Optional[float] = None
        self._y_std: Optional[float] = None
        self._fitted: bool = False

    # ---- public API ----
    def fit(
        self,
        df: pd.DataFrame,
        use_cols: Tuple[str, str, str] = ("Date", "GID_2", "prediction"),
        y_col: str = "Cases",
        train_mask: Optional[pd.Series] = None,
        cutoff_date: Optional[pd.Timestamp] = None,
        transform=None,
    ):
        if transform:
            df[y_col] = transform(df[y_col])
            df["prediction"] = transform(df["prediction"])

        dfm = _prepare_fit_table(df, use_cols, y_col, train_mask, cutoff_date)
        gid_order = sorted(dfm[self.geo_col].unique().tolist())
        X_raw, _ = _align_embeddings(self.predictors_raw, gid_order)

        X = X_raw

        # one row per timepoint: replicate per gid embedding to match residual rows
        gid_to_idx = {g: i for i, g in enumerate(gid_order)}
        idxs = dfm[self.geo_col].map(gid_to_idx)
        idxs = np.asarray(idxs.astype(int))
        X_fit = X[idxs]  # (T_total, D)
        y = dfm["residual"].to_numpy(np.float32)  # (T_total,)

        if self.standardize_y:
            self._y_mean = float(y.mean())
            self._y_std = float(y.std() + 1e-8)
            y_std = (y - self._y_mean) / self._y_std
        else:
            y_std = y

        self._space = _EmbeddingSpace(X=X, gid_order=gid_order)
        self._fit_impl(X_fit, y_std)  # <- subclass
        self._fitted = True

        X_fit = X_fit[~np.isnan(y_std),]
        y_std = y_std[~np.isnan(y_std)]

        # quick train fit score (in-sample)
        # pred_train = self._predict_impl(X_fit)
        # r2 = 1.0 - np.sum((y_std - pred_train) ** 2) / np.sum(
        #     (y_std - y_std.mean()) ** 2
        # )
        # logging.info("[Adapter] In-sample R^2 on residuals: %.3f", r2)

    def bias_for_gid(self, gid: str) -> float:
        if not self._fitted or self._space is None:
            return 0.0
        try:
            i = self._space.gid_order.index(gid)
        except ValueError:
            raise ValueError(
                f"{self.geo_col}={gid} unknown to adapter; was it included during fit?"
            )
        x = self._space.X[i : i + 1]  # (1,D)
        pred = self._predict_impl(x)[0]
        if self.standardize_y:
            pred = pred * (self._y_std or 1.0) + (self._y_mean or 0.0)
        return float(pred)

    def apply(
        self,
        pred_df: pd.DataFrame,
        date_col: str = "Date",
        geo_col: str = "GID_2",
        pred_col: str = "prediction",
        out_col: str = "prediction",
    ) -> pd.DataFrame:
        """
        Adds per-geo bias to ALL rows (works for per-sample long DataFrames too).
        """
        if not self._fitted:
            logging.warning(
                "Adapter not fitted; returning input unchanged with copied column."
            )
            out = pred_df
            out[out_col] = out[pred_col]
            return out

        # vectorized: map gid->bias
        map_bias = {g: self.bias_for_gid(g) for g in self._space.gid_order}
        out = pred_df
        out[out_col] = out[pred_col] + out[geo_col].map(map_bias).astype(np.float32)
        return out

    # ---- subclass hooks ----
    def _fit_impl(self, X: np.ndarray, y: np.ndarray):
        raise NotImplementedError

    def _predict_impl(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


# ---------- Ridge adapter (simple & strong baseline) ----------


class RidgeAdapter(AdapterBase):
    def __init__(self, *args, alpha: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self._ridge = None

    def _fit_impl(self, X: np.ndarray, y: np.ndarray):
        X = X[~np.isnan(y),]
        y = y[~np.isnan(y)]

        self._ridge = Ridge(alpha=self.alpha, fit_intercept=True, random_state=0)
        self._ridge.fit(X, y)

    def _predict_impl(self, X: np.ndarray) -> np.ndarray:
        return self._ridge.predict(X).astype(np.float32)


# ---------- Tiny MLP adapter ----------

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    nn = None


class MLPAdapter(AdapterBase):
    def __init__(
        self,
        *args,
        hidden: int = 64,
        l2: float = 1e-4,
        epochs: int = 200,
        lr: float = 1e-3,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.hidden = hidden
        self.l2 = l2
        self.epochs = epochs
        self.lr = lr
        self._mlp = None
        self._in_dim = None

    def _build_model(self, in_dim: int):
        m = nn.Sequential(
            nn.Linear(in_dim, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, 1),
        )
        for p in m.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return m

    def _fit_impl(self, X: np.ndarray, y: np.ndarray):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for MLPAdapter.")
        X_t = torch.from_numpy(X.astype(np.float32))
        y_t = torch.from_numpy(y.astype(np.float32)).view(-1, 1)

        self._in_dim = X.shape[1]
        self._mlp = self._build_model(self._in_dim)
        opt = torch.optim.Adam(self._mlp.parameters(), lr=self.lr, weight_decay=self.l2)
        loss_fn = nn.MSELoss()

        self._mlp.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            pred = self._mlp(X_t)
            loss = loss_fn(pred, y_t)
            loss.backward()
            opt.step()

    def _predict_impl(self, X: np.ndarray) -> np.ndarray:
        self._mlp.eval()
        with torch.no_grad():
            X_t = torch.from_numpy(X.astype(np.float32))
            out = self._mlp(X_t).squeeze(1).cpu().numpy().astype(np.float32)
        return out


# Quantile adapter


class QuantileAdapter(AdapterBase):
    def __init__(self, *args, quantile: float = 0.5, alpha: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        assert 0.0 < quantile < 1.0
        self.q = float(quantile)
        self.alpha = float(alpha)
        self._qr = None

    def _fit_impl(self, X: np.ndarray, y: np.ndarray):
        # Drop NaNs if not already filtered
        mask = ~np.isnan(y)
        Xf = X[mask]
        yf = y[mask]

        # Fit a linear model minimizing pinball loss
        self._qr = QuantileRegressor(quantile=self.q, alpha=self.alpha, solver="highs")
        self._qr.fit(Xf, yf)

    def _predict_impl(self, X: np.ndarray) -> np.ndarray:
        return self._qr.predict(X).astype(np.float32)


class HGBQuantileAdapter(AdapterBase):
    """
    HistGradientBoostingRegressor(loss='quantile').
    """

    def __init__(
        self,
        *args,
        quantile: float = 0.5,
        max_iter: int = 200,
        learning_rate: float = 0.1,
        max_leaf_nodes: Optional[int] = 31,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert 0.0 < quantile < 1.0
        self.q = float(quantile)
        self.max_iter = int(max_iter)
        self.learning_rate = float(learning_rate)
        self.max_leaf_nodes = max_leaf_nodes
        self._model = None

    def _fit_impl(self, X: np.ndarray, y: np.ndarray):
        # X: (N, D), y: (N,)
        mask = ~np.isnan(y)
        if mask.sum() == 0:
            raise ValueError("No non-NaN targets to fit.")
        Xf = X[mask]
        yf = y[mask]

        # sklearn expects double dtype; keep float32 for memory but HGB accepts float32
        self._model = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=self.q,
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_leaf_nodes=self.max_leaf_nodes,
            random_state=0,
        )
        # Fit (will be much faster than repeated LP solves)
        self._model.fit(Xf, yf)

    def _predict_impl(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not fitted yet")
        return self._model.predict(X).astype(np.float32)


try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except Exception:
    LIGHTGBM_AVAILABLE = False
    lgb = None


class LightGBMQuantileAdapter(AdapterBase):
    """
    Fast LightGBM quantile adapter. Very fast on medium->large datasets.
    Requires `lightgbm` package.
    """

    def __init__(
        self,
        *args,
        quantile: float = 0.5,
        n_estimators: int = 200,
        learning_rate: float = 0.1,
        num_leaves: int = 31,
        n_jobs: int = 1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("lightgbm is required for LightGBMQuantileAdapter.")
        assert 0.0 < quantile < 1.0
        self.q = float(quantile)
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.num_leaves = int(num_leaves)
        self.n_jobs = int(n_jobs)
        self._model = None

    def _fit_impl(self, X: np.ndarray, y: np.ndarray):
        mask = ~np.isnan(y)
        if mask.sum() == 0:
            raise ValueError("No non-NaN targets to fit.")
        Xf = X[mask]
        yf = y[mask]

        # lightgbm expects 2D array; it's fine with float32
        self._model = lgb.LGBMRegressor(
            objective="quantile",
            alpha=self.q,  # quantile parameter
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            n_jobs=self.n_jobs,
            random_state=0,
        )
        # Fit
        self._model.fit(Xf, yf)

    def _predict_impl(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not fitted yet")
        return self._model.predict(X).astype(np.float32)


def _residual_regression_fit_and_apply(
    df, horizon, q, df_predictors, method, window, geo_col="GID_2", adapter=None
):
    dfh = df[(df["quantile"] == q) & (df["horizon"] == horizon)]

    if method == "pinball":
        # base_alpha = 1e-1  # 0=pure pinball, 1=baseline model
        adapter = HGBQuantileAdapter(
            predictors_df=df_predictors,
            standardize_y=False,
            quantile=q,
            geo_col=geo_col,
            # alpha=base_alpha / (q * (1 - q)),  # tails need more regularisation
        )
    elif adapter is None:
        raise ValueError(f"Adapter instance must be provided for method '{method}'.")

    # Expanding window per date (strictly causal)
    out_slices = []

    # Convert once to Timestamps for robust comparisons
    dates = dfh["Date"]
    unique_dates = np.sort(dates.unique())

    # Step over target dates
    for d in unique_dates:
        logging.info(f"Fitting adapter for horizon {horizon}, quantile {q}, date {d}")

        # Prediction origin is target_date minus horizon (prevents Cases from
        # leaking into the data, e.g. i+4 into an i+12 prediction)
        origin_date = d - horizon
        if window:
            # Sliding window
            mask_fit = (dates <= origin_date) & (dates > origin_date - window)
        else:
            # Expanding window
            mask_fit = dates <= origin_date

        # Apply correction to the target date only
        mask_apply = dates == d

        df_fit = dfh.loc[mask_fit].copy()
        df_apply = dfh.loc[mask_apply].copy()

        if df_fit.empty:
            # Nothing to fit yet; keep predictions as-is for the first date
            out_slices.append(df_apply)
            continue

        try:
            adapter.fit(df=df_fit, transform=None)
        except ValueError as e:
            logging.warning(f"Adapter failed to fit at date {d}: {e}")
            out_slices.append(df_apply)
            continue

        pred_df_corrected = adapter.apply(
            df_apply, out_col="prediction", geo_col=geo_col
        )
        df_apply.loc[:, "prediction"] = pred_df_corrected["prediction"]
        out_slices.append(df_apply)

    df_out = pd.concat(out_slices, axis=0)
    return df_out


def residual_regression(
    df_model, df_predictors, method, geo_col="GID_2", window=None, horizons=None
):
    # Prepare embeddings
    if df_predictors is None:
        logging.warning("Model predictors not provided. Skipping adapter.")
        return df_model

    provinces = df_model[geo_col].unique()
    df_predictors = df_predictors[df_predictors[geo_col].isin(provinces)]
    dupes = df_predictors.loc[df_predictors[geo_col].duplicated(), geo_col].unique()
    if len(dupes):
        # Duplicate geo codes indicate an encoding error in the embeddings
        # file; warn and keep the first occurrence of each code.
        warnings.warn(
            f"Embeddings contain duplicate '{geo_col}' codes: {list(dupes)[:5]}"
            " — this indicates an encoding error; keeping the first occurrence.",
            UserWarning,
            stacklevel=2,
        )
        df_predictors = df_predictors.drop_duplicates(subset=[geo_col])
    df_predictors.set_index(geo_col, inplace=True)
    feature_cols = [c for c in df_predictors.columns if c.startswith("feature")]
    df_predictors = df_predictors[feature_cols]
    embedding_gids = set(df_predictors.index.unique())
    missing = set(df_model[geo_col]) - embedding_gids
    if missing:
        # Some provinces have no embeddings: warn and continue on the rest
        # (mirrors the old analysis_core.py call-site subsampling).
        warnings.warn(
            f"No embeddings for {len(missing)} geo code(s): "
            f"{sorted(missing)[:5]}{'...' if len(missing) > 5 else ''} "
            "— continuing with the provinces that have embeddings.",
            UserWarning,
            stacklevel=2,
        )
        df_model = df_model[df_model[geo_col].isin(embedding_gids)]
        if df_model.empty:
            raise ValueError(
                "No model provinces have embeddings; cannot run residual regression."
            )

    adapter = None
    if method == "ridge":
        adapter = RidgeAdapter(
            predictors_df=df_predictors,
            standardize_y=False,
            alpha=2.0,
            geo_col=geo_col,
        )
    elif method == "mlp":
        adapter = MLPAdapter(
            predictors_df=df_predictors,
            standardize_y=False,
            geo_col=geo_col,
        )
    elif method == "pinball":
        # Build per quantile loop
        pass
    else:
        raise Exception(f"Unrecognised method requested: {method}")

    log_transform = True
    df_work = df_model.copy()
    if log_transform:
        df_work["Cases"] = np.log1p(df_work["Cases"])
        df_work["prediction"] = np.log1p(df_work["prediction"])

    # Ensure a horizon column exists (kept from original behavior)
    if "horizon" not in df_work.columns:
        df_work["horizon"] = 1
    if "quantile" not in df_work.columns:
        df_work["quantile"] = 0.5

    if not horizons:
        horizons = df_work["horizon"].unique()
    quantiles = sorted(df_work["quantile"].unique())

    df_out = []
    for horizon in horizons:
        # logging.info(f"Fitting adapter (expanding window) for horizon {horizon}")

        if method == "pinball":
            for q in quantiles:
                # logging.info(f"Processing quantile {q} for horizon {horizon}")
                df_out.append(
                    _residual_regression_fit_and_apply(
                        df=df_work,
                        horizon=horizon,
                        q=q,
                        df_predictors=df_predictors,
                        method=method,
                        window=window,
                        geo_col=geo_col,
                        adapter=adapter,
                    )
                )
        else:
            median_fit = _residual_regression_fit_and_apply(
                df=df_work,
                horizon=horizon,
                q=0.5,
                df_predictors=df_predictors,
                method=method,
                window=window,
                geo_col=geo_col,
                adapter=adapter,
            )
            df_orig = df_work[
                (df_work["horizon"] == horizon) & (df_work["quantile"] == 0.5)
            ]
            # Merge original predictions into median_fit
            median_fit = median_fit.merge(
                df_orig[["Date", geo_col, "prediction"]],
                on=["Date", geo_col],
                suffixes=("", "_orig"),
            )
            median_fit["offset"] = (
                median_fit["prediction"] - median_fit["prediction_orig"]
            )

            for q in quantiles:
                if q == 0.5:
                    df_out.append(
                        median_fit.drop(columns=["prediction_orig", "offset"])
                    )
                else:
                    # Apply offset to other quantiles
                    df_q = df_work[
                        (df_work["horizon"] == horizon) & (df_work["quantile"] == q)
                    ].copy()
                    df_q = df_q.merge(
                        median_fit[["Date", geo_col, "offset"]],
                        on=["Date", geo_col],
                        how="left",
                    )
                    df_q["prediction"] = df_q["prediction"] + df_q["offset"]
                    df_q.drop(columns=["offset"], inplace=True)
                    df_q["horizon"] = horizon
                    df_out.append(df_q)

    df_out_all = pd.concat(df_out, axis=0)

    if log_transform:
        df_out_all["Cases"] = np.expm1(df_out_all["Cases"])
        df_out_all["prediction"] = np.expm1(df_out_all["prediction"])

    # Return in the original row order
    # df_out_all = df_out_all.loc[df_model.index]

    # Sort
    df_out_all = df_out_all.sort_values(
        by=["Date", geo_col, "horizon", "quantile"]
    ).reset_index(drop=True)

    return df_out_all
