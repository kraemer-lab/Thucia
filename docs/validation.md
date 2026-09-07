# Validation

Thucia validates forecast models by **backtesting**: cutting the observed
history at a series of cutoff dates, fitting the model on the past only,
scoring the held-out window, and aggregating a per-horizon error
(WIS / RMSE / R²) plus skill relative to a reference model.

The backtest consumes the **output of `prepare_model_inputs`**, so the expensive
geo and covariate merges run once rather than once per cutoff.

```python
from thucia.core import run_backtest, BacktestConfig

result = run_backtest(
    inputs,  # prepared model inputs
    cfg,  # a PipelineConfig
    BacktestConfig(
        model_name="baseline",
        reference_model="baseline",
        min_history=24,
        step=3,
    ),
)

print(result.summary)  # per-horizon mean WIS / RMSE / R2 / n (+ skill)
print(result.scores)  # per (cutoff, GID_2, Date, horizon) rows
```

## BacktestConfig

```{list-table}
:header-rows: 1
:widths: 20 14 66

* - Field
  - Default
  - Meaning
* - `model_name`
  - `"baseline"`
  - The model to fit at each cutoff.
* - `reference_model`
  - `None`
  - Optional reference model; when set, `summary` adds a per-horizon WIS skill
    score (`1 - WIS_model / WIS_ref`).
* - `cutoffs`
  - `None`
  - Explicit cutoff dates; when `None`, cutoffs are auto-generated.
* - `min_history`
  - `12`
  - Minimum observed periods up to (and including) a cutoff to allow it.
* - `step`
  - `1`
  - Spacing between auto-generated cutoffs (in periods).
* - `window`
  - `None`
  - `None` = expanding window; an int gives a rolling window of that many
    periods.
* - `fast_only`
  - `True`
  - Refuse models whose `ModelSpec.fast` is `False`. Only `baseline` and
    `movavg` are fast; pass `fast_only=False` to backtest heavier models.
* - `keep_forecasts`
  - `False`
  - Retain each cutoff's held-out quantile frame in `result.forecasts`.
```

## The result

`run_backtest` returns a `BacktestResult`:

- `scores` — one row per `(cutoff, GID_2, Date, horizon)` with `WIS`, `Cases`,
  `R2`, and `RMSE`.
- `summary` — per-horizon mean `WIS`, `RMSE`, and `R2` across windows, plus `n`
  (number of windows) and `skill` when a reference model is requested.
- `forecasts` — when `keep_forecasts=True`, a mapping `cutoff -> held-out
  quantile frame`.

By default backtest fits run in-memory and in a temporary directory, so a
backtest never litters your output tree.
