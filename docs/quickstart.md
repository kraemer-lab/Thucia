# Quickstart

This page walks through a complete forecasting run on small synthetic data. It
uses only the fast `baseline` model, so it runs entirely offline.

## 1. Build a small pipeline config

The pipeline is driven by a single
{class}`PipelineConfig <thucia.core.pipeline.PipelineConfig>` object that holds
the use-case choices (horizons, date bounds, model list, covariate sources).

```python
import pandas as pd
from thucia.core import PipelineConfig

cfg = PipelineConfig(
    path="my_run",  # output directory
    start_date=pd.Period("2019-01", freq="M"),
    train_end_date=pd.Period("2018-12", freq="M"),
    horizons=[1, 3, 6],
    num_samples=50,
    future_periods=12,
    source_specs=["worldclim.*"],  # covariates to merge
)

# cfg.path is created automatically
cfg.path.mkdir(parents=True, exist_ok=True)
```

## 2. Prepare a raw case table

The pipeline starts from a **line list**: one row per case with a date and an
admin-2 region. For this example we fabricate two regions over four months.

```python
import numpy as np

raw = pd.DataFrame(
    {
        "Date": pd.to_datetime(
            ["2019-01-15", "2019-02-15", "2019-03-15", "2019-04-15"] * 2
        ),
        "GID_1": ["BRA.1_1"] * 4 + ["BRA.2_1"] * 4,
        "GID_2": ["BRA.1.1_1"] * 4 + ["BRA.2.1_1"] * 4,
        "Cases": [2, 3, 5, 4, 1, 2, 3, 2],
    }
)
```

## 3. Run the pipeline stages

```python
from thucia.core import (
    cases_per_period,
    merge_covariates,
    prepare_model_inputs,
    fit_model,
    score_model,
)

# Aggregate to months, pad admin-2 regions, and append future placeholder rows.
padded = cases_per_period(raw, cfg, freq="M")

# Merge covariates and add the incidence-rate column (pop_count-based).
merged = merge_covariates(padded, cfg) if cfg.source_specs else padded

# Build the model-input frame: Log_Cases, covariates, sanitised and NaN-free.
inputs, covariate_cols = prepare_model_inputs(merged, cfg)

# Fit the "baseline" model and get a quantile forecast frame.
quantiles = fit_model(inputs, "baseline", cfg, db_file=None)

# Score each horizon: WIS and R2 per region.
scored = score_model(quantiles, cfg)
print(scored[["GID_2", "horizon", "WIS", "R2"]].head())
```

```{note}
In a real deployment the covariate sources (WorldClim and friends) download
data over the network and cache it locally — see
{doc}`geo-covariates`. The merge of `worldclim.*` is the slowest pipeline step.
```

## 4. Backtest for model selection

Instead of fitting a single model, backtest a candidate model across a sweep of
cutoff dates to estimate its generalisation error:

```python
from thucia.core import run_backtest, BacktestConfig

result = run_backtest(
    inputs,
    cfg,
    BacktestConfig(model_name="baseline", min_history=3, step=1),
)

print(result.summary)  # per-horizon mean WIS / RMSE / R2
print(result.scores.head())  # per (cutoff, region, date, horizon)
```

By default backtesting refuses models that are not "fast" (only `baseline` and
`movavg`); pass `fast_only=False` to allow the heavier models — see
{doc}`validation`.

## 5. Ingest real case data

Fetch case data from a provider via the case-source plugin registry. The
Infodengue driver covers Brazilian municipalities and takes the disease as a
parameter:

```python
from thucia.core import load_case_source

source = load_case_source("infodengue", iso3="BRA", states=["Rondônia"])
df = source.fetch(disease="zika", align=False)
print(df.head())  # ADM1, ADM2, Date, Cases
```

See {doc}`case-sources` for the full parameter list.

## Where next?

- {doc}`pipeline` — the stage-by-stage reference.
- {doc}`models` — the full model catalogue.
- {doc}`validation` — backtesting and forecasting skill.
