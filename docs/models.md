# Models

Thucia ships **twelve forecast models** under `thucia.core.models`. They are
discovered automatically at import, described by declarative metadata, and
called through a **unified interface** with a list-form `horizons` argument.

List the available models and resolve one by name:

```python
from thucia.core import list_models, get_model

print(list_models())  # all advertised model names
model = get_model("sarima")  # resolve the callable
```

## The catalogue

```{list-table}
:header-rows: 1
:widths: 12 16 8 30 22

* - Name
  - Family
  - Fast
  - Description
  - Extra needed
* - `baseline`
  - statistical
  - ✓
  - Simple seasonal baseline; the default reference forecast.
  - —
* - `movavg`
  - statistical
  - ✓
  - Moving-average / seasonal-average reference.
  - —
* - `sarima`
  - statistical
  - —
  - Seasonal ARIMA (`season_length` auto-detected from the data frequency).
  - —
* - `tcn`
  - darts
  - —
  - Temporal convolutional network.
  - `torch`¹
* - `tft`
  - darts
  - —
  - Temporal Fusion Transformer.
  - `torch`¹
* - `nbeats`
  - darts
  - —
  - N-BEATS deep forecasting (slow to fit; used for the highest quality fits).
  - `torch`¹
* - `nhits`
  - darts
  - —
  - N-HiTS, a fast, interpretable N-BEATS variant.
  - `torch`¹
* - `tide`
  - darts
  - —
  - TiDE: a simple MLP forecaster.
  - `torch`¹
* - `xgboost`
  - darts
  - —
  - Gradient-boosted trees with a quantile likelihood.
  - —
* - `chronos`
  - darts
  - —
  - A pretrained foundation forecasting model.
  - `chronos`
* - `timesfm`
  - darts
  - —
  - Google's TimesFM foundation model (needs a PyTorch-enabled build).
  - `timesfm[torch]`
* - `inla`
  - container
  - —
  - R INLA statistical model run inside a container.
  - Docker/Podman
```

¹ The darts models need `torch`, which is pulled in by the base `darts[all]`
dependency. `timesfm` and `chronos` additionally need their own extras — see
{doc}`installation`.

## Which models are "fast"?

The {doc}`validation` backtest refuses to fit heavy models across many cutoff
windows by default. Only models marked `fast` (currently `baseline` and
`movavg`) are allowed when `BacktestConfig(fast_only=True)`. Pass
`fast_only=False` to backtest the heavier models.

```python
from thucia.core import BacktestConfig

BacktestConfig(model_name="sarima", fast_only=False)  # allow a slow model
```

## Running a model

In most cases you won't call a model directly — the pipeline fits it through
{func}`fit_model <thucia.core.pipeline.fit_model>`. Calling
`run_model` directly is also supported:

```python
from thucia.core import run_model

out = run_model("sarima", model, prepared_df, path="my_run")
# writes `sarima_cases_quantiles.duckdb`
```

Every model shares a common signature (`df, start_date, end_date, gid_1,
horizons=[...], case_col, covariate_cols, retrain, db_file,
model_admin_level, multivariate, num_samples`) plus model-specific knobs such
as `season_length`. Keep `horizons` as a **list** — the pipeline and backtest
pass multi-horizon lists to every model.

## Declarative metadata (ModelSpec)

Model dispatch is **metadata-driven**, not special-cased by name. Each model
module declares a module-level `ModelSpec` describing its family, whether it is
`fast`, which config knobs it `supports`, how it samples, and which extras it
needs.

```python
from thucia.core import get_model_spec

spec = get_model_spec("sarima")
print(spec.family, spec.supports, spec.sampling)
```

Because `fit_model` and the backtest read the spec, adding a model with new
knobs is a one-line `SPEC` — no changes to the pipeline dispatch.
