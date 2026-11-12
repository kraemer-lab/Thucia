# Models

Thucia ships **thirteen forecast models** under `thucia.core.models`. They are
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
  - Amazon's Chronos-2 pretrained foundation model (zero-shot; past and future
    covariates). Emits its 13-level pre-trained quantile subset.
  - `torch`¹
* - `timesfm25`
  - darts
  - —
  - Google's TimesFM 2.5 foundation model (zero-shot; Apache-2.0 weights).
    Takes **no covariates**. Emits the 9-level pre-trained quantile grid.
  - `torch`¹
* - `timesfm3`
  - darts
  - —
  - Google's TimesFM 3.0 foundation model (zero-shot; native past/future
    covariates, variate attention). Emits the 9-level pre-trained quantile grid.
    **Weights are non-commercial** — requires `accept_license=True`.
  - `torch`¹
* - `inla`
  - container
  - —
  - R INLA statistical model run inside a container.
  - Docker/Podman
```

¹ The darts models need `torch`, which is pulled in by the base `darts[all]`
dependency; the foundation models need no extra of their own — see
{doc}`installation`.

## Foundation models and quantiles

`chronos`, `timesfm25` and `timesfm3` are pre-trained checkpoints with a *fixed*
quantile head, so the levels they can emit are limited to the grid the
checkpoint was trained on:

| Model | Pre-trained grid | Requesting another level |
|---|---|---|
| `chronos` | 13 levels (`CHRONOS_QUANTILES`): 0.01, 0.05, 0.1 … 0.9, 0.95, 0.99 | `ValueError` |
| `timesfm25`, `timesfm3` | 9 levels (`TIMESFM_QUANTILES`): 0.1, 0.2 … 0.9 | `ValueError` |

Two ways to choose what they emit:

```python
# a subset of the pre-trained grid
fit_model(df, "timesfm3", config, ...)        # honours config.quantiles
PipelineConfig(quantiles=[0.1, 0.5, 0.9])      # subset request

# or a deterministic median only (a single 0.5 component)
PipelineConfig(median_only=True)
```

Leaving `quantiles=None` uses the model's full pre-trained grid. Requesting a
level outside it raises (never silently relabels), and requesting a grid without
0.5 warns because WIS and the ensemble then interpolate the median. Because
{func}`create_ensemble <thucia.core.models.ensemble.create_ensemble>` intersects
the models' grids, mixing a foundation model into an ensemble reduces the common
grid (down to 9 levels if either TimesFM is included).

Covariates are routed per column:

```python
PipelineConfig(
    past_covariate_cols=None,             # None -> every covariate (default)
    future_covariate_cols="all",          # also pass them as future-known covariates
)
```

(`timesfm25` supports no covariates and warns when any are supplied.)

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

Every model shares a common signature (`df, start_date, end_date, geo_col,
geo_parent, geo_parent_filter, horizons=[...], case_col, covariate_cols,
retrain, db_file, train_col, multivariate, num_samples`) plus model-specific
knobs such as `season_length`. `geo_col` names the forecast region, `geo_parent`
its larger grouping, and `train_col` the column to fit per-region at (`None`
means a whole-dataset fit). Keep `horizons` as a **list** — the pipeline and
backtest pass multi-horizon lists to every model.

## Configured selections

A bare model name is enough for a single default fit, but a comparison run usually
needs more than a name: a label for the artefacts, and settings that differ from
the shared {class}`PipelineConfig <thucia.core.pipeline.PipelineConfig>`.
{class}`ModelSelection <thucia.core.models._meta.ModelSelection>` carries both,
and `fit_model` accepts it anywhere it accepts a name:

```python
from thucia.core import ModelSelection, fit_models

selections = [
    ModelSelection("baseline"),
    # same model, two configurations: each needs its own label
    ModelSelection(
        "sarima",
        label="sarima_retrain52",
        overrides={"retrain": True, "season_length": 52},
    ),
    ModelSelection("chronos", label="chronos_median", overrides={"median_only": True}),
    # params go straight to the model callable (here `baseline(symmetrize=...)`)
    ModelSelection("baseline", label="baseline_raw", params={"symmetrize": False}),
]

fits = fit_models(prepared_inputs, selections, config)   # {label: quantile frame}
```

- **`overrides`** are `PipelineConfig` field names applied for this model only;
  `fit_model` validates them against the config, and warns when the selected
  model never consumes one (it would otherwise be a silent no-op).
- **`params`** are extra keyword arguments for the model callable. Unknown names
  raise instead of being absorbed by a model's `**kwargs` and ignored.
- **`label`** names the output file (`{label}_cases_quantiles.duckdb`) and is the
  identifier to use in score frames. Two selections of the same model must have
  distinct labels; duplicates raise before anything is fitted.

`fit_model(df, selection_or_name, config)` fits one, `fit_models` fits several
against the same prepared frame, and
{attr}`BacktestConfig.model_name <thucia.core.validation.BacktestConfig>` accepts
a selection too, so a variant can be backtested with its overrides intact.

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
