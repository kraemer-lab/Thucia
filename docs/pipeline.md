# Pipeline

The forecasting workflow is built from a small set of **reusable, data-in /
data-out stages** in `thucia.core.pipeline`. Each stage is independent and
tested in isolation, so you can compose them, stop part-way, or swap in custom
steps. Use-case parameters (horizons, cutoffs, covariate sources, model lists)
live in a single {class}`PipelineConfig <thucia.core.pipeline.PipelineConfig>`
object rather than being hard-coded in the stages.

## The stage chain

```{mermaid}
flowchart LR
    A[Raw line-list cases] --> B[cases_per_period]
    B --> C[merge_covariates]
    C --> D[prepare_model_inputs]
    D --> E[fit_model]
    E --> F[score_model]
    E --> G[aggregate_quantiles]
    G --> H[build_ensemble]
    H --> I[apply_residual_regression]
```

## Stages

```{list-table}
:header-rows: 1
:widths: 25 35 40

* - Stage
  - Consumes
  - Produces
* - `cases_per_period(df, config, freq="M")`
  - Raw per-case rows (`Date`, `GID_1`, `GID_2`, `Cases`); any `geo_col`/`geo_parent` naming works — the rows are padded to every region before aggregation.
  - A period-aggregated frame with admin-2 regions padded to a common grid and
    `future_periods` of future placeholder rows (`Cases=NaN`, `future=True`).
* - `merge_covariates(df, config)`
  - The padded period frame.
  - The same frame with covariate columns merged on `[geo_col, "Date"]` plus the
    incidence-rate column `DIR`.
* - `prepare_model_inputs(df, config)`
  - The merged frame.
  - `(frame, covariate_cols)`: `Log_Cases`, optional lag features, sanitised and
    NaN-free covariates.
* - `fit_model(df, model_name, config, db_file=None)`
  - Prepared model inputs.
  - A quantile forecast frame (`Date`, `GID_2`, `quantile`, `prediction`, ...).
* - `score_model(df_quantiles, config)`
  - The quantile frame.
  - Per-horizon WIS and R² per region.
* - `aggregate_quantiles(df_quantiles, config)`
  - Per-region (GID_2) quantiles.
  - Quantiles aggregated to a coarser admin level via Monte-Carlo copula sums.
* - `build_ensemble(dfs, config, model_names=None)`
  - Several per-model quantile frames.
  - `(combined, weights)`: a weighted quantile ensemble.
* - `apply_residual_regression(df_quantiles, embeddings, config)`
  - A quantile frame plus user-supplied region embeddings.
  - The frame with post-hoc per-region forecast-error regression applied.
```

See the {doc}`API reference <api>` for exact signatures and defaults.

## PipelineConfig

All of the knobs the stages need are carried by `PipelineConfig`. The most
commonly changed fields:

```{list-table}
:header-rows: 1
:widths: 22 14 64

* - Field
  - Default
  - Meaning
* - `path`
  - `"."`
  - Output directory for DB/NetCDF artifacts.
* - `iso3` / `adm1`
  - `None`
  - Country / admin-1 filter used to resolve region codes.
* - `horizons`
  - `[1, 3, 6, 12]`
  - Forecast horizons (in periods of the data frequency) to produce and score.
* - `future_periods`
  - `12`
  - Number of future rows to append per region.
* - `start_date`, `train_start_date`, `train_end_date`
  - `None`
  - Date bounds for training and forecasting.
* - `cutoff_date`
  - `None`
  - Optional fixed cutoff separating training from forecast data.
* - `source_specs`
  - `["worldclim.*", "edo.spi6", "noaa.oni", "worldpop.pop_count"]`
  - Covariate source specs to merge, in `"origin.field"` form.
* - `covariate_interpolation`
  - `"linear"`
  - How coarser-granularity covariates are interpolated onto the case grid
    (`"linear"`, `"ffill"`, or `"bfill"`).
* - `regions` / `region_col`
  - `None`
  - Optional polygon map (shapefile/GeoPackage path, or in-memory GeoDataFrame)
    keyed by `region_col` holding the `geo_col` codes. When supplied it drives
    both region padding and raster covariate extraction, so non-GADM geo schemes
    work end-to-end; GADM-shaped codes need no map.
* - `geo_col` / `geo_parent`
  - `"GID_2"` / `"GID_1"`
  - Geo unit columns: `geo_col` is the forecast region, `geo_parent` its larger
    grouping (used for filtering/subsetting). `geo_parent=None` omits the parent
    column from padded rows. `cases_per_period` pads the frame to every region
    via `ensure_all_regions`: an explicit `regions=` roster first, then the GADM
    admin-2 list for GADM-shaped codes, then a categorical column's
    `.cat.categories` as an implicit non-GADM region list.
* - `train_col`
  - `None`
  - Column to fit per-region at (e.g. `GID_1` for an admin-1 fit); when `None`
    the pipeline fits per `geo_col`.
* - `case_col`
  - `"Log_Cases"`
  - Column used by the models.
* - `num_samples`
  - `200`
  - Number of posterior samples for models that sample.
* - `retrain`
  - `False`
  - Whether to retrain on every forecast.
* - `season_length`
  - `None`
  - Season length for `sarima`; auto-detected when `None` (M→12, W→52, D→365).
* - `lag_spec`
  - `None`
  - Optional lag-feature recipe; when `None` every non-base input column is used
    as a covariate.
```

## Cadence

Monthly, weekly, and daily data are supported end-to-end. `cases_per_period`
handles any pandas period frequency (weekly anchors such as `W-SAT` / `W-SUN`
are preserved), and the models and covariate interpolation adapt to the data
frequency automatically. When a covariate source is coarser than the case grid
(e.g. monthly sources on weekly cases), its values are interpolated onto every
period and a `UserWarning` is emitted.

## Quantiles

Every model emits forecasts on a single canonical quantile grid in
`thucia.core.quantiles` (15 levels from 0.01 to 0.99). Import it rather than
redefining it — `chronos` is the one exception, using a documented 13-level
subset. Models that internally produce posterior samples are converted to
quantiles by `run_model`.
