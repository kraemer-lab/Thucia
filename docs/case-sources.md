# Case data sources

Case ingestion lives behind a small **plugin registry** in
`thucia.core.cases.sources`, mirroring the covariate-source idiom. A
`CaseSource` driver fetches a case table for a given disease and geography; the
disease is always a parameter, never hard-coded in a driver.

## Loading a source

```python
from thucia.core import load_case_source, load_sources

load_sources()  # import every driver so the registry is populated
src = load_case_source("infodengue", iso3="BRA")
df = src.fetch(disease="dengue")
```

- `load_sources()` imports every driver module in the `sources` package;
  per-module errors are logged and skipped, so one broken driver doesn't block
  the others.
- `load_case_source(ref, **params)` looks up the driver by `ref` (raising
  `PluginNotFoundError` for unknown refs) and returns a **configured instance**.
  Constructor defaults can be overridden per call to `fetch(**params)`.
- `CaseSource` is the base class: subclasses set `ref` and `name`, self-register
  via `@case_registry.register()`, and implement `fetch(**params) -> DataFrame`.

The registry ({func}`thucia.core.cases.sources.case_registry`) is a
{class}`thucia.core.registry.Registry` — the same primitive that backs the
covariate sources and cache backends, so plugin discovery is consistent across
the library.

## InfodengueSource (`ref="infodengue"`)

Infodengue (a Brazilian arboviral surveillance service) reports cases by IBGE
municipality code. `InfodengueSource` queries the alert API, aggregates the
weekly rows to **monthly** totals per municipality, and returns a frame with
columns `ADM1, ADM2, Date, Cases` plus (after alignment) `GID_1, GID_2`.

```python
from thucia.core import load_case_source

src = load_case_source("infodengue", iso3="BRA", states=["Rondônia"])
df = src.fetch(disease="zika", align=True)
```

### fetch parameters

```{list-table}
:header-rows: 1
:widths: 22 78

* - Parameter
  - Meaning
* - `disease`
  - Which disease to fetch — `"dengue"` (default), `"chikungunya"`, or
    `"zika"`. **This is a parameter, not a hard-coded constant.**
* - `iso3`
  - Country code. Infodengue covers **Brazil only**; any other value raises a
    `ValueError`.
* - `states`
  - Optional list of IBGE state names to filter to (`None` = all).
* - `geocodes`
  - Optional explicit IBGE municipality codes, which bypasses downloading the
    IBGE codebook.
* - `ey_start`, `ew_start`, `ey_end`, `ew_end`
  - Epidemiological year / week window; defaults to roughly the last year.
* - `align`
  - Whether to align municipality names to GADM admin regions via
    `align_admin2_regions` (default `True`).
* - `out_path`
  - If set, writes the result to a NetCDF file with `write_nc`.
```

```{note}
The driver is **disease-generic by contract**: nothing about dengue is baked
into the class name or the dispatch, and the disease is supplied as a
`fetch()` parameter. This makes it straightforward to add drivers for other
diseases or geographies by subclassing `CaseSource` and registering a new
`ref`.
```
