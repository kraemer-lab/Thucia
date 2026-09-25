# Geo & covariate sources

Climate and population covariates are provided by a **plugin registry** in
`thucia.core.geo.sources`. Each source subclasses `SourceBase`, declares a
`ref`, and implements a `merge` that attaches its columns to the case grid. You
select sources by `"origin.field"` spec strings, e.g. `"worldclim.*"` or
`"edo.spi6"`, and merge them through
{func}`merge_sources <thucia.core.geo.merge_sources>`.

```python
from thucia.core.geo import merge_sources

df = merge_sources(df, ["worldclim.*", "edo.spi6", "worldpop.pop_count"])
```

Non-GADM geo schemes work too: raster sources extract values over the polygon
map you supply as `PipelineConfig.regions` (a shapefile/GeoPackage with
geometry, keyed by `region_col`), and `merge_sources`/`merge_covariates` thread
`geo_col`/`iso3` through every plugin. GADM-shaped codes need no map — the
cached GADM GeoPackage is used automatically.

The {doc}`pipeline` applies the configured `source_specs` automatically in the
`merge_covariates` stage.

## Available sources

```{list-table}
:header-rows: 1
:widths: 16 20 30 34

* - `ref`
  - Name
  - Columns produced
  - Notes
* - `worldclim`
  - WorldClim CRU-TS
  - `tmin`, `tmax`, `prec` (mean over the region)
  - Climatology from geodata.ucdavis.edu. Archival historical data with a CMIP6
    forecast fallback. The slowest pipeline step; results cached in SQLite.
* - `edo`
  - European Drought Observatory
  - `SPI6`
  - Standardised precipitation index. Multiprocessed, with a progress bar.
* - `noaa`
  - Oceanic Niño Index (ONI)
  - `TotalONI`, `AnomONI`
  - El Niño / La Niña indicator, merged on `Date` only (not per-region).
* - `worldpop`
  - WorldPop
  - `pop_count` (and population-estimate variants)
  - 1 km population rasters; `pop_count` feeds the incidence-rate column.
    Per-country rasters need a country key: explicit `iso3`, the regions map's
    `COUNTRY` column, or the GADM code prefix.
```

### Spec format

Covariates are selected with `"origin.field"` strings, where `field` may be
`*` to take everything a source produces (e.g. `"worldclim.*"`) or a specific
column (e.g. `"edo.spi6"`). Omitting the `.` raises a `ValueError`.

## Granularity & interpolation

Each source declares a native `granularity` (WorldClim, EDO, and NOAA are
`"M"` — monthly). When a source is coarser than the case grid (for example a
monthly source on **weekly** cases), `merge_sources` interpolates the source's
values onto every period, per region. The interpolation method is configurable
via `PipelineConfig.covariate_interpolation` (`"linear"` by default; also
`"ffill"` or `"bfill"`) and emits a `UserWarning` describing the fill.

## Caching

Downloads and derived statistics are cached under the platform cache directory
(`thucia.core.fs.get_cache_folder()`):

- `cache_folder/climate/` — WorldClim and EDO SQLite caches.
- `cache_folder/geo/<ISO3>/gadm41_<ISO3>.gpkg` — GADM admin-2 GeoPackages used
  for region alignment, padding, and map plotting.

```{note}
The first time you merge a covariate for a new country or period, Thucia
downloads the relevant data. This is why the WorldClim merge is the slowest
stage of the pipeline on a cold cache.
```
