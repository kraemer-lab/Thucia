# Data layer

Thucia reads and writes case data through a thin `thucia.core.fs` layer backed
by **DuckDB** (the default), **NetCDF**, and **Zarr**. All the helpers are
re-exported so you can `from thucia.core import read_db, write_db, ...`.

## Files, not just in-memory frames

```python
from thucia.core import read_db, write_db, DataFrame

# Write a pandas frame to a DuckDB database file.
write_db(df, "my_run/cases")

# Read it back as a lazy Thucia DataFrame.
tdf = read_db("my_run/cases")
print(tdf.head())
```

`read_db` / `write_db` default to `.duckdb` files and append the extension for
you if you omit it. `read_db` also understands `.nc` and `.zarr` (and, when you
omit an extension, tries `.duckdb`, then `.nc`, then `.zarr`).

## The lazy DataFrame

{class}`DataFrame <thucia.core.fs.DataFrame>` is a **lazy wrapper over a
DuckDB table** — columns, index subsets, and boolean filters are pushed down as
SQL rather than materialising the whole frame in memory.

```python
tdf["Log_Cases"]  # SELECT one column (lazy)
tdf[["GID_2", "Cases"]]  # SELECT a subset of columns
tdf.head(10)  # first rows
tdf.query("Cases > 100")  # arbitrary SQL filter
len(tdf)  # SELECT COUNT(*)
tdf.columns  # column names

# Materialise the whole table as a pandas DataFrame (restores Period dtypes).
pandas_df = tdf.df
```

```{note}
`write_db` writes a single table named `data`. When you construct a
`DataFrame` yourself without specifying `table`, the single user table (a table
whose name does not start with `__`) is auto-detected.
```

## Special columns handled transparently

**Period columns.** Case dates are typically `pd.Period`s (e.g. monthly). They
cannot be stored natively in DuckDB, so they are written as end-timestamps with
their metadata recorded in a `__column_metadata__` table, and restored to their
`Period` dtype on read. The same machinery round-trips periods for NetCDF/Zarr
(via Dataset attributes).

**Categorical geo codes.** `GID_1`, `GID_2`, and `Status` are stored as DuckDB
`ENUM`s. On every write/append the ENUM is (re)built as the union of the codes
already stored and the ones in the incoming frame, so geo columns can grow
across append pieces:

```python
tdf.append(new_rows)  # geo ENUM automatically extends to the new codes
```

The geo categoricals are decoded back to strings for the `INSERT` — DuckDB
casts those `VARCHAR`s into the ENUM column at runtime (registering the raw
pandas categorical codes instead would collide with the table's ENUM).

**Aliasing legacy `GID_1`/`GID_2`.** `read_db`/`read_nc`/`read_zarr` accept
`geo_col`, `geo_parent`, and `migrate`. Files written under the old GADM column
names are exposed under the logical names you ask for, without any rename:

```python
tdf = read_db("my_run/cases", geo_col="region", geo_parent="state")
tdf.columns  # ['Date', 'state', 'region', 'Cases']
```

Pass `migrate=True` (DuckDB only) to physically rename the stored columns; for
NetCDF/Zarr the aliasing is applied to the pandas frame on read.

## Run metadata (provenance)

Fitted artefacts record how they were produced. A model fitted through
`fit_model` writes a `__run_metadata__` table (DuckDB) or a JSON Dataset
attribute (NetCDF/Zarr) holding five documents: `selection` (model, label,
overrides, params), `config` (the effective `PipelineConfig`, including the
regions map as a path or frame description), `code` (the Thucia implementation
that ran — see below), `inputs` (shape plus a structural digest of the observed
rows) and `environment` (python version, platform, timestamp).

```python
from thucia.core import read_run_metadata, read_db

docs = read_run_metadata("my_run/sarima52_cases_quantiles.duckdb")
docs["selection"]["overrides"]  # {'retrain': True, 'season_length': 52}
read_db("my_run/sarima52_cases_quantiles.duckdb").metadata == docs
```

The metadata is written by `write_db(..., metadata={...})` when an artefact is
created, or `DataFrame.set_metadata(docs)` for a file produced by something
else. It is preserved on `append` (never clobbered by more rows), replaced on a
full rewrite, and absent-from-old-file reads simply return `{}`.

### Pinning the implementation

`code` identifies the code that produced the artefact, so "same model, same
data, same assumptions" can be checked rather than assumed:

```text
{
  "source": "editable",          # editable | git | index | local | archive | source
  "distribution": {"name": "Thucia", "version": "0.1.dev135+g7d41bf06d.d20261112"},
  "commit": {"id": "2fb8b62…", "short": "2fb8b62", "describe": "2fb8b62", "dirty": true},
  "direct_url": {"url": "file:///…/Thucia", "dir_info": {"editable": true}},
  "digest": "sha256:e282c7de…",  # content fingerprint of the imported package
  "digest_basis": "sources",     # "sources" (directory walk) or "record" (wheel RECORD)
  "deps": {"darts": "0.48.1", "torch": "2.10.0", "numpy": "2.3.5", …}
}
```

The **digest** is the pin that works everywhere: a sha256 over the package's
`*.py` sources (path-tagged, sorted, `__pycache__` and a setuptools-scm
generated `_version.py` excluded), so it identifies the code actually imported
even for a dirty editable checkout — and it needs no git repository, so it
survives a published wheel. `distribution.version` is build-time metadata and
goes stale for editable installs, so prefer the digest.

`commit` is populated **only** when the running package *is* a checked-out
source tree (`<repo>/src/thucia` or `<repo>/thucia`); a wheel installed into a
project's own `.venv` deliberately does not inherit the surrounding
repository's commit, and `dirty` is `null` rather than `false` when it cannot be
observed.

```python
from thucia.core import code_document, verify_code_identity

code_document()["digest"]
verify_code_identity(read_run_metadata("my_run/forecast.duckdb"))  # {} == identical
```

`fit_model`/`fit_models` accept pins and refuse to fit when they are not met:

```python
fit_model(df, "baseline", config, expect="my_run/baseline_cases_quantiles.duckdb")
fit_model(df, "baseline", config, expect_commit="2fb8b62")  # clean tree required
fit_model(df, "baseline", config, expect_version="0.2.0")  # registry installs
```

`expect_commit` raises for an install that carries no commit (a wheel) rather
than passing silently — pin those by version or by an artefact's digest.

## NetCDF / Zarr

NetCDF and Zarr are convenient for sharing ready-to-use datasets, and they
preserve period columns automatically:

```python
from thucia.core import write_nc, read_nc, write_zarr, read_zarr

write_nc(df, "cases.nc")
back = read_nc("cases.nc")  # Period Date column restored
```

`write_nc` / `write_zarr` accept either a pandas DataFrame or an xarray
Dataset.

## R data frames (`.rds` / `.rda`)

Pre-aggregated panels are sometimes shared as R data.frames. `read_rds` loads
one into a pandas DataFrame using the optional `rdata` package (install with
`pip install thucia[panels]`):

```python
from thucia.core import read_rds

panel = read_rds("cases_panel.rds")  # -> pandas DataFrame
```

When the file holds several named objects, pass `name=` to pick one. See
{doc}`pipeline` for `panel_to_frame`, which maps such a panel onto the pipeline
schema.

## Where files live

Downloads and derived caches are stored under the platform cache folder, which
you can query with `thucia.core.fs.get_cache_folder()` (on macOS this resolves
to `~/Library/Caches/global.Health/thucia/`). See {doc}`geo-covariates` for the
covariate caches kept there.

### Relocating the cache

The cache root is resolved **on every call** (never frozen at import), in this
order:

1. an explicit `thucia.core.set_cache_folder(path)` override, for a single
   process;
2. the `THUCIA_CACHE_DIR` environment variable;
3. the platform cache directory above (`XDG_CACHE_HOME` is already honoured on
   Linux through `platformdirs`; the variable is the portable option).

This matters on HPC systems where the home directory has a small quota: point
the cache at scratch space before running anything.

```console
$ export THUCIA_CACHE_DIR=/scratch/$USER/thucia-cache
$ thucia provenance run.duckdb      # any command uses the new root
```

```python
import thucia.core

thucia.core.set_cache_folder("/scratch/me/thucia-cache")  # or None to reset
print(thucia.core.get_cache_folder())
```

The directory is not created for you; sources create the subdirectories they
write to. An empty (or fresh) root behaves like a cold cache, so the GADM
GeoPackages and WorldClim/EDO/WorldPop rasters are downloaded again the first
time they are needed.
