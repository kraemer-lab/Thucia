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
`ENUM`s created with the full category list. When appending rows, all enum
categories must already be present from the first write.

```python
tdf.append(new_rows)  # all GID categories must exist from the first write
```

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

## Where files live

Downloads and derived caches are stored under the platform cache folder, which
you can query with `thucia.core.fs.get_cache_folder()` (on macOS this resolves
to `~/Library/Caches/global.Health/thucia/`). See {doc}`geo-covariates` for the
covariate caches kept there.
