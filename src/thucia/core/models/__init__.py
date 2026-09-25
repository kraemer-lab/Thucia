from __future__ import annotations

import importlib
import logging
import pkgutil
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from thucia.core.cases import write_db
from thucia.core.fs import DataFrame
from thucia.core.geo import convert_to_incidence_rate  # noqa: F401

from .utils import samples_to_quantiles

# from .utils import filter_admin1  # noqa: F401
# from .utils import interpolate_missing_dates  # noqa: F401
# from .utils import quantiles  # noqa: F401
# from .utils import set_historical_na_to_zero  # noqa: F401

#
# Discover and lazy import model definitions
#

# discover candidate module -> symbol mappings once
#
# Contract: a module is a *model* iff it exposes a callable whose name matches
# the module name (e.g. `sarima.py` -> `sarima(df, ...)`). Helper libraries that
# live in this directory but do not satisfy that contract must be registered in
# _HELPER_MODULES so they are not advertised as models.
_HELPER_MODULES = frozenset({"ensemble", "quantiles"})

_exports: dict[str, str] = {}
for _m in pkgutil.iter_modules(__path__):
    name = _m.name
    if name.startswith("_") or _m.ispkg or name in _HELPER_MODULES:
        continue
    _exports[name] = name

__all__ = sorted(_exports)  # advertise what the package exports


def __getattr__(name: str) -> Any:  # called on first access if not yet in globals
    mod_name = _exports.get(name)
    if mod_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod = importlib.import_module(f".{mod_name}", __name__)
    obj = getattr(mod, name)  # expect symbol name == module name
    globals()[name] = obj  # cache for future lookups
    return obj


class _ModelNamespace(types.ModuleType):
    """Module subclass so a model callable always wins over its same-named
    submodule. Importing `thucia.core.models.<name>` (e.g. via a direct
    `from thucia.core.models.sarima import sarima`) makes the import system set
    the parent attribute to the *submodule*, shadowing the model function and
    breaking `getattr(models, name)` / `from thucia.core.models import name`.
    Intercept attribute access to resolve the callable regardless of import
    order.
    """

    def __getattribute__(self, name: str) -> Any:
        if name in _exports:
            try:
                current = object.__getattribute__(self, name)
            except AttributeError:
                current = None
            if callable(current):
                return current
            return __getattr__(name)
        return object.__getattribute__(self, name)


sys.modules[__name__].__class__ = _ModelNamespace


def list_models() -> list[str]:
    """Return the names of all advertised forecast models."""
    return list(__all__)


def get_model(name: str):
    """Resolve a model callable by name (raises ValueError if unknown)."""
    if name not in _exports:
        raise ValueError(f"Unknown model '{name}'. Available models: {list_models()}")
    return __getattr__(name)


def get_model_spec(name: str):
    """Resolve a model's declarative `ModelSpec` (lazy; never eager-imports).

    A conservative default is returned for any advertised model that does not
    (yet) declare a `SPEC`, so ad-hoc models never break the pipeline.
    """
    from ._meta import ModelSpec

    if name not in _exports:
        raise ValueError(f"Unknown model '{name}'. Available models: {list_models()}")
    mod = importlib.import_module(f".{name}", __name__)
    spec = getattr(mod, "SPEC", None)
    if spec is None:
        spec = ModelSpec(name=name)
    return spec


def run_model(
    name: str,
    model,
    df: pd.DataFrame,
    path: Path,
    save_samples=False,
    save_quantiles=True,
    model_args=None,
    model_kwargs=None,
):
    # Cases
    tic = pd.Timestamp.now()
    model_args = model_args if model_args is not None else []
    model_kwargs = model_kwargs if model_kwargs is not None else {}
    df_model = model(df, *model_args, **model_kwargs)
    toc = pd.Timestamp.now()

    if "Cases" not in df_model.columns and "Log_Cases" in df_model.columns:
        if isinstance(df_model, pd.DataFrame):
            df_model["Cases"] = np.expm1(df_model["Log_Cases"]).clip(lower=0)
            df_model["prediction"] = np.expm1(df_model["prediction"]).clip(lower=0)
        elif isinstance(df_model, DataFrame):
            raise NotImplementedError(
                "Exponentiation not implemented for thucia DataFrame"
            )

    logging.info(f"{name} model run time: {toc - tic}")

    # Save samples
    if save_samples and isinstance(df_model, pd.DataFrame):
        if "sample" not in df_model.columns:
            write_db(df_model, path / f"{name}_cases_samples")
        else:
            logging.warning(
                f"Model {name} did not produce samples, saving quantiles instead."
            )
            save_quantiles = True

    # Check if we need to convert samples to quantiles
    if save_quantiles and "quantile" not in df_model.columns:
        if "sample" in df_model.columns:
            if isinstance(df_model, pd.DataFrame):
                df_model = samples_to_quantiles(
                    df_model, geo_col=model_kwargs.get("geo_col", "GID_2")
                )
            elif isinstance(df_model, DataFrame):
                df_model = samples_to_quantiles(df_model.df)
            else:
                raise TypeError("df_model must be a pd.DataFrame or thucia DataFrame")
        else:
            logging.warning(
                f"Model {name} did not produce quantiles or samples, skipping save."
            )
            save_quantiles = False

    # Save quantiles
    if save_quantiles and not isinstance(df_model, DataFrame):
        if "quantile" in df_model.columns:
            write_db(df_model, path / f"{name}_cases_quantiles")
        else:
            logging.warning(
                f"Model {name} did not produce quantiles, cannot save quantiles."
            )

    # # Dengue incidence rate
    # df_dir_samples = convert_to_incidence_rate(df_cases_samples, df)
    # if save_samples:
    #     write_nc(df_dir_samples, path / f"{name}_dir_samples.nc")
    # if save_quantiles:
    #     df_dir_quantiles = samples_to_quantiles(df_dir_samples)
    #     write_nc(df_dir_quantiles, path / f"{name}_dir_quantiles.nc")

    return df_model
