# Public API surface for thucia.core.
#
# Names resolve lazily (PEP 562) so importing `thucia.core` stays cheap —
# submodules are loaded only when a name is actually accessed. Internal
# implementation details live in the submodules and are not part of this
# surface.
from __future__ import annotations

import importlib
from typing import Any

#: public name -> submodule that provides it (functions/classes are read from
#: the submodule; ``None`` means the submodule itself is the object).
_API: dict[str, tuple[str, str | None]] = {
    "cache": ("thucia.core.cache", None),
    "cases": ("thucia.core.cases", None),
    "containers": ("thucia.core.containers", None),
    "fs": ("thucia.core.fs", None),
    "geo": ("thucia.core.geo", None),
    "models": ("thucia.core.models", None),
    "panels": ("thucia.core.panels", None),
    "pipeline": ("thucia.core.pipeline", None),
    "validation": ("thucia.core.validation", None),
    "DataFrame": ("thucia.core.fs", "DataFrame"),
    "read_db": ("thucia.core.cases", "read_db"),
    "write_db": ("thucia.core.cases", "write_db"),
    "read_nc": ("thucia.core.cases", "read_nc"),
    "write_nc": ("thucia.core.cases", "write_nc"),
    "read_zarr": ("thucia.core.cases", "read_zarr"),
    "write_zarr": ("thucia.core.cases", "write_zarr"),
    "wis": ("thucia.core.cases", "wis"),
    "r2": ("thucia.core.cases", "r2"),
    "rmse": ("thucia.core.cases", "rmse"),
    "run_model": ("thucia.core.models", "run_model"),
    "list_models": ("thucia.core.models", "list_models"),
    "get_model": ("thucia.core.models", "get_model"),
    "get_model_spec": ("thucia.core.models", "get_model_spec"),
    "ModelSpec": ("thucia.core.models._meta", "ModelSpec"),
    "PipelineConfig": ("thucia.core.pipeline", "PipelineConfig"),
    "cases_per_period": ("thucia.core.pipeline", "cases_per_period"),
    "merge_covariates": ("thucia.core.pipeline", "merge_covariates"),
    "prepare_model_inputs": ("thucia.core.pipeline", "prepare_model_inputs"),
    "fit_model": ("thucia.core.pipeline", "fit_model"),
    "score_model": ("thucia.core.pipeline", "score_model"),
    "aggregate_quantiles": ("thucia.core.pipeline", "aggregate_quantiles"),
    "build_ensemble": ("thucia.core.pipeline", "build_ensemble"),
    "apply_residual_regression": ("thucia.core.pipeline", "apply_residual_regression"),
    "run_backtest": ("thucia.core.validation", "run_backtest"),
    "BacktestConfig": ("thucia.core.validation", "BacktestConfig"),
    "BacktestResult": ("thucia.core.validation", "BacktestResult"),
    "expand_cutoffs": ("thucia.core.validation", "expand_cutoffs"),
    "load_case_source": ("thucia.core.cases.sources", "load_case_source"),
    "load_sources": ("thucia.core.cases.sources", "load_sources"),
    "ensure_complete_grid": ("thucia.core.cases", "ensure_complete_grid"),
    "attach_geo_attributes": ("thucia.core.geo", "attach_geo_attributes"),
    "panel_to_frame": ("thucia.core.panels", "panel_to_frame"),
    "read_rds": ("thucia.core.panels", "read_rds"),
    "Registry": ("thucia.core.registry", "Registry"),
    "PluginNotFoundError": ("thucia.core.registry", "PluginNotFoundError"),
}

__all__ = [
    "cache",
    "cases",
    "containers",
    "fs",
    "geo",
    "models",
    "panels",
    "pipeline",
    "validation",
    "DataFrame",
    "read_db",
    "write_db",
    "read_nc",
    "write_nc",
    "read_zarr",
    "write_zarr",
    "wis",
    "r2",
    "rmse",
    "run_model",
    "list_models",
    "get_model",
    "get_model_spec",
    "ModelSpec",
    "PipelineConfig",
    "cases_per_period",
    "merge_covariates",
    "prepare_model_inputs",
    "fit_model",
    "score_model",
    "aggregate_quantiles",
    "build_ensemble",
    "apply_residual_regression",
    "run_backtest",
    "BacktestConfig",
    "BacktestResult",
    "expand_cutoffs",
    "load_case_source",
    "load_sources",
    "ensure_complete_grid",
    "attach_geo_attributes",
    "panel_to_frame",
    "read_rds",
    "Registry",
    "PluginNotFoundError",
]


def __getattr__(name: str) -> Any:
    mapping = _API.get(name)
    if mapping is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_path, attr = mapping
    module = importlib.import_module(module_path)
    obj = module if attr is None else getattr(module, attr)
    globals()[name] = obj  # cache for future lookups
    return obj
