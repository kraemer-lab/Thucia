import importlib
import inspect
import pkgutil

import pytest
import thucia.core.models as models


def _all_module_names():
    # Every module file (including helper libraries) for import-cleanliness checks
    return sorted(
        m.name
        for m in pkgutil.iter_modules(models.__path__)
        if not m.name.startswith("_") and not m.ispkg
    )


def test_models_package_advertises_only_models():
    # __all__ lists the auto-discovered models (per the lazy __getattr__ contract)
    assert len(models.__all__) > 0
    for name in models.__all__:
        assert hasattr(models, name), f"{name} advertised but not resolvable"


def test_each_advertised_model_is_callable():
    for name in models.__all__:
        assert callable(getattr(models, name)), f"{name} is not callable"


def test_unknown_model_raises_attribute_error():
    with pytest.raises(AttributeError):
        models.definitely_not_a_model


def test_model_functions_accept_keyword_contract():
    for name in models.__all__:
        fn = getattr(models, name)
        sig = inspect.signature(fn)
        assert "df" in sig.parameters, f"{name} missing required arg 'df'"


def test_all_modules_import_cleanly():
    for name in _all_module_names():
        importlib.import_module(f"thucia.core.models.{name}")


def test_list_models_matches_all():
    assert models.list_models() == models.__all__


def test_get_model_resolves_callable():
    assert callable(models.get_model("baseline"))


def test_get_model_unknown_raises():
    with pytest.raises(ValueError, match="not_a_model"):
        models.get_model("not_a_model")


# Keyword set passed by thucia.core.pipeline.fit_model / run_backtest to
# run_model() for every model (the common kwargs + knobs declared in each
# model's ModelSpec.supports).
COMMON_KWARGS = {
    "start_date": "2020-01",
    "geo_col": "GID_2",
    "geo_parent": "GID_1",
    "geo_parent_filter": None,
    "horizons": [1, 3, 6, 12],
    "case_col": "Log_Cases",
    "covariate_cols": ["x"],
    "train_col": "GID_2",
    "db_file": None,
}
#: Canonical knob name -> the kwarg `fit_model` passes to the model callable.
KNOW_TO_KWARG = {
    "train_end_date": "train_end_date",
    "retrain": "retrain",
    "multivariate": "multivariate",
    "samples": "num_samples",
    "season_length": "season_length",
}
KNOB_VALUES = {
    "train_end_date": "2022-01",
    "retrain": False,
    "multivariate": False,
    "samples": 200,
    "season_length": 12,
}


def test_each_advertised_model_resolves_a_spec():
    # Adding a model with new knobs should be a one-line SPEC, not an edit to
    # the pipeline; every advertised model must expose a valid ModelSpec.
    for name in models.__all__:
        spec = models.get_model_spec(name)
        assert spec.name == name
        assert callable(getattr(models, name)), f"{name} is not callable"


def test_spec_supports_are_known_and_bindable():
    # Whatever a spec declares must (a) be a known knob and (b) bind on the
    # model callable's signature together with the common kwargs -- this is the
    # guarantee that fit_model's spec-driven dispatch stays valid. The
    # container-based `inla` model has its own ``inla(df, gid_1)`` interface and
    # is not part of the shared forecasting contract.
    from thucia.core.models._meta import SUPPORTED_KWARGS

    for name in models.__all__:
        if name == "inla":
            continue
        spec = models.get_model_spec(name)
        assert spec.supports <= SUPPORTED_KWARGS, f"{name}: unknown supports"
        sig = inspect.signature(getattr(models, name))
        kw = dict(COMMON_KWARGS)
        for knob in spec.supports:
            kw[KNOW_TO_KWARG[knob]] = KNOB_VALUES[knob]
        sig.bind_partial(**kw)  # raises TypeError on unknown kwargs


def test_spec_fast_flag_matches_known_fast_models():
    # Only the cheap statistical baselines are marked fast (the fast-only
    # backtest allowlist now derives from the specs, not a hard-coded set).
    for name in models.__all__:
        expected = name in {"baseline", "movavg"}
        assert models.get_model_spec(name).fast is expected, name


def test_horizons_default_is_a_list():
    # All models share a list-form `horizons` default (never a scalar `horizon`).
    # baseline/movavg are loose (no horizons param) and inla has its own interface.
    for name in models.__all__:
        if name in {"baseline", "movavg", "inla"}:
            continue
        sig = inspect.signature(getattr(models, name))
        p = sig.parameters["horizons"]
        assert p.default == [1], f"{name}.horizons default should be [1]"
