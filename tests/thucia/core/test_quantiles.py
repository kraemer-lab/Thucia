import numpy as np
import pandas as pd
from thucia.core.cases import quantile_sum_fast
from thucia.core.models.chronos import CHRONOS_QUANTILES
from thucia.core.models.quantiles import quantiles as models_quantiles
from thucia.core.models.utils import quantiles as utils_quantiles
from thucia.core.models.utils.residual_quantiles import quantiles as residual_quantiles
from thucia.core.quantiles import quantiles as canonical


def test_canonical_quantiles_are_a_single_object():
    # All entry points must resolve to the same list object (single source of truth)
    assert utils_quantiles is canonical
    assert models_quantiles is canonical
    assert residual_quantiles is canonical


def test_canonical_grid_is_sorted_and_symmetric():
    assert canonical == sorted(canonical)
    assert canonical[0] > 0 and canonical[-1] < 1
    # symmetric around 0.5 (use tolerance: 1 - 0.7 != 0.3 in float)
    for q in canonical:
        assert any(np.isclose(1 - q, other) for other in canonical)
    assert 0.5 in canonical


def test_chronos_quantiles_are_a_subset_of_canonical():
    assert set(CHRONOS_QUANTILES) <= set(canonical)
    assert 0.025 not in CHRONOS_QUANTILES  # documented native-chronos limitation


def test_quantile_sum_fast_default_grid_matches_canonical():
    # The default probabilities grid (used when probabilities=None) must match
    # the canonical quantile set.
    preds = np.linspace(1, 10, len(canonical))
    df = pd.DataFrame(
        {
            "Date": ["2020-01"] * (len(canonical) * 2),
            "GID_2": (["G0"] * len(canonical)) + (["G1"] * len(canonical)),
            "horizon": [1] * (len(canonical) * 2),
            "quantile": canonical + canonical,
            "prediction": list(preds) + list(preds + 1),
        }
    )
    out = quantile_sum_fast(
        df, date="2020-01", gids=["G0", "G1"], horizon=1, probabilities=None
    )
    assert out["quantile"].tolist() == canonical
