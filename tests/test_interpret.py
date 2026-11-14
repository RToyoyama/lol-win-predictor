import numpy as np
import pandas as pd
import pytest

from lol_win.analysis.interpret import (
    BAND_LABELS,
    FEATURE_LABELS,
    confident_misses,
    coverage_accuracy_curve,
    decision_bands,
    orient_to_favorite,
    shap_explanation,
)
from lol_win.config import INPUT_COLUMNS, RANDOM_STATE, TARGET
from lol_win.data.clean import to_silver
from lol_win.data.ingest import read_raw, to_bronze
from lol_win.models.train import build_model


@pytest.fixture(scope="module")
def sample() -> tuple[pd.DataFrame, pd.Series]:
    silver = to_silver(to_bronze(read_raw())).sample(n=800, random_state=RANDOM_STATE)
    return silver[list(INPUT_COLUMNS)], silver[TARGET]


@pytest.mark.parametrize("name", ["logreg", "lightgbm"])
def test_shap_is_additive_and_in_original_units(name, sample) -> None:
    X, y = sample
    model = build_model(name, "diffs_core").fit(X, y)
    explanation = shap_explanation(model, X.head(20), background=X)

    proba = model.predict_proba(X.head(20))[:, 1]
    log_odds = np.log(proba / (1 - proba))
    shap_values = np.asarray(explanation.values)  # noqa: PD011 (shap.Explanation)
    reconstructed = shap_values.sum(axis=1) + explanation.base_values
    np.testing.assert_allclose(reconstructed, log_odds, atol=1e-4)

    gold = list(explanation.feature_names).index(FEATURE_LABELS["total_gold_diff"])
    expected_gold = (X["blue_total_gold"] - X["red_total_gold"]).head(20).to_numpy()
    np.testing.assert_allclose(explanation.data[:, gold], expected_gold)


def test_shap_rejects_unsupported_estimator(sample) -> None:
    X, y = sample
    model = build_model("gold_rule").fit(X, y)
    with pytest.raises(TypeError, match="não suportado"):
        shap_explanation(model, X.head(5), background=X)


def test_decision_bands_cover_all_games() -> None:
    p = np.array([0.05, 0.10, 0.30, 0.50, 0.70, 0.90, 0.95, 1.0])
    y = np.array([0, 1, 0, 1, 1, 1, 1, 1])
    bands = decision_bands(y, p)
    assert list(bands.index) == list(BAND_LABELS)
    assert bands["n"].sum() == len(p)
    assert bands["share"].sum() == pytest.approx(1.0)
    assert bands.loc["<15%", "favorite_accuracy"] == pytest.approx(0.5)
    assert bands.loc[">85%", "favorite_accuracy"] == 1.0


def test_coverage_accuracy_curve_is_monotonic_in_coverage() -> None:
    rng = np.random.default_rng(42)
    p = rng.uniform(0, 1, 5000)
    y = (rng.uniform(0, 1, 5000) < p).astype(int)
    curve = coverage_accuracy_curve(y, p)
    assert curve["coverage"].iloc[0] == 1.0
    assert curve["coverage"].is_monotonic_decreasing
    assert curve["accuracy"].iloc[-1] > curve["accuracy"].iloc[0]


def test_confident_misses_both_sides() -> None:
    p = np.array([0.90, 0.90, 0.10, 0.10, 0.60])
    y = np.array([0, 1, 1, 0, 0])
    assert confident_misses(y, p).tolist() == [True, False, True, False, False]


def test_orient_to_favorite_flips_red_favorites() -> None:
    features = pd.DataFrame({"total_gold_diff": [2000, -3000], "blue_first_blood": [1, 1]})
    oriented = orient_to_favorite(features, [0.8, 0.2])
    assert oriented["total_gold_diff"].tolist() == [2000, 3000]
    assert oriented["favorite_first_blood"].tolist() == [1, 0]
