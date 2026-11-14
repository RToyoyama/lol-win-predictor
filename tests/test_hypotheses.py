import numpy as np
import pandas as pd
import pytest
from scipy import stats

from lol_win.analysis.hypotheses import (
    bootstrap_gold_equivalent,
    chi2_association,
    cohens_h,
    fit_logit,
    gold_equivalent,
    minimum_detectable_effect,
    odds_ratio_table,
    proportion_test,
    raw_vs_adjusted_odds,
)

TRUE_GOLD_BETA = 0.0005
TRUE_DRAGON_BETA = 0.3
TRUE_GOLD_VALUE = TRUE_DRAGON_BETA / TRUE_GOLD_BETA  # 600 de gold


@pytest.fixture(scope="module")
def synthetic_games() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 20_000
    gold = rng.normal(0, 2500, n)
    dragon = rng.integers(-1, 2, n)
    logit = TRUE_GOLD_BETA * gold + TRUE_DRAGON_BETA * dragon
    wins = rng.random(n) < 1 / (1 + np.exp(-logit))
    return pd.DataFrame({"gold": gold, "dragon": dragon, "blue_wins": wins.astype(int)})


def test_proportion_test_matches_scipy() -> None:
    outcomes = pd.Series([1] * 520 + [0] * 480)
    result = proportion_test(outcomes)
    assert result.rate == pytest.approx(0.52)
    assert result.p_value == pytest.approx(stats.binomtest(520, 1000, 0.5).pvalue)
    assert result.ci_low < 0.52 < result.ci_high


def test_cohens_h_zero_when_equal() -> None:
    assert cohens_h(0.5, 0.5) == 0.0
    assert cohens_h(0.6, 0.5) == pytest.approx(-cohens_h(0.4, 0.5))


def test_minimum_detectable_effect_shrinks_with_n() -> None:
    small, large = minimum_detectable_effect(1_000), minimum_detectable_effect(10_000)
    assert large < small
    assert large == pytest.approx(0.014, abs=0.001)


def test_chi2_association_independent_vs_dependent() -> None:
    independent = pd.DataFrame({"x": [0, 1] * 500, "blue_wins": [0, 0, 1, 1] * 250})
    dependent = pd.DataFrame({"x": [0] * 500 + [1] * 500, "blue_wins": [0] * 450 + [1] * 550})
    assert chi2_association(independent, "x").p_value > 0.5
    result = chi2_association(dependent, "x")
    assert result.p_value < 1e-10
    assert 0 < result.cramers_v <= 1


def test_fit_logit_recovers_true_coefficients(synthetic_games: pd.DataFrame) -> None:
    params = fit_logit(synthetic_games, ["gold", "dragon"]).params
    assert params["gold"] == pytest.approx(TRUE_GOLD_BETA, rel=0.1)
    assert params["dragon"] == pytest.approx(TRUE_DRAGON_BETA, rel=0.15)


def test_odds_ratio_table_scales_per_unit(synthetic_games: pd.DataFrame) -> None:
    result = fit_logit(synthetic_games, ["gold", "dragon"])
    table = odds_ratio_table(result, per={"gold": 1000})
    assert "const" not in table.index
    assert table.loc["gold", "odds_ratio"] == pytest.approx(np.exp(result.params["gold"] * 1000))
    assert (table["or_ci_low"] < table["odds_ratio"]).all()


def test_gold_equivalent_is_coefficient_ratio() -> None:
    assert gold_equivalent({"gold": 0.0005, "dragon": 0.3}, "dragon", "gold") == pytest.approx(600)


def test_bootstrap_ci_covers_true_value_and_is_reproducible(
    synthetic_games: pd.DataFrame,
) -> None:
    kwargs = {"objectives": ["dragon"], "gold_column": "gold", "n_boot": 100}
    first = bootstrap_gold_equivalent(synthetic_games, **kwargs)
    second = bootstrap_gold_equivalent(synthetic_games, **kwargs)
    pd.testing.assert_frame_equal(first, second)
    row = first.loc["dragon"]
    assert row["ci_low"] < TRUE_GOLD_VALUE < row["ci_high"]


def test_raw_vs_adjusted_shrinks_confounded_effect() -> None:
    rng = np.random.default_rng(42)
    n = 20_000
    gold = rng.normal(0, 2500, n)
    first_blood = (rng.random(n) < 1 / (1 + np.exp(-gold / 1500))).astype(int)
    wins = (rng.random(n) < 1 / (1 + np.exp(-TRUE_GOLD_BETA * gold))).astype(int)
    df = pd.DataFrame({"gold": gold, "fb": first_blood, "blue_wins": wins})

    table = raw_vs_adjusted_odds(df, ["fb"], controls=["gold"])
    raw, adjusted = (
        table.loc[("fb", "bruto"), "odds_ratio"],
        table.loc[("fb", "ajustado"), "odds_ratio"],
    )
    assert raw > 1.5
    assert adjusted == pytest.approx(1.0, abs=0.1)
