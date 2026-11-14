import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from lol_win.analysis.eda import (
    compute_vif,
    equality_rate,
    outlier_summary,
    save_figure,
    target_correlations,
    win_rate_by_bins,
    win_rate_by_group,
)


@pytest.fixture
def games() -> pd.DataFrame:
    return pd.DataFrame(
        {"gold": [-3000, -1000, 500, 2000, 4000, 6000], "dragon": [0, 0, 1, 0, 1, 1]}
    ).assign(blue_wins=[0, 0, 1, 0, 1, 1])


def test_win_rate_by_group_counts_and_ci(games: pd.DataFrame) -> None:
    result = win_rate_by_group(games, "dragon")
    assert result.loc[1, "win_rate"] == 1.0
    assert result.loc[0, "n"] == 3
    assert (result["ci_low"] <= result["win_rate"]).all()
    assert (result["win_rate"] <= result["ci_high"]).all()


def test_win_rate_by_bins_uses_explicit_edges(games: pd.DataFrame) -> None:
    result = win_rate_by_bins(games, "gold", bins=[-5000, 0, 5000, 10000])
    assert result["n"].tolist() == [2, 3, 1]
    assert result["center"].tolist() == [-2500, 2500, 7500]


def test_compute_vif_orthogonal_columns_near_one() -> None:
    rng = np.random.default_rng(42)
    features = pd.DataFrame(rng.normal(size=(2000, 3)), columns=["a", "b", "c"])
    assert compute_vif(features)["vif"].between(0.95, 1.05).all()


def test_compute_vif_detects_collinearity() -> None:
    rng = np.random.default_rng(42)
    a = rng.normal(size=2000)
    features = pd.DataFrame({"a": a, "b": a + rng.normal(scale=0.05, size=2000)})
    assert (compute_vif(features)["vif"] > 100).all()


def test_outlier_summary_labels() -> None:
    result = outlier_summary(pd.Series(range(101)), quantiles=(0.5, 0.99, 1.0))
    assert result.to_dict() == {"p50": 50, "p99": 99, "max": 100}


def test_target_correlations_sorted_by_abs(games: pd.DataFrame) -> None:
    result = target_correlations(games, ["dragon", "gold"])
    assert result["r"].abs().is_monotonic_decreasing


def test_equality_rate_with_float_tolerance() -> None:
    left = pd.Series([0.1 + 0.2, 1.0, 2.0])
    right = pd.Series([0.3, 1.0, 3.0])
    assert equality_rate(left, right) == pytest.approx(2 / 3)


def test_save_figure_writes_png(tmp_path) -> None:
    fig, _ = plt.subplots()
    path = save_figure(fig, "test_fig", figures_dir=tmp_path)
    plt.close(fig)
    assert path.is_file()
    assert path.suffix == ".png"
