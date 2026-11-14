import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from lol_win.config import INPUT_COLUMNS, RANDOM_STATE, TARGET
from lol_win.data.clean import to_silver
from lol_win.data.ingest import read_raw, to_bronze
from lol_win.models.baselines import GoldRuleClassifier
from lol_win.models.evaluate import SCORING, summarize_cv
from lol_win.models.registry import CANDIDATES, MODEL_NAMES, build_model
from lol_win.models.train import ExperimentConfig, default_grid, run_grid, setup_mlflow


@pytest.fixture(scope="module")
def sample() -> tuple[pd.DataFrame, pd.Series]:
    silver = to_silver(to_bronze(read_raw())).sample(n=400, random_state=RANDOM_STATE)
    return silver[list(INPUT_COLUMNS)], silver[TARGET]


def test_gold_rule_probabilities() -> None:
    X = pd.DataFrame({"total_gold_diff": [1500, -200, 0]})
    rule = GoldRuleClassifier().fit(X, pd.Series([1, 0, 1]))
    np.testing.assert_allclose(rule.predict_proba(X)[:, 1], [1.0, 0.0, 0.5])
    assert rule.predict(X).tolist() == [1, 0, 0]
    assert clone(rule).get_params() == {"column": "total_gold_diff"}


def test_gold_rule_requires_gold_column() -> None:
    with pytest.raises(KeyError):
        GoldRuleClassifier().fit(pd.DataFrame({"x": [1]}), pd.Series([1]))


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_model_fits_and_predicts_probabilities(name, sample) -> None:
    X, y = sample
    proba = build_model(name).fit(X, y).predict_proba(X)
    assert proba.shape == (len(X), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_build_model_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="desconhecido"):
        build_model("xgboost")  # type: ignore[arg-type]


def test_summarize_cv_flips_negative_metrics() -> None:
    results = {"test_roc_auc": np.array([0.8, 0.82]), "test_log_loss": np.array([-0.5, -0.7])}
    summary = summarize_cv(results)
    assert summary["test_roc_auc_mean"] == pytest.approx(0.81)
    assert summary["test_log_loss_mean"] == pytest.approx(0.6)
    assert "train_roc_auc_mean" not in summary


def test_default_grid_shape() -> None:
    grid = default_grid()
    assert len(grid) == 2 + 4 * len(CANDIDATES)
    assert len({c.run_name for c in grid}) == len(grid)


def test_run_grid_smoke_logs_to_mlflow(sample, tmp_path) -> None:
    X, y = sample
    setup_mlflow(f"sqlite:///{tmp_path / 'mlflow.db'}", "test", tmp_path / "artifacts")
    grid = [ExperimentConfig("gold_rule"), ExperimentConfig("logreg", "diffs_core")]
    table = run_grid(grid, X, y, cv_folds=3, run_name="smoke")

    assert set(table["run_name"]) == {c.run_name for c in grid}
    assert {f"test_{m}_mean" for m in SCORING} <= set(table.columns)
    runs = mlflow.search_runs(experiment_names=["test"])
    assert len(runs) == len(grid) + 1
    assert (runs["tags.stage"].dropna() == "cv").sum() == len(grid)
