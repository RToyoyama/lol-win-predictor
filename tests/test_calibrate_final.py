import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

from lol_win.config import INPUT_COLUMNS, RANDOM_STATE, TARGET
from lol_win.data.clean import to_silver
from lol_win.data.ingest import read_raw, to_bronze
from lol_win.models.calibrate import (
    Variant,
    build_variant,
    choose_variant,
    compare_calibration,
    expected_calibration_error,
)
from lol_win.models.evaluate import bootstrap_metrics, compute_metrics
from lol_win.models.final import LeakageAlarmError, check_leakage, gold_rule_probabilities
from lol_win.models.predict import load_model, predict_proba
from lol_win.models.tune import TuningResult

LOGREG = TuningResult(
    model="logreg",
    feature_set="diffs_core",
    params={"C": 0.25, "l1_ratio": 1.0, "solver": "saga", "max_iter": 5000},
    metrics={"test_log_loss_std": 0.01},
    n_trials=1,
)


@pytest.fixture(scope="module")
def sample() -> tuple[pd.DataFrame, pd.Series]:
    silver = to_silver(to_bronze(read_raw())).sample(n=600, random_state=RANDOM_STATE)
    return silver[list(INPUT_COLUMNS)], silver[TARGET]


def test_ece_zero_for_perfect_and_positive_for_overconfident() -> None:
    rng = np.random.default_rng(42)
    p = rng.uniform(0, 1, 50_000)
    y = (rng.uniform(0, 1, p.size) < p).astype(int)
    assert expected_calibration_error(y, p) < 0.01
    overconfident = np.clip((p - 0.5) * 2 + 0.5, 0, 1)
    assert expected_calibration_error(y, overconfident) > 0.1


def test_bootstrap_metrics_reproducible_and_brackets_point() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 500)
    y = (rng.uniform(0, 1, 500) < p).astype(int)
    first = bootstrap_metrics(y, p, n_boot=200)
    pd.testing.assert_frame_equal(first, bootstrap_metrics(y, p, n_boot=200))
    assert (first["ci_low"] <= first["value"]).all()
    assert (first["value"] <= first["ci_high"]).all()
    assert set(first.index) == set(compute_metrics(y, p))


def test_choose_variant_prefers_simplest_within_tolerance() -> None:
    table = pd.DataFrame(
        {
            "model": ["lightgbm", "logreg", "logreg"],
            "calibration": ["isotonic", "sigmoid", "none"],
            "log_loss": [0.520, 0.524, 0.527],
        }
    )
    assert choose_variant(table, tolerance=0.01) == Variant("logreg", "none")
    assert choose_variant(table, tolerance=0.001) == Variant("lightgbm", "isotonic")


def test_build_variant_wraps_only_when_calibrating() -> None:
    assert isinstance(build_variant(Variant("logreg", "none"), LOGREG), Pipeline)
    assert isinstance(build_variant(Variant("logreg", "sigmoid"), LOGREG), CalibratedClassifierCV)


def test_compare_calibration_smoke(sample) -> None:
    X, y = sample
    table, oof = compare_calibration({"logreg": LOGREG}, X, y, cv_folds=3)
    assert len(table) == 3
    assert {"ece", "log_loss", "roc_auc"} <= set(table.columns)
    assert all(p.shape == (len(X),) for p in oof.values())


def test_leakage_alarm() -> None:
    check_leakage(0.82)
    with pytest.raises(LeakageAlarmError):
        check_leakage(0.90)


def test_gold_rule_probabilities_on_raw_stats(sample) -> None:
    X, _ = sample
    p = gold_rule_probabilities(X)
    assert set(np.unique(p)) <= {0.0, 0.5, 1.0}


def test_predict_roundtrip_from_joblib(sample, tmp_path) -> None:
    X, y = sample
    model = build_variant(Variant("logreg", "sigmoid"), LOGREG).fit(X, y)
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    loaded = load_model(path)
    np.testing.assert_allclose(predict_proba(X, loaded), model.predict_proba(X)[:, 1])
    with pytest.raises(KeyError, match="ausentes"):
        predict_proba(X.drop(columns="blue_total_gold"), loaded)


def test_load_model_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="make evaluate"):
        load_model(tmp_path / "nope.joblib")
