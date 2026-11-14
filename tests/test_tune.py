import json

import optuna
import pandas as pd
import pytest

from lol_win.config import INPUT_COLUMNS, RANDOM_STATE, TARGET
from lol_win.data.clean import to_silver
from lol_win.data.ingest import read_raw, to_bronze
from lol_win.models.registry import build_tuned_model, load_tuning_results
from lol_win.models.tune import TUNED_MODELS, suggest_params, summarize_study, tune_model


@pytest.fixture(scope="module")
def sample() -> tuple[pd.DataFrame, pd.Series]:
    silver = to_silver(to_bronze(read_raw())).sample(n=400, random_state=RANDOM_STATE)
    return silver[list(INPUT_COLUMNS)], silver[TARGET]


@pytest.fixture(autouse=True)
def quiet_optuna() -> None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)


@pytest.mark.parametrize("name", TUNED_MODELS)
def test_search_space_builds_valid_pipeline(name, sample) -> None:
    X, y = sample
    trial = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0)).ask()
    feature_set, params = suggest_params(trial, name)
    proba = build_tuned_model(name, feature_set, params).fit(X, y).predict_proba(X)
    assert proba.shape == (len(X), 2)


def test_suggest_params_rejects_untuned_model() -> None:
    trial = optuna.create_study().ask()
    with pytest.raises(ValueError, match="Sem espaço"):
        suggest_params(trial, "random_forest")


def test_tune_model_is_reproducible_and_summarizable(sample, tmp_path) -> None:
    X, y = sample
    first = tune_model("logreg", X, y, n_trials=3, cv_folds=3)
    second = tune_model("logreg", X, y, n_trials=3, cv_folds=3)
    assert first.best_params == second.best_params

    result = summarize_study("logreg", first)
    assert result.n_trials == 3
    assert result.params["solver"] == "saga"
    assert result.metrics["test_log_loss_mean"] == pytest.approx(first.best_value)
    assert "test_roc_auc_mean" in result.metrics

    path = tmp_path / "tuning.json"
    path.write_text(json.dumps({"logreg": result.to_dict()}))
    assert load_tuning_results(path)["logreg"] == result
