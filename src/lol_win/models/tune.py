"""Tuning de hiperparâmetros com Optuna (Etapa 6A).

Cada trial roda a mesma CV estratificada da Etapa 5 e minimiza o **log loss**: o produto é uma
barra de probabilidade, então importa a qualidade da probabilidade, não só a ordenação (AUC).
"""

import argparse
import json
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from lol_win.analysis.eda import save_figure
from lol_win.config import (
    CV_FOLDS,
    GOLD_TRAIN,
    INPUT_COLUMNS,
    METRICS_DIR,
    RANDOM_STATE,
    TARGET,
)
from lol_win.features.build import FeatureSet
from lol_win.models.evaluate import SCORING, summarize_cv
from lol_win.models.train import ModelName, build_model, setup_mlflow

logger = logging.getLogger(__name__)

TUNED_MODELS: Final[tuple[ModelName, ...]] = ("logreg", "lightgbm")
DEFAULT_TRIALS: Final[dict[str, int]] = {"logreg": 50, "lightgbm": 100}
TUNABLE_FEATURE_SETS: Final[list[str]] = ["diffs", "diffs_core"]
TUNING_RESULTS_PATH: Final[Path] = METRICS_DIR / "tuning_results.json"


def suggest_params(trial: optuna.Trial, name: ModelName) -> tuple[FeatureSet, dict[str, Any]]:
    """Define o espaço de busca de cada modelo.

    Args:
        trial: Trial do Optuna.
        name: ``"logreg"`` ou ``"lightgbm"``.

    Returns:
        Tupla ``(feature_set, params do estimador final)``.

    Raises:
        ValueError: Se o modelo não tiver espaço de busca definido.
    """
    feature_set: FeatureSet = trial.suggest_categorical("feature_set", TUNABLE_FEATURE_SETS)
    if name == "logreg":
        return feature_set, {
            "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
            "solver": "saga",
            "max_iter": 5000,
        }
    if name == "lightgbm":
        return feature_set, {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 4, 31),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 300),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }
    raise ValueError(f"Sem espaço de busca para {name!r}. Opções: {TUNED_MODELS}")


def build_tuned_model(name: ModelName, feature_set: FeatureSet, params: dict[str, Any]) -> Pipeline:
    """Monta o pipeline do modelo com os hiperparâmetros dados.

    Args:
        name: Modelo.
        feature_set: Conjunto de features.
        params: Hiperparâmetros do estimador final.

    Returns:
        ``Pipeline`` não ajustado.
    """
    pipeline = build_model(name, feature_set)
    return pipeline.set_params(**{f"model__{key}": value for key, value in params.items()})


def tune_model(
    name: ModelName,
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int,
    cv_folds: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,
) -> optuna.Study:
    """Roda um estudo Optuna minimizando o log loss médio da CV.

    Cada trial guarda as demais métricas de CV em ``trial.user_attrs``.

    Args:
        name: Modelo.
        X: Stats brutas de treino.
        y: Alvo.
        n_trials: Número de trials.
        cv_folds: Número de folds.
        random_state: Semente do sampler e dos folds.

    Returns:
        Estudo concluído.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    def objective(trial: optuna.Trial) -> float:
        feature_set, params = suggest_params(trial, name)
        pipeline = build_tuned_model(name, feature_set, params)
        results = cross_validate(pipeline, X, y, cv=cv, scoring=SCORING, return_train_score=True)
        summary = summarize_cv(results)
        for key, value in summary.items():
            trial.set_user_attr(key, value)
        return summary["test_log_loss_mean"]

    study = optuna.create_study(
        study_name=f"tune_{name}",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials)
    return study


@dataclass(frozen=True)
class TuningResult:
    """Resumo do melhor trial de um estudo."""

    model: ModelName
    feature_set: FeatureSet
    params: dict[str, Any]
    metrics: dict[str, float]
    n_trials: int

    def to_dict(self) -> dict[str, Any]:
        """Serializa para JSON."""
        return {
            "model": self.model,
            "feature_set": self.feature_set,
            "params": self.params,
            "metrics": self.metrics,
            "n_trials": self.n_trials,
        }


def summarize_study(name: ModelName, study: optuna.Study) -> TuningResult:
    """Extrai configuração e métricas do melhor trial.

    Args:
        name: Modelo.
        study: Estudo concluído.

    Returns:
        Resultado do tuning.
    """
    best = study.best_trial
    feature_set, params = suggest_params(optuna.trial.FixedTrial(best.params), name)
    return TuningResult(
        model=name,
        feature_set=feature_set,
        params=params,
        metrics={key: float(value) for key, value in best.user_attrs.items()},
        n_trials=len(study.trials),
    )


def plot_optimization_history(studies: dict[str, optuna.Study]) -> plt.Figure:
    """Log loss de cada trial e o melhor acumulado, um painel por modelo.

    Args:
        studies: Estudos por nome de modelo.

    Returns:
        Figura matplotlib.
    """
    fig, axes = plt.subplots(1, len(studies), figsize=(6 * len(studies), 4), squeeze=False)
    for ax, (name, study) in zip(axes[0], studies.items(), strict=True):
        values = np.array([t.value for t in study.trials])
        ax.scatter(np.arange(len(values)), values, s=12, color="grey", alpha=0.6, label="trial")
        ax.plot(np.minimum.accumulate(values), color="#2B6CB0", label="melhor até aqui")
        low, high = np.nanmin(values), np.nanquantile(values, 0.9)
        ax.set_ylim(low - 0.002, high + 0.002)
        ax.set(title=f"{name}: log loss na CV", xlabel="trial", ylabel="log loss")
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def log_study_to_mlflow(result: TuningResult, study: optuna.Study) -> None:
    """Registra o melhor trial e o histórico completo (CSV) num run do MLflow.

    Args:
        result: Resumo do melhor trial.
        study: Estudo concluído.
    """
    with (
        tempfile.TemporaryDirectory() as tmp,
        mlflow.start_run(run_name=f"tune · {result.model}"),
    ):
        trials_csv = Path(tmp) / f"tuning_trials_{result.model}.csv"
        study.trials_dataframe().to_csv(trials_csv, index=False)
        mlflow.set_tags({"stage": "tune", "model": result.model, "objective": "log_loss"})
        mlflow.log_params(
            {"feature_set": result.feature_set, "n_trials": result.n_trials, **result.params}
        )
        mlflow.log_metrics(result.metrics)
        mlflow.log_artifact(str(trials_csv))


def main(
    trials: dict[str, int] | None = None,
    sample: int | None = None,
    output_path: Path = TUNING_RESULTS_PATH,
) -> dict[str, TuningResult]:
    """Roda o tuning de cada modelo, registra no MLflow e salva o JSON de resultados.

    Args:
        trials: Trials por modelo (padrão ``DEFAULT_TRIALS``).
        sample: Usa só ``sample`` linhas do treino (smoke test); nada é salvo em ``reports/``.
        output_path: Caminho do JSON de resultados.

    Returns:
        Resultados por modelo.
    """
    trials = trials or DEFAULT_TRIALS
    train = pd.read_parquet(GOLD_TRAIN)
    if sample is not None:
        train = train.sample(n=sample, random_state=RANDOM_STATE)
    X, y = train[list(INPUT_COLUMNS)], train[TARGET]

    setup_mlflow()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    studies: dict[str, optuna.Study] = {}
    results: dict[str, TuningResult] = {}
    for name in TUNED_MODELS:
        study = tune_model(name, X, y, n_trials=trials[name])
        result = summarize_study(name, study)
        log_study_to_mlflow(result, study)
        studies[name], results[name] = study, result
        logger.info(
            "%s: log loss %.4f · AUC %.4f · %s · %s",
            name,
            result.metrics["test_log_loss_mean"],
            result.metrics["test_roc_auc_mean"],
            result.feature_set,
            result.params,
        )

    if sample is None:
        output_path.write_text(
            json.dumps({k: v.to_dict() for k, v in results.items()}, indent=2, ensure_ascii=False)
            + "\n"
        )
        fig = plot_optimization_history(studies)
        save_figure(fig, "tune_optimization_history")
        plt.close(fig)
    return results


def load_tuning_results(path: Path = TUNING_RESULTS_PATH) -> dict[str, TuningResult]:
    """Lê os resultados salvos do tuning.

    Args:
        path: JSON gerado por ``main``.

    Returns:
        Resultados por modelo.
    """
    raw = json.loads(path.read_text())
    return {name: TuningResult(**payload) for name, payload in raw.items()}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=None, help="usa N linhas (smoke test)")
    parser.add_argument("--trials", type=int, default=None, help="trials por modelo (sobrescreve)")
    args = parser.parse_args()
    override = {name: args.trials for name in TUNED_MODELS} if args.trials else None
    main(trials=override, sample=args.sample)
