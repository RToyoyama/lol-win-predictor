"""Treino com validação cruzada e registro no MLflow (Etapa 5)."""

import argparse
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

from lol_win.analysis.eda import save_figure
from lol_win.config import (
    CV_FOLDS,
    GOLD_TRAIN,
    INPUT_COLUMNS,
    METRICS_DIR,
    MLFLOW_ARTIFACTS_DIR,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    RANDOM_STATE,
    TARGET,
    WARDS_CAP_QUANTILE,
)
from lol_win.features.build import FeatureSet
from lol_win.models.evaluate import SCORING, plot_cv_comparison, summarize_cv
from lol_win.models.registry import BASELINES, CANDIDATES, ModelName, build_model

logger = logging.getLogger(__name__)

GOLD_ONLY_BENCHMARK_AUC: Final[float] = 0.803  # LR só com total_gold_diff, CV no treino (Etapa 4)


@dataclass(frozen=True)
class ExperimentConfig:
    """Uma configuração da grade de experimentos."""

    model: ModelName
    feature_set: FeatureSet = "diffs"
    wards_quantile: float | None = WARDS_CAP_QUANTILE

    @property
    def run_name(self) -> str:
        """Nome legível do run (ex.: ``logreg · diffs · cap p99``)."""
        if self.model in BASELINES:
            return self.model
        cap = "sem cap" if self.wards_quantile is None else f"cap p{self.wards_quantile * 100:g}"
        return f"{self.model} · {self.feature_set} · {cap}"


def default_grid() -> list[ExperimentConfig]:
    """Grade da Etapa 5: baselines, modelos × conjuntos de features e ablação do cap.

    Returns:
        Lista de configurações.
    """
    grid = [ExperimentConfig("dummy"), ExperimentConfig("gold_rule")]
    for model in CANDIDATES:
        grid += [ExperimentConfig(model, fs) for fs in ("diffs", "diffs_abs", "diffs_core")]
        grid.append(ExperimentConfig(model, "diffs", wards_quantile=None))
    return grid


def setup_mlflow(
    tracking_uri: str = MLFLOW_TRACKING_URI,
    experiment: str = MLFLOW_EXPERIMENT,
    artifacts_dir: Path = MLFLOW_ARTIFACTS_DIR,
) -> str:
    """Aponta o MLflow para o backend local e garante que o experimento exista.

    Args:
        tracking_uri: URI do backend (SQLite).
        experiment: Nome do experimento.
        artifacts_dir: Pasta onde os artefatos dos runs serão gravados.

    Returns:
        ID do experimento.
    """
    mlflow.set_tracking_uri(tracking_uri)
    existing = mlflow.get_experiment_by_name(experiment)
    experiment_id = (
        existing.experiment_id
        if existing
        else mlflow.create_experiment(experiment, artifact_location=artifacts_dir.as_uri())
    )
    mlflow.set_experiment(experiment_id=experiment_id)
    return experiment_id


def run_cv_experiment(
    config: ExperimentConfig,
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int = CV_FOLDS,
) -> dict[str, float]:
    """Roda CV estratificada de uma configuração e registra tudo num run (aninhado) do MLflow.

    Args:
        config: Configuração do experimento.
        X: Stats brutas de treino (``INPUT_COLUMNS``).
        y: Alvo.
        cv_folds: Número de folds.

    Returns:
        Métricas resumidas (média e desvio, treino e validação).
    """
    pipeline = build_model(config.model, config.feature_set, config.wards_quantile)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    results = cross_validate(pipeline, X, y, cv=cv, scoring=SCORING, return_train_score=True)
    summary = summarize_cv(results)

    with mlflow.start_run(run_name=config.run_name, nested=True):
        mlflow.set_tags({"stage": "cv", "model": config.model, "feature_set": config.feature_set})
        model_params = pipeline.named_steps["model"].get_params()
        mlflow.log_params(
            {
                "model": config.model,
                "feature_set": config.feature_set,
                "wards_quantile": config.wards_quantile,
                "cv_folds": cv_folds,
                "n_train": len(X),
                **{f"model__{k}": v for k, v in model_params.items()},
            }
        )
        mlflow.log_metrics(summary)
    logger.info(
        "%-40s AUC %.4f ± %.4f",
        config.run_name,
        summary["test_roc_auc_mean"],
        summary["test_roc_auc_std"],
    )
    return summary


def run_grid(
    grid: Sequence[ExperimentConfig],
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int = CV_FOLDS,
    run_name: str = "etapa5_cv_grid",
) -> pd.DataFrame:
    """Executa a grade dentro de um run pai do MLflow.

    Args:
        grid: Configurações.
        X: Stats brutas de treino.
        y: Alvo.
        cv_folds: Número de folds.
        run_name: Nome do run pai.

    Returns:
        Tabela com uma linha por configuração, ordenada por AUC decrescente.
    """
    rows = []
    with mlflow.start_run(run_name=run_name):
        for config in grid:
            summary = run_cv_experiment(config, X, y, cv_folds)
            rows.append(
                {
                    "run_name": config.run_name,
                    "model": config.model,
                    "feature_set": config.feature_set,
                    "wards_quantile": config.wards_quantile,
                    **summary,
                }
            )
    table = pd.DataFrame(rows)
    table["auc_gap_train_test"] = table["train_roc_auc_mean"] - table["test_roc_auc_mean"]
    return table.sort_values("test_roc_auc_mean", ascending=False).reset_index(drop=True)


def main(
    sample: int | None = None, cv_folds: int = CV_FOLDS, output_dir: Path = METRICS_DIR
) -> pd.DataFrame:
    """Roda a grade completa no treino e salva tabela + figura comparativa.

    Args:
        sample: Usa só ``sample`` linhas do treino (smoke test). ``None`` = treino completo.
        cv_folds: Número de folds.
        output_dir: Pasta da tabela CSV.

    Returns:
        Tabela comparativa.
    """
    train = pd.read_parquet(GOLD_TRAIN)
    if sample is not None:
        train = train.sample(n=sample, random_state=RANDOM_STATE)
    X, y = train[list(INPUT_COLUMNS)], train[TARGET]

    setup_mlflow()
    table = run_grid(default_grid(), X, y, cv_folds)

    if sample is None:
        output_dir.mkdir(parents=True, exist_ok=True)
        table.to_csv(output_dir / "cv_comparison.csv", index=False, float_format="%.5f")
        baselines = table.set_index("model")
        candidates = table[~table["model"].isin(BASELINES)]
        references = {
            "roc_auc": {
                "regra do gold": baselines.loc["gold_rule", "test_roc_auc_mean"],
                "LR só gold_diff (benchmark)": GOLD_ONLY_BENCHMARK_AUC,
            },
            "log_loss": {"dummy": baselines.loc["dummy", "test_log_loss_mean"]},
        }
        fig = plot_cv_comparison(candidates, references)
        save_figure(fig, "model_cv_comparison")
        plt.close(fig)
    return table


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=None, help="usa N linhas (smoke test)")
    parser.add_argument("--folds", type=int, default=CV_FOLDS)
    args = parser.parse_args()
    main(sample=args.sample, cv_folds=args.folds)
