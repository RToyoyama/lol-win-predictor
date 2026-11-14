"""Registro de modelos: monta pipelines a partir de nome + configuração.

Não depende de MLflow nem de Optuna, para poder ser usado no container (export do modelo).
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, get_args

from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from lol_win.config import RANDOM_STATE, TUNING_RESULTS_PATH, WARDS_CAP_QUANTILE
from lol_win.features.build import FeatureSet, make_feature_pipeline
from lol_win.models.baselines import GoldRuleClassifier

ModelName = Literal["dummy", "gold_rule", "logreg", "random_forest", "lightgbm"]
MODEL_NAMES: Final[tuple[str, ...]] = get_args(ModelName)
BASELINES: Final[frozenset[str]] = frozenset({"dummy", "gold_rule"})
CANDIDATES: Final[tuple[ModelName, ...]] = ("logreg", "random_forest", "lightgbm")


def _model_steps(name: ModelName) -> list[tuple[str, BaseEstimator]]:
    """Etapas que vêm depois das features, por modelo (hiperparâmetros sem tuning)."""
    match name:
        case "dummy":
            return [("model", DummyClassifier(strategy="prior"))]
        case "gold_rule":
            return [("model", GoldRuleClassifier())]
        case "logreg":
            return [
                ("scale", StandardScaler()),
                ("model", LogisticRegression(C=1.0, max_iter=2000, random_state=RANDOM_STATE)),
            ]
        case "random_forest":
            return [
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=500,
                        min_samples_leaf=20,
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                )
            ]
        case "lightgbm":
            return [
                (
                    "model",
                    LGBMClassifier(
                        n_estimators=400,
                        learning_rate=0.03,
                        num_leaves=15,
                        min_child_samples=40,
                        subsample=0.8,
                        subsample_freq=1,
                        colsample_bytree=0.8,
                        random_state=RANDOM_STATE,
                        verbose=-1,
                    ),
                )
            ]
    raise ValueError(f"Modelo desconhecido: {name!r}. Opções: {MODEL_NAMES}")


def build_model(
    name: ModelName,
    feature_set: FeatureSet = "diffs",
    wards_quantile: float | None = WARDS_CAP_QUANTILE,
) -> Pipeline:
    """Monta o pipeline completo: stats brutas → features → modelo.

    Args:
        name: Modelo (baseline ou candidato).
        feature_set: Conjunto de features.
        wards_quantile: Quantil do teto de wards (``None`` desliga).

    Returns:
        ``Pipeline`` não ajustado.
    """
    features = make_feature_pipeline(feature_set, wards_quantile)
    return Pipeline([*features.steps, *_model_steps(name)])


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


def load_tuning_results(path: Path = TUNING_RESULTS_PATH) -> dict[str, TuningResult]:
    """Lê os resultados salvos do tuning.

    Args:
        path: JSON gerado por ``lol_win.models.tune``.

    Returns:
        Resultados por modelo.
    """
    raw = json.loads(path.read_text())
    return {name: TuningResult(**payload) for name, payload in raw.items()}
