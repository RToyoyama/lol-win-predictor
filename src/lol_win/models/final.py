"""Etapa 6B: calibração, escolha final, avaliação única no teste, registro e exportação."""

import argparse
import json
import logging
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import lightgbm
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import sklearn
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.base import BaseEstimator
from sklearn.metrics import RocCurveDisplay

from lol_win.analysis.eda import save_figure
from lol_win.config import (
    AUC_LEAKAGE_ALARM,
    BACKGROUND_PATH,
    BACKGROUND_SIZE,
    FINAL_METRICS_PATH,
    GOLD_TEST,
    GOLD_TRAIN,
    INPUT_COLUMNS,
    METRICS_DIR,
    MLFLOW_MODEL_ALIAS,
    MLFLOW_MODEL_NAME,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    RANDOM_STATE,
    TARGET,
)
from lol_win.features.build import FeatureBuilder, WardsClipper, build_features
from lol_win.models.baselines import GoldRuleClassifier
from lol_win.models.calibrate import (
    Variant,
    build_variant,
    choose_variant,
    compare_calibration,
    expected_calibration_error,
    plot_calibration_curves,
)
from lol_win.models.evaluate import bootstrap_metrics, compute_metrics
from lol_win.models.train import setup_mlflow
from lol_win.models.tune import TuningResult, load_tuning_results

logger = logging.getLogger(__name__)


SKOPS_TRUSTED_TYPES = [
    f"{cls.__module__}.{cls.__qualname__}" for cls in (WardsClipper, FeatureBuilder)
]


class LeakageAlarmError(RuntimeError):
    """AUC no teste acima do teto realista: investigar vazamento antes de seguir."""


def check_leakage(test_auc: float, alarm: float = AUC_LEAKAGE_ALARM) -> None:
    """Interrompe o pipeline se a AUC de teste for suspeitamente alta.

    Args:
        test_auc: ROC-AUC no teste.
        alarm: Limite (CLAUDE.md: acima de 0,85, parar e investigar).

    Raises:
        LeakageAlarmError: Se ``test_auc > alarm``.
    """
    if test_auc > alarm:
        raise LeakageAlarmError(
            f"AUC de teste {test_auc:.3f} > {alarm}: investigar vazamento antes de seguir."
        )


def gold_rule_probabilities(stats: pd.DataFrame) -> np.ndarray:
    """Probabilidades da regra "quem tem mais gold vence" (contexto no teste).

    Args:
        stats: Stats brutas.

    Returns:
        Array ``(n,)`` com P(azul vence) ∈ {0, 0.5, 1}.
    """
    features = build_features(stats)
    rule = GoldRuleClassifier().fit(features, pd.Series([0, 1]))
    return rule.predict_proba(features)[:, 1]


def plot_test_roc(y_test: pd.Series, curves: dict[str, np.ndarray]) -> plt.Figure:
    """Curvas ROC no teste.

    Args:
        y_test: Alvo do teste.
        curves: Probabilidades por rótulo.

    Returns:
        Figura matplotlib.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    for label, p_blue in curves.items():
        RocCurveDisplay.from_predictions(y_test, p_blue, name=label, ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    ax.set(title="Curva ROC no teste (holdout)")
    fig.tight_layout()
    return fig


def build_metadata(
    variant: Variant,
    tuning: TuningResult,
    test_metrics: pd.DataFrame,
    n_train: int,
    run_id: str,
    registry_version: str,
) -> dict[str, Any]:
    """Metadados que acompanham o artefato exportado.

    Args:
        variant: Variante escolhida.
        tuning: Configuração de tuning do modelo.
        test_metrics: Métricas de teste com IC.
        n_train: Partidas de treino.
        run_id: Run do MLflow que registrou o modelo.
        registry_version: Versão no Model Registry.

    Returns:
        Dicionário serializável em JSON.
    """
    return {
        "model": variant.model,
        "calibration": variant.calibration,
        "variant": variant.name,
        "feature_set": tuning.feature_set,
        "params": tuning.params,
        "input_columns": list(INPUT_COLUMNS),
        "trained_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_train": n_train,
        "test_metrics": test_metrics.round(5).to_dict(orient="index"),
        "mlflow": {
            "run_id": run_id,
            "registered_model": MLFLOW_MODEL_NAME,
            "version": registry_version,
            "alias": MLFLOW_MODEL_ALIAS,
        },
        "versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "lightgbm": lightgbm.__version__,
        },
    }


def register_model(
    model: BaseEstimator,
    X_train: pd.DataFrame,
    variant: Variant,
    tuning: TuningResult,
    test_metrics: pd.DataFrame,
    artifacts: list[Path],
) -> tuple[str, str]:
    """Loga o modelo final no MLflow, registra no Model Registry e aplica o alias.

    Args:
        model: Estimador final ajustado.
        X_train: Stats de treino (para assinatura e exemplo de entrada).
        variant: Variante escolhida.
        tuning: Configuração de tuning.
        test_metrics: Métricas de teste com IC.
        artifacts: Arquivos extras (figuras, tabelas) a anexar ao run.

    Returns:
        Tupla ``(run_id, versão registrada)``.
    """
    example = X_train.head(3)
    signature = infer_signature(example, model.predict_proba(example)[:, 1])
    with mlflow.start_run(run_name=f"final · {variant.name}") as run:
        mlflow.set_tags(
            {"stage": "final", "model": variant.model, "calibration": variant.calibration}
        )
        mlflow.log_params({"feature_set": tuning.feature_set, **tuning.params})
        for metric, row in test_metrics.iterrows():
            mlflow.log_metrics(
                {
                    f"test_{metric}": row["value"],
                    f"test_{metric}_ci_low": row["ci_low"],
                    f"test_{metric}_ci_high": row["ci_high"],
                }
            )
        for path in artifacts:
            mlflow.log_artifact(str(path))
        info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=example,
            registered_model_name=MLFLOW_MODEL_NAME,
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )
    version = str(info.registered_model_version)
    MlflowClient().set_registered_model_alias(MLFLOW_MODEL_NAME, MLFLOW_MODEL_ALIAS, version)
    return run.info.run_id, version


def export_background(
    train: pd.DataFrame, path: Path = BACKGROUND_PATH, n: int = BACKGROUND_SIZE
) -> pd.DataFrame:
    """Exporta uma amostra do treino (stats brutas) para acompanhar o modelo.

    Serve de referência para o SHAP e de fonte de "partidas reais" no app, deixando o
    artefato autocontido (o container não tem ``data/``).

    Args:
        train: Camada gold de treino.
        path: Parquet de saída.
        n: Tamanho da amostra.

    Returns:
        Amostra exportada.
    """
    sample = train.sample(n=min(n, len(train)), random_state=RANDOM_STATE)
    sample = sample[list(INPUT_COLUMNS)].reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(path, index=False)
    return sample


def main() -> dict[str, Any]:
    """Executa a Etapa 6B de ponta a ponta.

    Returns:
        Conteúdo de ``reports/metrics/final.json``.
    """
    train, test = pd.read_parquet(GOLD_TRAIN), pd.read_parquet(GOLD_TEST)
    X_train, y_train = train[list(INPUT_COLUMNS)], train[TARGET]
    X_test, y_test = test[list(INPUT_COLUMNS)], test[TARGET]
    tuning = load_tuning_results()

    calibration_table, oof = compare_calibration(tuning, X_train, y_train)
    tolerance = max(r.metrics["test_log_loss_std"] for r in tuning.values())
    variant = choose_variant(calibration_table, tolerance)
    logger.info("calibração (OOF no treino):\n%s", calibration_table.round(4).to_string())
    logger.info("escolhido: %s (tolerância de log loss = %.4f)", variant.name, tolerance)

    calibration_csv = METRICS_DIR / "calibration_comparison.csv"
    calibration_table.to_csv(calibration_csv, index=False, float_format="%.5f")
    fig = plot_calibration_curves(y_train.to_numpy(), oof)
    calibration_png = save_figure(fig, "final_calibration_curve")
    plt.close(fig)

    model = build_variant(variant, tuning[variant.model]).fit(X_train, y_train)
    p_test = model.predict_proba(X_test)[:, 1]
    test_metrics = bootstrap_metrics(y_test.to_numpy(), p_test)
    check_leakage(test_metrics.loc["roc_auc", "value"])

    p_gold_rule = gold_rule_probabilities(X_test)
    fig = plot_test_roc(y_test, {variant.name: p_test, "regra do gold": p_gold_rule})
    roc_png = save_figure(fig, "final_roc_curve_test")
    plt.close(fig)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    export_background(train)
    setup_mlflow()
    run_id, version = register_model(
        model,
        X_train,
        variant,
        tuning[variant.model],
        test_metrics,
        [calibration_csv, calibration_png, roc_png],
    )

    cv_metrics = tuning[variant.model].metrics
    final = {
        "chosen": variant.name,
        "selection_rule": (
            "variante mais simples com log loss OOF a até 1 desvio-padrão da CV do melhor; "
            "LR < LightGBM; sem calibração < sigmoid < isotônica"
        ),
        "log_loss_tolerance": tolerance,
        "test": test_metrics.round(5).to_dict(orient="index"),
        "test_ece": expected_calibration_error(y_test.to_numpy(), p_test),
        "cv_train": {
            m: cv_metrics[f"test_{m}_mean"]
            for m in ("roc_auc", "log_loss", "brier", "accuracy", "f1")
        },
        "gold_rule_test": compute_metrics(y_test.to_numpy(), p_gold_rule),
        "n_train": len(train),
        "n_test": len(test),
        "mlflow": {"run_id": run_id, "registered_model": MLFLOW_MODEL_NAME, "version": version},
    }
    FINAL_METRICS_PATH.write_text(json.dumps(final, indent=2, ensure_ascii=False) + "\n")
    metadata = build_metadata(
        variant, tuning[variant.model], test_metrics, len(train), run_id, version
    )
    MODEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    logger.info("teste:\n%s", test_metrics.round(4).to_string())
    return final


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--background-only",
        action="store_true",
        help="só exporta models/background.parquet (não reavalia nem registra o modelo)",
    )
    if parser.parse_args().background_only:
        export_background(pd.read_parquet(GOLD_TRAIN))
    else:
        main()
