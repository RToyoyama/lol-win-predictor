"""Reconstrói e exporta o modelo campeão a partir da configuração versionada.

Usado no build do Docker: reajusta no treino a variante escolhida na Etapa 6
(``reports/metrics/final.json`` + ``tuning_results.json``), sem tocar no teste nem no MLflow.
"""

import argparse
import json
import logging
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator

from lol_win.config import (
    BACKGROUND_SIZE,
    FINAL_METRICS_PATH,
    GOLD_TRAIN,
    INPUT_COLUMNS,
    MODEL_PATH,
    RANDOM_STATE,
    TARGET,
    TUNING_RESULTS_PATH,
)
from lol_win.models.calibrate import Variant, build_variant
from lol_win.models.registry import load_tuning_results

logger = logging.getLogger(__name__)


def export_background(train: pd.DataFrame, path: Path, n: int = BACKGROUND_SIZE) -> pd.DataFrame:
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


def rebuild_champion(
    train: pd.DataFrame,
    final_path: Path = FINAL_METRICS_PATH,
    tuning_path: Path = TUNING_RESULTS_PATH,
) -> tuple[BaseEstimator, dict[str, Any]]:
    """Reajusta no treino a variante campeã e monta os metadados.

    Args:
        train: Camada gold de treino.
        final_path: ``final.json`` da avaliação final (variante escolhida e métricas de teste).
        tuning_path: ``tuning_results.json`` (hiperparâmetros).

    Returns:
        Tupla ``(modelo ajustado, metadados)``.
    """
    final = json.loads(final_path.read_text())
    variant = Variant.from_name(final["chosen"])
    tuning = load_tuning_results(tuning_path)[variant.model]
    model = build_variant(variant, tuning).fit(train[list(INPUT_COLUMNS)], train[TARGET])
    metadata = {
        "model": variant.model,
        "calibration": variant.calibration,
        "variant": variant.name,
        "feature_set": tuning.feature_set,
        "params": tuning.params,
        "input_columns": list(INPUT_COLUMNS),
        "trained_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_train": len(train),
        "test_metrics": final["test"],
        "mlflow": final["mlflow"],
        "source": "rebuilt by lol_win.models.export from versioned config",
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__},
    }
    return model, metadata


def main(output_dir: Path = MODEL_PATH.parent, train_path: Path = GOLD_TRAIN) -> dict[str, Any]:
    """Reconstrói o campeão e grava modelo, metadados e amostra de referência.

    Args:
        output_dir: Pasta de saída (``model.joblib``, ``metadata.json``, ``background.parquet``).
        train_path: Parquet de treino.

    Returns:
        Metadados gravados.
    """
    train = pd.read_parquet(train_path)
    model, metadata = rebuild_champion(train)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "model.joblib")
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"
    )
    export_background(train, output_dir / "background.parquet")
    logger.info("campeão %s exportado em %s", metadata["variant"], output_dir)
    return metadata


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=MODEL_PATH.parent)
    main(output_dir=parser.parse_args().output_dir)
