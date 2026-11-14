"""Carregamento do modelo final e predição (usado pela API e pelo app)."""

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from lol_win.config import INPUT_COLUMNS, MODEL_METADATA_PATH, MODEL_PATH


@lru_cache(maxsize=4)
def load_model(path: Path = MODEL_PATH) -> BaseEstimator:
    """Carrega o modelo exportado (uma vez por caminho; chamadas seguintes usam cache).

    Args:
        path: Arquivo ``.joblib`` gerado na avaliação final.

    Returns:
        Estimador ajustado que recebe stats brutas (``INPUT_COLUMNS``).

    Raises:
        FileNotFoundError: Se o modelo ainda não foi exportado (rode ``make evaluate``).
    """
    if not path.exists():
        raise FileNotFoundError(f"Modelo não encontrado em {path}. Rode `make evaluate`.")
    return joblib.load(path)


def load_metadata(path: Path = MODEL_METADATA_PATH) -> dict[str, Any]:
    """Lê os metadados do modelo exportado.

    Args:
        path: JSON de metadados.

    Returns:
        Dicionário com versão, features, parâmetros e métricas de teste.
    """
    return json.loads(path.read_text())


def predict_proba(stats: pd.DataFrame, model: BaseEstimator | None = None) -> np.ndarray:
    """Probabilidade de vitória do lado azul a partir das stats brutas aos 10 min.

    Args:
        stats: Uma linha por partida com todas as ``INPUT_COLUMNS``.
        model: Estimador ajustado; ``None`` carrega o modelo exportado.

    Returns:
        Array ``(n,)`` com P(azul vence).

    Raises:
        KeyError: Se faltar alguma coluna de entrada.
    """
    missing = set(INPUT_COLUMNS) - set(stats.columns)
    if missing:
        raise KeyError(f"Colunas de entrada ausentes: {sorted(missing)}")
    model = model or load_model()
    return model.predict_proba(stats[list(INPUT_COLUMNS)])[:, 1]
