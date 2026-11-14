"""Carregamento do modelo final e predição (usado pela API e pelo app)."""

import json
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any, Final

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from lol_win.config import INPUT_COLUMNS, MODEL_METADATA_PATH, MODEL_PATH

BAND_EDGES: Final[tuple[float, ...]] = (0.0, 0.15, 0.35, 0.65, 0.85, 1.0)
BAND_LABELS: Final[tuple[str, ...]] = ("<15%", "15–35%", "35–65%", "65–85%", ">85%")
BAND_DESCRIPTIONS: Final[tuple[str, ...]] = (
    "vermelho muito favorito",
    "vermelho favorito",
    "jogo em aberto",
    "azul favorito",
    "azul muito favorito",
)


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


def probability_band(p_blue: float, edges: Sequence[float] = BAND_EDGES) -> tuple[str, str]:
    """Faixa de decisão de uma probabilidade (mesmas faixas de ``decision_bands``).

    As faixas são fechadas à direita, como em ``pd.cut``: 0,15 cai em "<15%" e 0,85 em "65–85%".

    Args:
        p_blue: Probabilidade de vitória azul em [0, 1].
        edges: Bordas das faixas.

    Returns:
        Tupla ``(rótulo, descrição)``, ex.: ``(">85%", "azul muito favorito")``.

    Raises:
        ValueError: Se ``p_blue`` estiver fora de [0, 1].
    """
    if not 0.0 <= p_blue <= 1.0:
        raise ValueError(f"Probabilidade fora de [0, 1]: {p_blue}")
    index = int(np.searchsorted(np.asarray(edges[1:-1]), p_blue, side="left"))
    return BAND_LABELS[index], BAND_DESCRIPTIONS[index]
