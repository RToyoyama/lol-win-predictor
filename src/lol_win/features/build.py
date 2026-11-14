"""Construção de features de modelagem (fonte única para treino, API e app)."""

from collections.abc import Sequence
from typing import Final

import pandas as pd

DIFF_STATS: Final[tuple[str, ...]] = (
    "kills",
    "assists",
    "wards_placed",
    "wards_destroyed",
    "dragons",
    "heralds",
    "towers_destroyed",
    "total_minions_killed",
    "total_jungle_minions_killed",
    "total_gold",
    "total_experience",
    "avg_level",
)


def side_diffs(df: pd.DataFrame, stats: Sequence[str] = DIFF_STATS) -> pd.DataFrame:
    """Calcula diferenças azul − vermelho para cada estatística.

    Args:
        df: DataFrame com colunas ``blue_<stat>`` e ``red_<stat>``.
        stats: Estatísticas a diferenciar.

    Returns:
        DataFrame com colunas ``<stat>_diff``, mesmo índice de ``df``.
    """
    return pd.DataFrame({f"{stat}_diff": df[f"blue_{stat}"] - df[f"red_{stat}"] for stat in stats})
