"""Contratos de dados (Pandera) das camadas do pipeline."""

from typing import Final

import numpy as np
import pandas as pd
import pandera.pandas as pa

from lol_win.config import SIDES

MAX_TOWERS_PER_SIDE: Final[int] = 11
MAX_CHAMPION_LEVEL: Final[int] = 18
PLAYERS_PER_TEAM: Final[int] = 5
MINUTES: Final[int] = 10
FLOAT_ATOL: Final[float] = 1e-6

_COUNT = pa.Check.ge(0)


def _side_columns(side: str) -> dict[str, pa.Column]:
    """Declara as colunas de um time com tipo e range físico do jogo.

    Args:
        side: Prefixo do time (``"blue"`` ou ``"red"``).

    Returns:
        Mapeamento nome da coluna -> definição Pandera.
    """
    int_counts = [
        "wards_placed",
        "wards_destroyed",
        "kills",
        "deaths",
        "assists",
        "total_gold",
        "total_experience",
        "total_minions_killed",
        "total_jungle_minions_killed",
    ]
    columns = {f"{side}_{name}": pa.Column(int, _COUNT) for name in int_counts}
    columns |= {
        f"{side}_first_blood": pa.Column(int, pa.Check.isin([0, 1])),
        f"{side}_elite_monsters": pa.Column(int, pa.Check.in_range(0, 2)),
        f"{side}_dragons": pa.Column(int, pa.Check.in_range(0, 1)),
        f"{side}_heralds": pa.Column(int, pa.Check.in_range(0, 1)),
        f"{side}_towers_destroyed": pa.Column(int, pa.Check.in_range(0, MAX_TOWERS_PER_SIDE)),
        f"{side}_avg_level": pa.Column(float, pa.Check.in_range(1, MAX_CHAMPION_LEVEL)),
        f"{side}_gold_diff": pa.Column(int),
        f"{side}_experience_diff": pa.Column(int),
        f"{side}_cs_per_min": pa.Column(float, _COUNT),
        f"{side}_gold_per_min": pa.Column(float, _COUNT),
    }
    return columns


def _is_integer(values: pd.Series) -> pd.Series:
    return pd.Series(np.isclose(values, values.round(), atol=FLOAT_ATOL), index=values.index)


def _close(left: pd.Series, right: pd.Series) -> pd.Series:
    return pd.Series(np.isclose(left, right, atol=FLOAT_ATOL), index=left.index)


def _consistency_checks() -> list[pa.Check]:
    """Regras determinísticas entre colunas, válidas em 100% das partidas aos 10 min.

    Returns:
        Lista de checks de DataFrame (avaliados linha a linha).
    """
    checks = [
        pa.Check(lambda d: d["blue_kills"] == d["red_deaths"], name="blue_kills_eq_red_deaths"),
        pa.Check(lambda d: d["red_kills"] == d["blue_deaths"], name="red_kills_eq_blue_deaths"),
        pa.Check(
            lambda d: d["blue_gold_diff"] == d["blue_total_gold"] - d["red_total_gold"],
            name="gold_diff_eq_total_gold_delta",
        ),
        pa.Check(lambda d: d["red_gold_diff"] == -d["blue_gold_diff"], name="gold_diff_symmetric"),
        pa.Check(
            lambda d: (
                d["blue_experience_diff"] == d["blue_total_experience"] - d["red_total_experience"]
            ),
            name="xp_diff_eq_total_xp_delta",
        ),
        pa.Check(
            lambda d: d["red_experience_diff"] == -d["blue_experience_diff"],
            name="xp_diff_symmetric",
        ),
        pa.Check(
            lambda d: d["blue_first_blood"] + d["red_first_blood"] == 1,
            name="exactly_one_first_blood",
        ),
        pa.Check(
            lambda d: d["blue_dragons"] + d["red_dragons"] <= 1,
            name="at_most_one_dragon_total",
        ),
    ]
    for side in SIDES:
        checks += [
            pa.Check(
                lambda d, s=side: d[f"{s}_elite_monsters"] == d[f"{s}_dragons"] + d[f"{s}_heralds"],
                name=f"{side}_elite_eq_dragons_plus_heralds",
            ),
            pa.Check(
                lambda d, s=side: _close(d[f"{s}_gold_per_min"], d[f"{s}_total_gold"] / MINUTES),
                name=f"{side}_gold_per_min_eq_total_over_10",
            ),
            pa.Check(
                lambda d, s=side: _close(
                    d[f"{s}_cs_per_min"], d[f"{s}_total_minions_killed"] / MINUTES
                ),
                name=f"{side}_cs_per_min_eq_minions_over_10",
            ),
            pa.Check(
                lambda d, s=side: _is_integer(d[f"{s}_avg_level"] * PLAYERS_PER_TEAM),
                name=f"{side}_avg_level_is_mean_of_5",
            ),
        ]
    return checks


def build_bronze_schema() -> pa.DataFrameSchema:
    """Monta o contrato da camada bronze.

    Returns:
        Schema estrito (sem colunas extras/faltantes, sem coerção de tipos).
    """
    columns: dict[str, pa.Column] = {
        "game_id": pa.Column(int, unique=True),
        "blue_wins": pa.Column(int, pa.Check.isin([0, 1])),
    }
    for side in SIDES:
        columns |= _side_columns(side)
    return pa.DataFrameSchema(
        columns=columns,
        checks=_consistency_checks(),
        strict=True,
        coerce=False,
        name="bronze_matches",
    )


BRONZE_SCHEMA: Final[pa.DataFrameSchema] = build_bronze_schema()

REDUNDANT_COLUMNS: Final[tuple[str, ...]] = (
    *(
        f"{side}_{name}"
        for side in SIDES
        for name in (
            "deaths",
            "elite_monsters",
            "gold_diff",
            "experience_diff",
            "cs_per_min",
            "gold_per_min",
        )
    ),
    "red_first_blood",
)


def build_silver_schema() -> pa.DataFrameSchema:
    """Monta o contrato da camada silver: bronze sem as colunas redundantes.

    Mantém tipos e ranges da bronze e as regras que ainda podem ser verificadas sem as colunas
    removidas.

    Returns:
        Schema estrito da camada silver.
    """
    columns = {
        name: column
        for name, column in BRONZE_SCHEMA.columns.items()
        if name not in REDUNDANT_COLUMNS
    }
    checks = [
        pa.Check(
            lambda d: d["blue_dragons"] + d["red_dragons"] <= 1,
            name="at_most_one_dragon_total",
        ),
        *(
            pa.Check(
                lambda d, s=side: _is_integer(d[f"{s}_avg_level"] * PLAYERS_PER_TEAM),
                name=f"{side}_avg_level_is_mean_of_5",
            )
            for side in SIDES
        ),
    ]
    return pa.DataFrameSchema(
        columns=columns, checks=checks, strict=True, coerce=False, name="silver_matches"
    )


SILVER_SCHEMA: Final[pa.DataFrameSchema] = build_silver_schema()
