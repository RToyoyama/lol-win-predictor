"""Lógica pura do simulador (sem Streamlit): cenários, ações rápidas e validação."""

import copy
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import pandas as pd
from annotated_types import Ge, Le
from pydantic import ValidationError

from lol_win.api.schemas import MatchStats, TeamStats
from lol_win.config import SIDES, TEAM_STATS

Scenario = dict[str, Any]
SIDE_LABELS: Final[dict[str, str]] = {"blue": "azul", "red": "vermelho"}
INTEGER_STATS: Final[frozenset[str]] = frozenset(TEAM_STATS) - {"avg_level"}
LEVEL_STEP: Final[float] = 0.2


class ActionError(ValueError):
    """Ação rápida que levaria a um cenário inválido."""


def field_bounds() -> dict[str, tuple[float, float]]:
    """Limites (mínimo, máximo) de cada estatística, lidos do schema da API.

    Returns:
        Mapeamento ``stat -> (ge, le)``.
    """
    bounds = {}
    for name, field in TeamStats.model_fields.items():
        low = next(m.ge for m in field.metadata if isinstance(m, Ge))
        high = next(m.le for m in field.metadata if isinstance(m, Le))
        bounds[name] = (float(low), float(high))
    return bounds


def _round_level(value: float) -> float:
    return round(round(value / LEVEL_STEP) * LEVEL_STEP, 1)


def scenario_from_row(row: pd.Series) -> Scenario:
    """Converte uma linha com ``INPUT_COLUMNS`` em cenário.

    Args:
        row: Linha de stats brutas (ex.: do ``background.parquet``).

    Returns:
        Cenário ``{"blue": {...}, "red": {...}, "first_blood": "blue" | "red"}``.
    """
    scenario: Scenario = {"first_blood": "blue" if int(row["blue_first_blood"]) == 1 else "red"}
    for side in SIDES:
        team = {stat: row[f"{side}_{stat}"] for stat in TEAM_STATS}
        scenario[side] = {
            stat: int(value) if stat in INTEGER_STATS else _round_level(float(value))
            for stat, value in team.items()
        }
    return scenario


def default_scenario(background: pd.DataFrame) -> Scenario:
    """Partida "parelha": os dois times com a mediana de cada estatística.

    Objetivos raros (dragão, arauto, torre) começam em zero; o first blood fica com o azul.

    Args:
        background: Amostra de stats brutas.

    Returns:
        Cenário simétrico (diferenças zero, probabilidade perto de 50%).
    """
    team = {}
    for stat in TEAM_STATS:
        pooled = pd.concat([background[f"{side}_{stat}"] for side in SIDES])
        median = float(pooled.median())
        team[stat] = int(round(median)) if stat in INTEGER_STATS else _round_level(median)
    team |= {"dragons": 0, "heralds": 0, "towers_destroyed": 0}
    team["kills"] = max(team["kills"], 1)
    return {"blue": dict(team), "red": dict(team), "first_blood": "blue"}


def random_scenario(background: pd.DataFrame, rng: np.random.Generator) -> Scenario:
    """Sorteia uma partida real da amostra de referência.

    Args:
        background: Amostra de stats brutas.
        rng: Gerador de números aleatórios.

    Returns:
        Cenário da partida sorteada.
    """
    return scenario_from_row(background.iloc[int(rng.integers(len(background)))])


def to_match(scenario: Scenario) -> MatchStats:
    """Valida o cenário com o mesmo contrato da API.

    Args:
        scenario: Cenário.

    Returns:
        ``MatchStats`` validado.

    Raises:
        pydantic.ValidationError: Se violar alguma regra do jogo.
    """
    return MatchStats.model_validate(scenario)


def validation_messages(scenario: Scenario) -> list[str]:
    """Mensagens legíveis das regras violadas (lista vazia se válido).

    Args:
        scenario: Cenário.

    Returns:
        Uma mensagem por violação, ex.: ``"azul · heralds: Input should be ..."``.
    """
    try:
        to_match(scenario)
    except ValidationError as error:
        messages = []
        for item in error.errors():
            location = [SIDE_LABELS.get(str(part), str(part)) for part in item["loc"]]
            prefix = " · ".join(location) + ": " if location else ""
            messages.append(prefix + item["msg"].removeprefix("Value error, "))
        return messages
    return []


def _add(side: str, stat: str, amount: float) -> Callable[[Scenario], Scenario]:
    def change(scenario: Scenario) -> Scenario:
        scenario[side][stat] += amount
        return scenario

    return change


def _swap(scenario: Scenario) -> Scenario:
    scenario["blue"], scenario["red"] = scenario["red"], scenario["blue"]
    scenario["first_blood"] = "red" if scenario["first_blood"] == "blue" else "blue"
    return scenario


@dataclass(frozen=True)
class Action:
    """Ação rápida do simulador."""

    label: str
    change: Callable[[Scenario], Scenario]


ACTIONS: Final[dict[str, Action]] = {
    "dragon_blue": Action("+1 dragão azul", _add("blue", "dragons", 1)),
    "dragon_red": Action("+1 dragão vermelho", _add("red", "dragons", 1)),
    "gold_blue": Action("+1.000 gold azul", _add("blue", "total_gold", 1000)),
    "gold_red": Action("+1.000 gold vermelho", _add("red", "total_gold", 1000)),
    "xp_blue": Action("+1.000 XP azul", _add("blue", "total_experience", 1000)),
    "xp_red": Action("+1.000 XP vermelho", _add("red", "total_experience", 1000)),
    "tower_blue": Action("+1 torre azul", _add("blue", "towers_destroyed", 1)),
    "tower_red": Action("+1 torre vermelha", _add("red", "towers_destroyed", 1)),
    "swap": Action("⇄ trocar lados", _swap),
}


def apply_action(scenario: Scenario, action: str) -> Scenario:
    """Aplica uma ação rápida e valida o resultado.

    Args:
        scenario: Cenário atual (não é modificado).
        action: Chave de ``ACTIONS``.

    Returns:
        Novo cenário.

    Raises:
        ActionError: Se a ação levar a um cenário inválido (ex.: segundo dragão).
        KeyError: Se a ação não existir.
    """
    updated = ACTIONS[action].change(copy.deepcopy(scenario))
    problems = validation_messages(updated)
    if problems:
        raise ActionError(f"{ACTIONS[action].label} não é possível: {problems[0]}")
    return updated
