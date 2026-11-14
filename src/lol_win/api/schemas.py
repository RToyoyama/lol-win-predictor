"""Contratos de entrada e saída da API (Pydantic v2)."""

from collections.abc import Sequence
from typing import Final, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from lol_win.config import INPUT_COLUMNS, SIDES

MAX_BATCH_SIZE: Final[int] = 1000

Side = Literal["blue", "red"]


class TeamStats(BaseModel):
    """Estatísticas de um time aos 10 minutos.

    Limites inferiores são regras do jogo; superiores são tetos de plausibilidade aos 10 min.
    """

    model_config = ConfigDict(extra="forbid")

    wards_placed: int = Field(ge=0, le=500, description="Wards colocadas")
    wards_destroyed: int = Field(ge=0, le=500, description="Wards inimigas destruídas")
    kills: int = Field(ge=0, le=100)
    assists: int = Field(ge=0, le=300)
    dragons: int = Field(ge=0, le=1, description="Dragões (no máximo 1 aos 10 min)")
    heralds: int = Field(ge=0, le=1, description="Arautos do Vale (no máximo 1 aos 10 min)")
    towers_destroyed: int = Field(ge=0, le=11, description="Torres inimigas derrubadas")
    total_minions_killed: int = Field(ge=0, le=500, description="CS de lane")
    total_jungle_minions_killed: int = Field(ge=0, le=500, description="CS de jungle")
    total_gold: int = Field(ge=0, le=50_000)
    total_experience: int = Field(ge=0, le=50_000)
    avg_level: float = Field(ge=1, le=18, description="Nível médio dos 5 campeões")


class MatchStats(BaseModel):
    """Snapshot de uma partida aos 10 minutos: os dois times + quem fez o first blood."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "blue": {
                        "wards_placed": 18,
                        "wards_destroyed": 3,
                        "kills": 7,
                        "assists": 9,
                        "dragons": 1,
                        "heralds": 0,
                        "towers_destroyed": 0,
                        "total_minions_killed": 220,
                        "total_jungle_minions_killed": 52,
                        "total_gold": 17200,
                        "total_experience": 18100,
                        "avg_level": 7.0,
                    },
                    "red": {
                        "wards_placed": 15,
                        "wards_destroyed": 2,
                        "kills": 4,
                        "assists": 5,
                        "dragons": 0,
                        "heralds": 1,
                        "towers_destroyed": 0,
                        "total_minions_killed": 212,
                        "total_jungle_minions_killed": 49,
                        "total_gold": 15600,
                        "total_experience": 17400,
                        "avg_level": 6.8,
                    },
                    "first_blood": "blue",
                }
            ]
        },
    )

    blue: TeamStats
    red: TeamStats
    first_blood: Side = Field(description="Time que fez o primeiro abate")

    @model_validator(mode="after")
    def check_game_rules(self) -> "MatchStats":
        """Regras entre times (espelham o contrato Pandera dos dados de treino)."""
        if self.blue.dragons + self.red.dragons > 1:
            raise ValueError("No máximo 1 dragão no total aos 10 minutos")
        if getattr(self, self.first_blood).kills < 1:
            raise ValueError(
                f"O time do first blood ({self.first_blood}) precisa ter ao menos 1 kill"
            )
        return self

    def to_input_row(self) -> dict[str, float]:
        """Converte para o formato plano de ``INPUT_COLUMNS``.

        Returns:
            Dicionário ``{coluna: valor}`` na ordem de ``INPUT_COLUMNS``.
        """
        row: dict[str, float] = {"blue_first_blood": int(self.first_blood == "blue")}
        for side in SIDES:
            team = getattr(self, side).model_dump()
            row |= {f"{side}_{stat}": value for stat, value in team.items()}
        return {column: row[column] for column in INPUT_COLUMNS}


def to_input_frame(matches: Sequence[MatchStats]) -> pd.DataFrame:
    """Único ponto de conversão payload → entrada do modelo (usado pela API e pelo app).

    Args:
        matches: Partidas validadas.

    Returns:
        DataFrame com as colunas de ``INPUT_COLUMNS``, uma linha por partida.
    """
    return pd.DataFrame([match.to_input_row() for match in matches], columns=list(INPUT_COLUMNS))


class Band(BaseModel):
    """Faixa de decisão da probabilidade."""

    label: str = Field(examples=[">85%"])
    description: str = Field(examples=["azul muito favorito"])


class PredictionResponse(BaseModel):
    """Probabilidades de vitória e faixa de decisão."""

    p_blue_win: float = Field(ge=0, le=1)
    p_red_win: float = Field(ge=0, le=1)
    favorite: Side
    faixa: Band
    model_version: str


class BatchRequest(BaseModel):
    """Lote de partidas."""

    model_config = ConfigDict(extra="forbid")

    matches: list[MatchStats] = Field(min_length=1, max_length=MAX_BATCH_SIZE)


class BatchResponse(BaseModel):
    """Predições na mesma ordem do lote recebido."""

    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    """Estado da API (readiness)."""

    status: Literal["ok", "unavailable"]
    model_loaded: bool
    model_version: str | None = None
    variant: str | None = None
