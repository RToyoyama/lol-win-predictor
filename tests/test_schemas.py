from collections.abc import Callable

import pandas as pd
import pandera.errors as pa_errors
import pytest

from lol_win.data.schemas import BRONZE_SCHEMA

Mutation = Callable[[pd.DataFrame], pd.DataFrame]


def _side(
    *,
    kills: int,
    deaths: int,
    gold: int,
    xp: int,
    first_blood: int,
    dragons: int,
    heralds: int,
    minions: int,
    level_sum: int,
) -> dict[str, float]:
    return {
        "wards_placed": 15,
        "wards_destroyed": 2,
        "first_blood": first_blood,
        "kills": kills,
        "deaths": deaths,
        "assists": kills,
        "elite_monsters": dragons + heralds,
        "dragons": dragons,
        "heralds": heralds,
        "towers_destroyed": 0,
        "total_gold": gold,
        "avg_level": level_sum / 5,
        "total_experience": xp,
        "total_minions_killed": minions,
        "total_jungle_minions_killed": 50,
        "cs_per_min": minions / 10,
        "gold_per_min": gold / 10,
    }


@pytest.fixture
def valid_bronze() -> pd.DataFrame:
    """Duas partidas sintéticas que respeitam todas as regras do contrato."""
    rows = []
    games = [
        (1, 1, (5, 3, 17000, 18000, 1, 1, 0, 220, 34), (3, 5, 16000, 17500, 0, 0, 1, 210, 33)),
        (2, 0, (2, 6, 15000, 17000, 0, 0, 0, 200, 33), (6, 2, 17200, 18100, 1, 1, 1, 230, 35)),
    ]
    for game_id, blue_wins, blue_args, red_args in games:
        keys = ("kills", "deaths", "gold", "xp", "first_blood", "dragons", "heralds", "minions")
        blue = _side(**dict(zip(keys, blue_args[:8], strict=True)), level_sum=blue_args[8])
        red = _side(**dict(zip(keys, red_args[:8], strict=True)), level_sum=red_args[8])
        row: dict[str, float] = {"game_id": game_id, "blue_wins": blue_wins}
        row |= {f"blue_{k}": v for k, v in blue.items()}
        row |= {f"red_{k}": v for k, v in red.items()}
        row["blue_gold_diff"] = blue["total_gold"] - red["total_gold"]
        row["red_gold_diff"] = -row["blue_gold_diff"]
        row["blue_experience_diff"] = blue["total_experience"] - red["total_experience"]
        row["red_experience_diff"] = -row["blue_experience_diff"]
        rows.append(row)
    return pd.DataFrame(rows)[list(BRONZE_SCHEMA.columns)]


def _set(column: str, value: float, row: int = 0) -> Mutation:
    def mutate(df: pd.DataFrame) -> pd.DataFrame:
        df.loc[row, column] = value
        return df

    return mutate


def _duplicate_game_id(df: pd.DataFrame) -> pd.DataFrame:
    df.loc[1, "game_id"] = df.loc[0, "game_id"]
    return df


def _drop_column(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns="blue_heralds")


def _extra_column(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(blue_champions="ahri")


def _kills_as_float(df: pd.DataFrame) -> pd.DataFrame:
    return df.astype({"blue_kills": float})


def _both_first_blood(df: pd.DataFrame) -> pd.DataFrame:
    df.loc[0, ["blue_first_blood", "red_first_blood"]] = 1
    return df


def _two_dragons(df: pd.DataFrame) -> pd.DataFrame:
    df.loc[0, ["blue_dragons", "red_dragons"]] = 1
    df.loc[0, "red_elite_monsters"] = df.loc[0, "red_dragons"] + df.loc[0, "red_heralds"]
    return df


def test_valid_fixture_passes(valid_bronze: pd.DataFrame) -> None:
    BRONZE_SCHEMA.validate(valid_bronze, lazy=True)


@pytest.mark.parametrize(
    ("mutation", "expected_failure"),
    [
        (_set("blue_kills", 99), "blue_kills_eq_red_deaths"),
        (_set("red_kills", 99), "red_kills_eq_blue_deaths"),
        (_set("blue_gold_diff", 1), "gold_diff_eq_total_gold_delta"),
        (_set("red_gold_diff", 1), "gold_diff_symmetric"),
        (_set("blue_experience_diff", 1), "xp_diff_eq_total_xp_delta"),
        (_set("red_experience_diff", 1), "xp_diff_symmetric"),
        (_both_first_blood, "exactly_one_first_blood"),
        (_two_dragons, "at_most_one_dragon_total"),
        (_set("blue_elite_monsters", 0), "blue_elite_eq_dragons_plus_heralds"),
        (_set("red_gold_per_min", 1.0), "red_gold_per_min_eq_total_over_10"),
        (_set("blue_cs_per_min", 1.0), "blue_cs_per_min_eq_minions_over_10"),
        (_set("blue_avg_level", 6.9), "blue_avg_level_is_mean_of_5"),
        (_set("blue_heralds", 2), "in_range"),
        (_set("red_avg_level", 19.0), "in_range"),
        (_set("blue_towers_destroyed", 12), "in_range"),
        (_set("blue_wards_placed", -1), "greater_than_or_equal_to"),
        (_set("blue_wins", 2), "isin"),
        (_duplicate_game_id, "field_uniqueness"),
        (_drop_column, "column_in_dataframe"),
        (_extra_column, "column_in_schema"),
        (_kills_as_float, "dtype"),
    ],
)
def test_contract_rejects_invalid_data(
    valid_bronze: pd.DataFrame, mutation: Mutation, expected_failure: str
) -> None:
    with pytest.raises(pa_errors.SchemaErrors) as exc_info:
        BRONZE_SCHEMA.validate(mutation(valid_bronze.copy()), lazy=True)
    reported = set(exc_info.value.failure_cases["check"].astype(str))
    assert any(expected_failure in check for check in reported), reported
