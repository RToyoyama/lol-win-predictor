import pandas as pd
import pytest

from lol_win.data.ingest import read_raw, run, to_bronze, to_snake_case
from lol_win.data.schemas import BRONZE_SCHEMA


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("gameId", "game_id"),
        ("blueWins", "blue_wins"),
        ("blueCSPerMin", "blue_cs_per_min"),
        ("redTotalJungleMinionsKilled", "red_total_jungle_minions_killed"),
        ("blueGoldDiff", "blue_gold_diff"),
        ("already_snake", "already_snake"),
    ],
)
def test_to_snake_case(raw: str, expected: str) -> None:
    assert to_snake_case(raw) == expected


def test_real_raw_passes_contract() -> None:
    bronze = to_bronze(read_raw())
    assert bronze.shape == (9879, 40)
    assert set(bronze.columns) == set(BRONZE_SCHEMA.columns)


def test_run_writes_parquet_roundtrip(tmp_path) -> None:
    out = tmp_path / "bronze" / "matches.parquet"
    written = run(bronze_path=out)
    reloaded = pd.read_parquet(out)
    pd.testing.assert_frame_equal(written.reset_index(drop=True), reloaded)
