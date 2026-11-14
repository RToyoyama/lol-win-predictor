import pandas as pd
import pytest

from lol_win.config import INPUT_COLUMNS, TARGET
from lol_win.data import split
from lol_win.data.clean import to_silver
from lol_win.data.ingest import read_raw, to_bronze
from lol_win.data.schemas import REDUNDANT_COLUMNS
from lol_win.data.split import split_train_test


@pytest.fixture(scope="module")
def silver() -> pd.DataFrame:
    return to_silver(to_bronze(read_raw()))


def test_silver_drops_exactly_the_redundant_columns(silver: pd.DataFrame) -> None:
    assert len(REDUNDANT_COLUMNS) == 13
    assert silver.shape == (9879, 27)
    assert not set(REDUNDANT_COLUMNS) & set(silver.columns)
    assert set(silver.columns) == {"game_id", TARGET, *INPUT_COLUMNS}


def test_split_is_stratified_disjoint_and_reproducible(silver: pd.DataFrame) -> None:
    train, test = split_train_test(silver)
    assert len(test) == pytest.approx(0.2 * len(silver), abs=1)
    assert not set(train["game_id"]) & set(test["game_id"])
    assert abs(train[TARGET].mean() - test[TARGET].mean()) < 0.005
    train_again, test_again = split_train_test(silver)
    pd.testing.assert_frame_equal(test, test_again)


def test_split_run_does_not_overwrite_without_force(silver: pd.DataFrame, tmp_path) -> None:
    silver_path = tmp_path / "silver.parquet"
    train_path, test_path = tmp_path / "train.parquet", tmp_path / "test.parquet"
    silver.to_parquet(silver_path, index=False)
    split.run(silver_path, train_path, test_path)

    sentinel = pd.DataFrame({"game_id": [1]})
    sentinel.to_parquet(test_path, index=False)
    _, kept = split.run(silver_path, train_path, test_path)
    assert kept.equals(sentinel)

    _, recreated = split.run(silver_path, train_path, test_path, force=True)
    assert len(recreated) > 1
