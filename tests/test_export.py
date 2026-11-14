import json

import joblib
import numpy as np
import pandas as pd
import pytest

from lol_win.config import INPUT_COLUMNS, MODEL_PATH, RANDOM_STATE
from lol_win.data.clean import to_silver
from lol_win.data.ingest import read_raw, to_bronze
from lol_win.data.split import split_train_test
from lol_win.models.calibrate import Variant
from lol_win.models.export import main, rebuild_champion


@pytest.fixture(scope="module")
def split() -> tuple[pd.DataFrame, pd.DataFrame]:
    return split_train_test(to_silver(to_bronze(read_raw())))


def test_variant_from_name_round_trip() -> None:
    variant = Variant("lightgbm", "sigmoid")
    assert Variant.from_name(variant.name) == variant


def test_rebuild_is_deterministic(split) -> None:
    train, test = split
    first, metadata = rebuild_champion(train)
    second, _ = rebuild_champion(train)
    x = test[list(INPUT_COLUMNS)]
    np.testing.assert_array_equal(first.predict_proba(x), second.predict_proba(x))
    assert metadata["input_columns"] == list(INPUT_COLUMNS)
    assert "roc_auc" in metadata["test_metrics"]


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="champion local ausente (rode make evaluate)")
def test_rebuild_matches_local_champion(split) -> None:
    train, test = split
    rebuilt, _ = rebuild_champion(train)
    champion = joblib.load(MODEL_PATH)
    x = test[list(INPUT_COLUMNS)]
    diff = np.abs(rebuilt.predict_proba(x)[:, 1] - champion.predict_proba(x)[:, 1])
    assert diff.max() < 1e-4


def test_main_writes_self_contained_artifacts(split, tmp_path) -> None:
    train, _ = split
    train_path = tmp_path / "train.parquet"
    train.to_parquet(train_path, index=False)
    metadata = main(output_dir=tmp_path / "models", train_path=train_path)
    out = tmp_path / "models"
    assert {p.name for p in out.iterdir()} == {
        "model.joblib",
        "metadata.json",
        "background.parquet",
    }
    assert json.loads((out / "metadata.json").read_text())["variant"] == metadata["variant"]
    background = pd.read_parquet(out / "background.parquet")
    assert list(background.columns) == list(INPUT_COLUMNS)
    assert len(background) == 500
    pd.testing.assert_frame_equal(
        background,
        train.sample(n=500, random_state=RANDOM_STATE)[list(INPUT_COLUMNS)].reset_index(drop=True),
    )
