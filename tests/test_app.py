import json

import joblib
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from lol_win.api.schemas import TeamStats
from lol_win.app.logic import (
    ACTIONS,
    ActionError,
    apply_action,
    default_scenario,
    field_bounds,
    random_scenario,
    to_match,
    validation_messages,
)
from lol_win.config import INPUT_COLUMNS, PROJECT_ROOT, RANDOM_STATE, TARGET
from lol_win.data.clean import to_silver
from lol_win.data.ingest import read_raw, to_bronze
from lol_win.models.train import build_model

APP_PATH = str(PROJECT_ROOT / "src" / "lol_win" / "app" / "streamlit_app.py")


@pytest.fixture(scope="module")
def silver():
    return to_silver(to_bronze(read_raw())).sample(n=1500, random_state=RANDOM_STATE)


@pytest.fixture(scope="module")
def background(silver):
    return silver[list(INPUT_COLUMNS)].head(300).reset_index(drop=True)


@pytest.fixture(scope="module")
def model_dir(silver, background, tmp_path_factory):
    directory = tmp_path_factory.mktemp("app_model")
    model = build_model("logreg", "diffs_core").fit(silver[list(INPUT_COLUMNS)], silver[TARGET])
    joblib.dump(model, directory / "model.joblib")
    background.to_parquet(directory / "background.parquet", index=False)
    (directory / "metadata.json").write_text(json.dumps({"variant": "logreg · test"}))
    return directory


def test_field_bounds_come_from_schema() -> None:
    bounds = field_bounds()
    assert set(bounds) == set(TeamStats.model_fields)
    assert bounds["dragons"] == (0, 1)
    assert bounds["avg_level"] == (1, 18)


def test_default_scenario_is_symmetric_and_valid(background) -> None:
    scenario = default_scenario(background)
    assert scenario["blue"] == scenario["red"]
    assert validation_messages(scenario) == []


def test_random_scenario_is_a_valid_real_game(background) -> None:
    scenario = random_scenario(background, np.random.default_rng(0))
    assert validation_messages(scenario) == []


def test_actions_change_only_target_and_do_not_mutate(background) -> None:
    scenario = default_scenario(background)
    updated = apply_action(scenario, "gold_blue")
    assert updated["blue"]["total_gold"] == scenario["blue"]["total_gold"] + 1000
    assert updated["red"] == scenario["red"]
    assert scenario["blue"]["total_gold"] != updated["blue"]["total_gold"]


def test_second_dragon_is_rejected(background) -> None:
    scenario = apply_action(default_scenario(background), "dragon_blue")
    with pytest.raises(ActionError, match="dragão"):
        apply_action(scenario, "dragon_red")


def test_swap_twice_is_identity(background) -> None:
    scenario = apply_action(default_scenario(background), "dragon_blue")
    assert apply_action(apply_action(scenario, "swap"), "swap") == scenario


def test_validation_messages_are_readable(background) -> None:
    scenario = default_scenario(background)
    scenario["blue"]["heralds"] = 3
    messages = validation_messages(scenario)
    assert messages and messages[0].startswith("azul · heralds")


def test_all_actions_have_labels() -> None:
    assert all(action.label for action in ACTIONS.values())


def test_app_renders_and_dragon_raises_blue_probability(model_dir, monkeypatch) -> None:
    monkeypatch.setenv("LOL_WIN_MODEL_PATH", str(model_dir / "model.joblib"))
    app = AppTest.from_file(APP_PATH, default_timeout=60).run()
    assert not app.exception
    before = app.session_state["p_blue"]
    assert 0.3 < before < 0.7

    app.button(key="act_dragon_blue").click().run()
    assert not app.exception
    assert app.session_state["p_blue"] > before

    app.button(key="act_dragon_red").click().run()
    assert any("dragão" in w.value for w in app.warning)


def test_to_match_round_trip(background) -> None:
    match = to_match(default_scenario(background))
    assert match.first_blood == "blue"
