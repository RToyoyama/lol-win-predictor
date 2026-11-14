import copy
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from lol_win.analysis.interpret import decision_bands
from lol_win.api.main import create_app
from lol_win.api.schemas import MAX_BATCH_SIZE, MatchStats, TeamStats, to_input_frame
from lol_win.config import INPUT_COLUMNS, RANDOM_STATE, TARGET, TEAM_STATS
from lol_win.data.clean import to_silver
from lol_win.data.ingest import read_raw, to_bronze
from lol_win.models.predict import BAND_LABELS, predict_proba, probability_band
from lol_win.models.train import build_model

EXAMPLE = MatchStats.model_config["json_schema_extra"]["examples"][0]


@pytest.fixture(scope="module")
def model_path(tmp_path_factory):
    silver = to_silver(to_bronze(read_raw())).sample(n=1500, random_state=RANDOM_STATE)
    model = build_model("logreg", "diffs_core").fit(silver[list(INPUT_COLUMNS)], silver[TARGET])
    directory = tmp_path_factory.mktemp("model")
    joblib.dump(model, directory / "model.joblib")
    metadata = {"variant": "logreg · none", "mlflow": {"registered_model": "lol", "version": "7"}}
    (directory / "metadata.json").write_text(json.dumps(metadata))
    return directory / "model.joblib"


@pytest.fixture(scope="module")
def client(model_path):
    with TestClient(create_app(model_path)) as test_client:
        yield test_client


def payload(**changes) -> dict:
    data = copy.deepcopy(EXAMPLE)
    for path, value in changes.items():
        side, _, field = path.partition("__")
        if field:
            data[side][field] = value
        else:
            data[side] = value
    return data


def test_team_schema_matches_team_stats() -> None:
    assert set(TeamStats.model_fields) == set(TEAM_STATS)


def test_health_ok(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_loaded": True,
        "model_version": "lol:v7",
        "variant": "logreg · none",
    }


def test_health_and_predict_503_without_model(tmp_path) -> None:
    with TestClient(create_app(tmp_path / "missing.joblib")) as no_model:
        assert no_model.get("/health").status_code == 503
        assert no_model.post("/predict", json=payload()).status_code == 503


def test_predict_matches_direct_model_call(client, model_path) -> None:
    response = client.post("/predict", json=payload())
    assert response.status_code == 200
    body = response.json()
    expected = predict_proba(to_input_frame([MatchStats(**EXAMPLE)]), joblib.load(model_path))[0]
    assert body["p_blue_win"] == pytest.approx(expected, abs=1e-4)
    assert body["p_blue_win"] + body["p_red_win"] == pytest.approx(1.0)
    assert body["favorite"] == ("blue" if body["p_blue_win"] >= 0.5 else "red")
    assert body["faixa"]["label"] in BAND_LABELS
    assert body["model_version"] == "lol:v7"


def test_predict_is_symmetric_when_swapping_teams(client) -> None:
    original = client.post("/predict", json=payload()).json()
    swapped_payload = payload(blue=EXAMPLE["red"], red=EXAMPLE["blue"], first_blood="red")
    swapped = client.post("/predict", json=swapped_payload).json()
    assert swapped["p_blue_win"] == pytest.approx(original["p_red_win"], abs=0.02)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"blue__kills": -1}, "greater than or equal"),
        ({"red__dragons": 1}, "No máximo 1 dragão"),
        ({"blue__heralds": 2}, "less than or equal"),
        ({"blue__avg_level": 19}, "less than or equal"),
        ({"blue__total_gold": 10**9}, "less than or equal"),
        ({"blue__kills": 7.5}, "integer"),
        ({"blue__golds": 100}, "Extra inputs"),
        ({"first_blood": "green"}, "'blue' or 'red'"),
        ({"blue__kills": 0}, "first blood"),
    ],
)
def test_predict_rejects_invalid_payloads(client, changes, message) -> None:
    response = client.post("/predict", json=payload(**changes))
    assert response.status_code == 422
    assert message in json.dumps(response.json(), ensure_ascii=False)


def test_predict_rejects_missing_field(client) -> None:
    data = payload()
    del data["red"]["total_gold"]
    assert client.post("/predict", json=data).status_code == 422


def test_batch_preserves_order(client) -> None:
    swapped = payload(blue=EXAMPLE["red"], red=EXAMPLE["blue"], first_blood="red")
    response = client.post("/predict/batch", json={"matches": [payload(), swapped]})
    assert response.status_code == 200
    first, second = response.json()["predictions"]
    single = client.post("/predict", json=payload()).json()
    assert first["p_blue_win"] == single["p_blue_win"]
    assert first["favorite"] != second["favorite"]


@pytest.mark.parametrize("size", [0, MAX_BATCH_SIZE + 1])
def test_batch_size_limits(client, size) -> None:
    response = client.post("/predict/batch", json={"matches": [payload()] * size})
    assert response.status_code == 422


def test_probability_band_matches_decision_bands() -> None:
    probabilities = np.array([0.0, 0.15, 0.1501, 0.35, 0.5, 0.65, 0.85, 0.8501, 1.0])
    table = decision_bands(np.zeros_like(probabilities), probabilities)
    expected = np.repeat(table.index.to_numpy(), table["n"].to_numpy())
    assert [probability_band(p)[0] for p in probabilities] == list(expected)
    with pytest.raises(ValueError):
        probability_band(1.2)


def test_to_input_frame_columns_and_encoding() -> None:
    frame = to_input_frame([MatchStats(**EXAMPLE)])
    assert list(frame.columns) == list(INPUT_COLUMNS)
    assert frame.loc[0, "blue_first_blood"] == 1
    assert frame.loc[0, "red_total_gold"] == EXAMPLE["red"]["total_gold"]
    assert isinstance(frame, pd.DataFrame)
