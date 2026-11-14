"""Simulador "e se?" da probabilidade de vitória aos 10 minutos (Streamlit)."""

import os
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from sklearn.base import BaseEstimator

from lol_win.analysis.interpret import shap_explanation
from lol_win.api.schemas import to_input_frame
from lol_win.app.logic import (
    ACTIONS,
    SIDE_LABELS,
    ActionError,
    Scenario,
    apply_action,
    default_scenario,
    field_bounds,
    random_scenario,
    to_match,
    validation_messages,
)
from lol_win.config import MODELS_DIR, SIDES
from lol_win.models.predict import load_metadata, load_model, predict_proba, probability_band

BLUE: Final[str] = "#2B6CB0"
RED: Final[str] = "#C53030"
FIELDS: Final[list[tuple[str, str, float]]] = [
    ("total_gold", "Gold total", 100),
    ("total_experience", "XP total", 100),
    ("avg_level", "Nível médio", 0.2),
    ("kills", "Kills", 1),
    ("assists", "Assists", 1),
    ("total_minions_killed", "CS (lane)", 1),
    ("total_jungle_minions_killed", "CS (jungle)", 1),
    ("dragons", "Dragões", 1),
    ("heralds", "Arautos", 1),
    ("towers_destroyed", "Torres derrubadas", 1),
    ("wards_placed", "Wards colocadas", 1),
    ("wards_destroyed", "Wards destruídas", 1),
]
UNUSED_BY_MODEL: Final[frozenset[str]] = frozenset({"wards_placed", "wards_destroyed"})


def model_dir() -> Path:
    """Pasta do modelo (``LOL_WIN_MODEL_PATH`` lido em tempo de execução, útil em testes)."""
    return Path(os.environ.get("LOL_WIN_MODEL_PATH", MODELS_DIR / "model.joblib")).parent


@st.cache_resource
def load_resources(directory: Path) -> tuple[BaseEstimator, pd.DataFrame, dict]:
    """Carrega modelo, amostra de referência e metadados uma vez por processo."""
    metadata_path = directory / "metadata.json"
    metadata = load_metadata(metadata_path) if metadata_path.exists() else {}
    background = pd.read_parquet(directory / "background.parquet")
    return load_model(directory / "model.joblib"), background, metadata


def widget_key(side: str, stat: str) -> str:
    """Chave do widget de uma estatística."""
    return f"{side}__{stat}"


def write_scenario(scenario: Scenario) -> None:
    """Copia um cenário para o estado dos widgets."""
    for side in SIDES:
        for stat, value in scenario[side].items():
            st.session_state[widget_key(side, stat)] = value
    st.session_state["first_blood"] = scenario["first_blood"]


def read_scenario() -> Scenario:
    """Lê o cenário atual a partir dos widgets."""
    scenario: Scenario = {"first_blood": st.session_state["first_blood"]}
    for side in SIDES:
        scenario[side] = {stat: st.session_state[widget_key(side, stat)] for stat, *_ in FIELDS}
    return scenario


def probability(model: BaseEstimator, scenario: Scenario) -> float:
    """P(azul vence) de um cenário válido."""
    return float(predict_proba(to_input_frame([to_match(scenario)]), model)[0])


def on_action(action: str, model: BaseEstimator) -> None:
    """Callback dos botões rápidos: aplica a ação e guarda a probabilidade anterior."""
    current = read_scenario()
    try:
        updated = apply_action(current, action)
    except ActionError as error:
        st.session_state["flash"] = str(error)
        return
    before = probability(model, current) if not validation_messages(current) else None
    st.session_state["last_event"] = (ACTIONS[action].label, before)
    write_scenario(updated)


def on_load(scenario: Scenario, label: str) -> None:
    """Callback de carregar cenário (reset ou partida real)."""
    write_scenario(scenario)
    st.session_state["last_event"] = (label, None)


def hud_bar(p_blue: float) -> str:
    """HTML da barra de win-probability (conteúdo gerado só pelo app)."""
    blue_pct, red_pct = round(p_blue * 100), 100 - round(p_blue * 100)
    return f"""
    <div style="display:flex;height:44px;border-radius:8px;overflow:hidden;
                font-weight:700;font-size:1.1rem;color:white;box-shadow:0 1px 4px #0003;">
      <div style="width:{blue_pct}%;background:{BLUE};display:flex;align-items:center;
                  padding-left:12px;min-width:3.5rem;">AZUL {blue_pct}%</div>
      <div style="width:{red_pct}%;background:{RED};display:flex;align-items:center;
                  justify-content:flex-end;padding-right:12px;min-width:3.5rem;">
        {red_pct}% VERMELHO</div>
    </div>
    """


def team_inputs(side: str, bounds: dict[str, tuple[float, float]]) -> None:
    """Coluna de inputs de um time."""
    color = BLUE if side == "blue" else RED
    st.markdown(
        f"<h4 style='color:{color};margin:0'>{SIDE_LABELS[side].upper()}</h4>",
        unsafe_allow_html=True,
    )
    for stat, label, step in FIELDS:
        low, high = bounds[stat]
        is_float = stat == "avg_level"
        st.number_input(
            label + (" ¹" if stat in UNUSED_BY_MODEL else ""),
            min_value=low if is_float else int(low),
            max_value=high if is_float else int(high),
            step=step if is_float else int(step),
            format="%.1f" if is_float else "%d",
            key=widget_key(side, stat),
        )


def render_waterfall(model: BaseEstimator, background: pd.DataFrame, scenario: Scenario) -> None:
    """Waterfall SHAP do cenário atual."""
    explanation = shap_explanation(model, to_input_frame([to_match(scenario)]), background)
    fig = plt.figure()
    shap.plots.waterfall(explanation[0], max_display=10, show=False)
    st.pyplot(plt.gcf(), clear_figure=True)
    plt.close(fig)


def main() -> None:
    """Monta a página."""
    st.set_page_config(page_title="LoL Win Predictor", page_icon="⚔️", layout="wide")
    model, background, metadata = load_resources(model_dir())
    if "first_blood" not in st.session_state:
        write_scenario(default_scenario(background))
        st.session_state["rng"] = np.random.default_rng()

    st.title("⚔️ LoL Win Predictor · simulador aos 10 minutos")
    test_auc = metadata.get("test_metrics", {}).get("roc_auc", {}).get("value", float("nan"))
    st.caption(
        "Monte o estado da partida aos 10 min (ou carregue uma partida real) e veja a chance "
        f"de vitória. Modelo: {metadata.get('variant', 'desconhecido')} · "
        f"AUC no teste {test_auc:.3f}."
    )

    scenario = read_scenario()
    problems = validation_messages(scenario)
    hud = st.container()

    st.divider()
    left, right = st.columns(2)
    bounds = field_bounds()
    with left:
        team_inputs("blue", bounds)
    with right:
        team_inputs("red", bounds)
    st.radio(
        "First blood ¹",
        options=list(SIDES),
        format_func=SIDE_LABELS.get,
        horizontal=True,
        key="first_blood",
    )
    st.caption(
        "¹ Coletados porque fazem parte do contrato de entrada, mas o modelo final "
        "(`diffs_core`) **não os usa**: no Notebook 02, wards e first blood não tiveram efeito "
        "além do gold. Mexer neles não muda a probabilidade, e isso é um achado, não um bug."
    )

    st.subheader("Eventos rápidos")
    columns = st.columns(5)
    for i, (key, action) in enumerate(ACTIONS.items()):
        columns[i % 5].button(
            action.label,
            key=f"act_{key}",
            on_click=on_action,
            args=(key, model),
            width="stretch",
        )
    reset_col, random_col, _ = st.columns([1, 1, 3])
    reset_col.button(
        "↺ partida parelha",
        key="act_reset",
        on_click=on_load,
        args=(default_scenario(background), "partida parelha"),
        width="stretch",
    )
    random_col.button(
        "🎲 partida real aleatória",
        key="act_random",
        on_click=on_load,
        args=(random_scenario(background, st.session_state["rng"]), "partida real"),
        width="stretch",
    )

    with hud:
        if flash := st.session_state.pop("flash", None):
            st.warning(flash)
        if problems:
            st.error("Cenário inválido:\n\n" + "\n".join(f"- {p}" for p in problems))
            st.session_state["p_blue"] = None
            return
        p_blue = probability(model, scenario)
        st.session_state["p_blue"] = p_blue
        st.markdown(hud_bar(p_blue), unsafe_allow_html=True)
        label, description = probability_band(p_blue)
        band_col, delta_col = st.columns(2)
        band_col.markdown(f"**Faixa {label}** · {description}")
        event, before = st.session_state.get("last_event", (None, None))
        if event and before is not None:
            delta_col.markdown(f"**{event}** → {100 * (p_blue - before):+.1f} p.p. para o azul")
        elif event:
            delta_col.markdown(f"Carregado: **{event}**")

    st.divider()
    chart_col, help_col = st.columns([3, 2])
    with chart_col:
        st.subheader("Por que essa probabilidade?")
        render_waterfall(model, background, scenario)
    with help_col:
        st.subheader("Como ler")
        st.markdown(
            "- Começa em **E[f(X)]**, a chance média em log-odds (≈ 50%), e cada barra soma "
            "(vermelho, a favor do azul) ou subtrai (azul, contra o azul) até **f(x)**.\n"
            "- Log-odds +1 leva 50% a 73%; +2 leva a 88%.\n"
            "- **Gold e XP andam juntos** (colineares): a divisão do crédito entre eles é uma "
            "convenção do método. Leia os dois como um bloco.\n"
            "- **Dragão** é o objetivo com mais peso próprio (≈ 560 de gold além do gold).\n"
            "- Teste o paradoxo da torre: **+1 torre** sem o gold dela *reduz* um pouco a chance, "
            "porque o modelo aprendeu que vantagem vinda de torre vale menos que vantagem de "
            "kills e farm (Notebook 02).\n"
            "- Aos 10 min, ~27% das partidas já estão decididas (>85% ou <15%, ~93% de acerto), "
            "e ~31% estão em aberto."
        )


main()
