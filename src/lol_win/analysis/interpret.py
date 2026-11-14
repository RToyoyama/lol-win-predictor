"""Interpretação do modelo e tradução de probabilidades em decisão (Notebook 03, app)."""

from collections.abc import Mapping, Sequence
from typing import Final

import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from statsmodels.stats.proportion import proportion_confint

from lol_win.config import RANDOM_STATE

BAND_EDGES: Final[tuple[float, ...]] = (0.0, 0.15, 0.35, 0.65, 0.85, 1.0)
BAND_LABELS: Final[tuple[str, ...]] = ("<15%", "15–35%", "35–65%", "65–85%", ">85%")
FEATURE_LABELS: Final[dict[str, str]] = {
    "total_gold_diff": "Gold",
    "total_experience_diff": "XP",
    "avg_level_diff": "Nível médio",
    "kills_diff": "Kills",
    "assists_diff": "Assists",
    "dragons_diff": "Dragões",
    "heralds_diff": "Arautos",
    "towers_destroyed_diff": "Torres",
    "total_minions_killed_diff": "CS (lane)",
    "total_jungle_minions_killed_diff": "CS (jungle)",
    "wards_placed_diff": "Wards colocadas",
    "wards_destroyed_diff": "Wards destruídas",
    "blue_first_blood": "First blood azul",
}


def _feature_step_index(model: Pipeline, step: str = "features") -> int:
    names = [name for name, _ in model.steps]
    if step not in names:
        raise ValueError(f"Pipeline sem a etapa {step!r}: {names}")
    return names.index(step)


def shap_explanation(
    model: Pipeline,
    X: pd.DataFrame,
    background: pd.DataFrame,
    labels: Mapping[str, str] | None = FEATURE_LABELS,
    max_background: int = 1000,
    random_state: int = RANDOM_STATE,
) -> shap.Explanation:
    """Valores SHAP (log-odds) do pipeline final, com features nas unidades originais.

    O pipeline é aplicado até a última transformação antes do modelo para calcular o SHAP,
    mas ``Explanation.data`` guarda as features **antes** do scaler (ex.: gold em unidades de
    gold), que é o que um jogador consegue ler.

    Args:
        model: Pipeline ajustado (stats brutas → features → [scaler] → modelo).
        X: Stats brutas das partidas a explicar.
        background: Stats brutas de referência (ex.: treino) para a linha de base.
        labels: Rótulos legíveis das features; ``None`` mantém os nomes técnicos.
        max_background: Máximo de linhas de referência (amostradas).
        random_state: Semente da amostragem da referência.

    Returns:
        ``shap.Explanation`` com ``values`` (n, k), ``base_values`` e ``data`` originais.

    Raises:
        TypeError: Se o estimador final não for suportado (LR ou modelo de árvore).
    """
    preprocess, estimator = model[:-1], model[-1]
    to_features = model[: _feature_step_index(model) + 1]
    sample = background.sample(n=min(max_background, len(background)), random_state=random_state)

    transformed = np.asarray(preprocess.transform(X), dtype=float)
    transformed_background = np.asarray(preprocess.transform(sample), dtype=float)
    if isinstance(estimator, LogisticRegression):
        masker = shap.maskers.Independent(transformed_background, max_samples=max_background)
        explainer = shap.LinearExplainer(estimator, masker)
    elif hasattr(estimator, "booster_") or hasattr(estimator, "estimators_"):
        explainer = shap.TreeExplainer(estimator, model_output="raw")
    else:
        raise TypeError(f"Estimador não suportado para SHAP: {type(estimator).__name__}")

    raw = explainer(transformed)
    raw_values = np.asarray(raw.values)  # noqa: PD011 (shap.Explanation, não pandas)
    values = raw_values[..., 1] if raw_values.ndim == 3 else raw_values
    base = raw.base_values[..., 1] if np.ndim(raw.base_values) == 2 else raw.base_values
    features = to_features.transform(X)
    names = [labels.get(c, c) if labels else c for c in features.columns]
    return shap.Explanation(
        values=values,
        base_values=np.broadcast_to(base, (len(X),)).astype(float),
        data=features.to_numpy(dtype=float),
        feature_names=names,
    )


def _favorite_correct(y_true: np.ndarray, p_blue: np.ndarray) -> np.ndarray:
    return y_true == (p_blue >= 0.5).astype(int)


def decision_bands(
    y_true: Sequence[int],
    p_blue: Sequence[float],
    edges: Sequence[float] = BAND_EDGES,
    labels: Sequence[str] = BAND_LABELS,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Tabela por faixa de probabilidade: volume, taxa real e acerto do favorito.

    Args:
        y_true: Alvo (1 = azul venceu).
        p_blue: Probabilidade prevista de vitória azul.
        edges: Bordas das faixas (cobrindo [0, 1]).
        labels: Rótulos das faixas.
        alpha: Nível de significância dos ICs (Wilson).

    Returns:
        DataFrame indexado pela faixa com ``n, share, mean_pred, blue_win_rate,
        favorite_accuracy, favorite_acc_ci_low, favorite_acc_ci_high``.
    """
    y = np.asarray(y_true)
    p = np.asarray(p_blue)
    band = pd.cut(p, bins=list(edges), labels=list(labels), include_lowest=True)
    frame = pd.DataFrame({"band": band, "y": y, "p": p, "hit": _favorite_correct(y, p)})
    grouped = frame.groupby("band", observed=False).agg(
        n=("y", "size"), mean_pred=("p", "mean"), blue_win_rate=("y", "mean"), hits=("hit", "sum")
    )
    low, high = proportion_confint(grouped["hits"], grouped["n"].clip(lower=1), alpha, "wilson")
    return grouped.assign(
        share=grouped["n"] / len(y),
        favorite_accuracy=grouped["hits"] / grouped["n"].replace(0, np.nan),
        favorite_acc_ci_low=low,
        favorite_acc_ci_high=high,
    ).drop(columns="hits")[
        [
            "n",
            "share",
            "mean_pred",
            "blue_win_rate",
            "favorite_accuracy",
            "favorite_acc_ci_low",
            "favorite_acc_ci_high",
        ]
    ]


def coverage_accuracy_curve(
    y_true: Sequence[int],
    p_blue: Sequence[float],
    thresholds: Sequence[float] = tuple(np.round(np.arange(0.50, 0.96, 0.01), 2)),
) -> pd.DataFrame:
    """Para cada limiar de confiança t, % de partidas com confiança ≥ t e acerto nelas.

    Confiança = probabilidade do favorito = ``max(p, 1 − p)``.

    Args:
        y_true: Alvo (1 = azul venceu).
        p_blue: Probabilidade prevista de vitória azul.
        thresholds: Limiares de confiança em [0,5; 1].

    Returns:
        DataFrame ``threshold, coverage, n, accuracy``.
    """
    y = np.asarray(y_true)
    p = np.asarray(p_blue)
    confidence = np.maximum(p, 1 - p)
    hit = _favorite_correct(y, p)
    rows = []
    for t in thresholds:
        mask = confidence >= t
        rows.append(
            {
                "threshold": float(t),
                "coverage": float(mask.mean()),
                "n": int(mask.sum()),
                "accuracy": float(hit[mask].mean()) if mask.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def confident_misses(
    y_true: Sequence[int], p_blue: Sequence[float], threshold: float = 0.8
) -> np.ndarray:
    """Marca as "viradas": o modelo deu ≥ ``threshold`` para um lado e esse lado perdeu.

    Args:
        y_true: Alvo (1 = azul venceu).
        p_blue: Probabilidade prevista de vitória azul.
        threshold: Confiança mínima do favorito.

    Returns:
        Máscara booleana ``(n,)``.
    """
    y = np.asarray(y_true)
    p = np.asarray(p_blue)
    confident = np.maximum(p, 1 - p) >= threshold
    return confident & ~_favorite_correct(y, p)


def orient_to_favorite(features: pd.DataFrame, p_blue: Sequence[float]) -> pd.DataFrame:
    """Reescreve as diffs do ponto de vista do favorito (positivo = vantagem do favorito).

    Permite juntar favoritos azuis e vermelhos na mesma análise.

    Args:
        features: Features em diffs azul − vermelho (``blue_first_blood`` vira "favorito fez FB").
        p_blue: Probabilidade prevista de vitória azul.

    Returns:
        DataFrame com o mesmo formato, orientado ao favorito.
    """
    sign = np.where(np.asarray(p_blue) >= 0.5, 1, -1)
    oriented = features.mul(sign, axis=0)
    if "blue_first_blood" in features.columns:
        fb = features["blue_first_blood"]
        oriented["blue_first_blood"] = np.where(sign == 1, fb, 1 - fb)
        oriented = oriented.rename(columns={"blue_first_blood": "favorite_first_blood"})
    return oriented
