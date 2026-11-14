"""Métricas de avaliação compartilhadas entre CV, tuning e avaliação final."""

from collections.abc import Mapping
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, log_loss, roc_auc_score

from lol_win.config import RANDOM_STATE

SCORING: Final[dict[str, str]] = {
    "roc_auc": "roc_auc",
    "log_loss": "neg_log_loss",
    "brier": "neg_brier_score",
    "accuracy": "accuracy",
    "f1": "f1",
}
LOWER_IS_BETTER: Final[frozenset[str]] = frozenset({"log_loss", "brier"})


def summarize_cv(cv_results: Mapping[str, np.ndarray]) -> dict[str, float]:
    """Resume a saída de ``cross_validate`` em média e desvio por métrica.

    Métricas "neg_" do sklearn voltam ao sinal natural (log loss e Brier positivos).

    Args:
        cv_results: Dicionário retornado por ``sklearn.model_selection.cross_validate``
            com ``scoring=SCORING`` (e opcionalmente ``return_train_score=True``).

    Returns:
        Dicionário plano, ex.: ``{"test_roc_auc_mean": 0.81, "test_roc_auc_std": 0.01, ...}``.
    """
    summary: dict[str, float] = {}
    for split in ("test", "train"):
        for metric in SCORING:
            key = f"{split}_{metric}"
            if key not in cv_results:
                continue
            values = np.asarray(cv_results[key], dtype=float)
            if metric in LOWER_IS_BETTER:
                values = -values
            summary[f"{key}_mean"] = float(values.mean())
            summary[f"{key}_std"] = float(values.std())
    return summary


def plot_cv_comparison(
    results: pd.DataFrame,
    references: Mapping[str, Mapping[str, float]] | None = None,
    metrics: tuple[str, ...] = ("roc_auc", "log_loss"),
) -> plt.Figure:
    """Dot plot (média ± desvio na CV) por configuração, com linhas de referência.

    Args:
        results: Tabela com coluna ``run_name`` e colunas ``test_<métrica>_mean/std``.
        references: Linhas verticais por métrica, ex.: ``{"roc_auc": {"regra do gold": 0.72}}``.
        metrics: Métricas a exibir, uma por painel.

    Returns:
        Figura matplotlib.
    """
    references = references or {}
    ordered = results.sort_values("test_roc_auc_mean")
    y = np.arange(len(ordered))
    fig, axes = plt.subplots(
        1, len(metrics), figsize=(6 * len(metrics), 0.45 * len(ordered) + 1.5), sharey=True
    )
    for ax, metric in zip(np.atleast_1d(axes), metrics, strict=True):
        ax.errorbar(
            ordered[f"test_{metric}_mean"],
            y,
            xerr=ordered[f"test_{metric}_std"],
            fmt="o",
            color="#2B6CB0",
            capsize=3,
        )
        for (label, value), style in zip(
            references.get(metric, {}).items(), ("--", ":", "-."), strict=False
        ):
            ax.axvline(value, color="grey", linestyle=style, label=f"{label} = {value:.3f}")
        if references.get(metric):
            ax.legend(
                fontsize=8, loc="upper left" if metric not in LOWER_IS_BETTER else "upper right"
            )
        direction = "menor é melhor" if metric in LOWER_IS_BETTER else "maior é melhor"
        ax.set(title=f"{metric} · CV 5-fold ({direction})", xlabel=metric)
    np.atleast_1d(axes)[0].set_yticks(y, ordered["run_name"])
    fig.tight_layout()
    return fig


def compute_metrics(
    y_true: np.ndarray, p_blue: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Calcula as 5 métricas do projeto a partir de probabilidades.

    Args:
        y_true: Alvo binário.
        p_blue: Probabilidade prevista de vitória do azul.
        threshold: Limiar para métricas de classe (acurácia, F1).

    Returns:
        Dicionário ``{roc_auc, log_loss, brier, accuracy, f1}``.
    """
    y_true = np.asarray(y_true)
    p_blue = np.asarray(p_blue)
    y_pred = (p_blue >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, p_blue)),
        "log_loss": float(log_loss(y_true, p_blue, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, p_blue)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def bootstrap_metrics(
    y_true: np.ndarray,
    p_blue: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Métricas com IC por bootstrap percentil (reamostragem das partidas).

    Args:
        y_true: Alvo binário.
        p_blue: Probabilidade prevista de vitória do azul.
        n_boot: Número de reamostragens.
        alpha: Nível de significância do IC.
        random_state: Semente.

    Returns:
        DataFrame indexado pela métrica com ``value, ci_low, ci_high``.
    """
    y_true = np.asarray(y_true)
    p_blue = np.asarray(p_blue)
    point = compute_metrics(y_true, p_blue)
    rng = np.random.default_rng(random_state)
    samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if np.unique(y_true[idx]).size < 2:
            continue
        samples.append(compute_metrics(y_true[idx], p_blue[idx]))
    boot = pd.DataFrame(samples)
    return pd.DataFrame(
        {
            "value": pd.Series(point),
            "ci_low": boot.quantile(alpha / 2),
            "ci_high": boot.quantile(1 - alpha / 2),
        }
    )
