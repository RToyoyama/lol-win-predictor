"""Comparação de calibração (sem / sigmoid / isotônica) com predições out-of-fold no treino."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal, get_args

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from lol_win.config import CV_FOLDS, RANDOM_STATE
from lol_win.models.evaluate import compute_metrics
from lol_win.models.train import ModelName
from lol_win.models.tune import TuningResult, build_tuned_model

CalibrationMethod = Literal["none", "sigmoid", "isotonic"]
CALIBRATION_METHODS: Final[tuple[str, ...]] = get_args(CalibrationMethod)
MODEL_COMPLEXITY: Final[dict[str, int]] = {"logreg": 0, "lightgbm": 1}
CALIBRATION_COMPLEXITY: Final[dict[str, int]] = {"none": 0, "sigmoid": 1, "isotonic": 2}


@dataclass(frozen=True)
class Variant:
    """Um candidato final: modelo com tuning + método de calibração."""

    model: ModelName
    calibration: CalibrationMethod

    @property
    def name(self) -> str:
        """Nome legível (ex.: ``logreg · sigmoid``)."""
        return f"{self.model} · {self.calibration}"

    @property
    def complexity(self) -> tuple[int, int]:
        """Ordem de simplicidade usada no desempate (menor = mais simples)."""
        return MODEL_COMPLEXITY[self.model], CALIBRATION_COMPLEXITY[self.calibration]


def build_variant(
    variant: Variant, tuning: TuningResult, cv_folds: int = CV_FOLDS
) -> BaseEstimator:
    """Monta o estimador de uma variante (pipeline, opcionalmente envolto em calibração).

    Args:
        variant: Modelo + método de calibração.
        tuning: Melhor configuração do tuning para o modelo.
        cv_folds: Folds internos do ``CalibratedClassifierCV``.

    Returns:
        Estimador não ajustado que recebe stats brutas.
    """
    pipeline = build_tuned_model(variant.model, tuning.feature_set, tuning.params)
    if variant.calibration == "none":
        return pipeline
    return CalibratedClassifierCV(pipeline, method=variant.calibration, cv=cv_folds)


def expected_calibration_error(y_true: np.ndarray, p_blue: np.ndarray, n_bins: int = 10) -> float:
    """ECE: média ponderada de |freq. observada − prob. média| em faixas uniformes.

    Args:
        y_true: Alvo binário.
        p_blue: Probabilidades previstas.
        n_bins: Número de faixas em [0, 1].

    Returns:
        ECE entre 0 (perfeito) e 1.
    """
    y_true = np.asarray(y_true)
    p_blue = np.asarray(p_blue)
    bins = np.minimum((p_blue * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in np.unique(bins):
        mask = bins == b
        ece += mask.mean() * abs(y_true[mask].mean() - p_blue[mask].mean())
    return float(ece)


def oof_probabilities(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,
) -> np.ndarray:
    """Probabilidades out-of-fold: cada partida é prevista por um modelo que não a viu.

    Args:
        estimator: Estimador não ajustado.
        X: Stats brutas de treino.
        y: Alvo.
        cv_folds: Número de folds externos.
        random_state: Semente dos folds.

    Returns:
        Array ``(n,)`` com P(azul vence).
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    return cross_val_predict(estimator, X, y, cv=cv, method="predict_proba")[:, 1]


def compare_calibration(
    tuning: Mapping[str, TuningResult],
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int = CV_FOLDS,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Avalia todas as variantes (modelo × calibração) com predições out-of-fold.

    Args:
        tuning: Resultados do tuning por modelo.
        X: Stats brutas de treino.
        y: Alvo.
        cv_folds: Número de folds.

    Returns:
        Tupla ``(tabela de métricas + ECE por variante, probabilidades OOF por variante)``.
    """
    rows, oof = [], {}
    for model, result in tuning.items():
        for method in CALIBRATION_METHODS:
            variant = Variant(model, method)
            p_blue = oof_probabilities(build_variant(variant, result, cv_folds), X, y, cv_folds)
            oof[variant.name] = p_blue
            rows.append(
                {
                    "variant": variant.name,
                    "model": model,
                    "calibration": method,
                    **compute_metrics(y, p_blue),
                    "ece": expected_calibration_error(y, p_blue),
                }
            )
    return pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True), oof


def choose_variant(table: pd.DataFrame, tolerance: float) -> Variant:
    """Escolhe a variante mais simples cujo log loss está a até ``tolerance`` do melhor.

    Simplicidade: LR antes de LightGBM; sem calibração antes de sigmoid antes de isotônica.

    Args:
        table: Saída de ``compare_calibration``.
        tolerance: Folga aceitável de log loss (ex.: 1 desvio-padrão da CV).

    Returns:
        Variante escolhida.
    """
    best = table["log_loss"].min()
    eligible = table[table["log_loss"] <= best + tolerance]
    variants = [Variant(row.model, row.calibration) for row in eligible.itertuples()]
    return min(variants, key=lambda v: v.complexity)


def plot_calibration_curves(
    y_true: np.ndarray, probabilities: Mapping[str, np.ndarray], n_bins: int = 10
) -> plt.Figure:
    """Diagrama de confiabilidade + distribuição das probabilidades.

    Args:
        y_true: Alvo binário.
        probabilities: Probabilidades por nome de variante.
        n_bins: Número de faixas.

    Returns:
        Figura matplotlib.
    """
    fig, (ax_curve, ax_hist) = plt.subplots(
        2, 1, figsize=(7, 8), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )
    ax_curve.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1, label="perfeito")
    for name, p_blue in probabilities.items():
        frac_pos, mean_pred = calibration_curve(y_true, p_blue, n_bins=n_bins, strategy="uniform")
        ece = expected_calibration_error(y_true, p_blue, n_bins)
        ax_curve.plot(mean_pred, frac_pos, marker="o", label=f"{name} (ECE {ece:.3f})")
        ax_hist.hist(p_blue, bins=40, range=(0, 1), histtype="step", label=name)
    ax_curve.set(
        ylabel="fração real de vitórias azuis",
        title="Calibração (predições out-of-fold no treino)",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax_curve.legend(fontsize=8)
    ax_hist.set(xlabel="P(azul vence) prevista", ylabel="partidas")
    fig.tight_layout()
    return fig
