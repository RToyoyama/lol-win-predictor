"""Funções reutilizáveis de análise exploratória (usadas pelos notebooks)."""

from collections.abc import Sequence
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.proportion import proportion_confint
from statsmodels.tools.tools import add_constant

from lol_win.config import FIGURES_DIR, TARGET

TEAM_PALETTE: Final[dict[str, str]] = {"blue": "#2B6CB0", "red": "#C53030"}
OUTCOME_PALETTE: Final[dict[int, str]] = {1: TEAM_PALETTE["blue"], 0: TEAM_PALETTE["red"]}
DEFAULT_QUANTILES: Final[tuple[float, ...]] = (0.5, 0.9, 0.95, 0.99, 0.995, 1.0)


def set_plot_style() -> None:
    """Aplica o estilo visual padrão dos notebooks."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({"figure.dpi": 110, "axes.titleweight": "bold"})


def save_figure(fig: plt.Figure, name: str, figures_dir: Path = FIGURES_DIR) -> Path:
    """Salva uma figura em PNG com nome descritivo.

    Args:
        fig: Figura matplotlib.
        name: Nome sem extensão, ex.: ``"eda_win_rate_by_gold_diff"``.
        figures_dir: Diretório de destino.

    Returns:
        Caminho do arquivo salvo.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def outlier_summary(values: pd.Series, quantiles: Sequence[float] = DEFAULT_QUANTILES) -> pd.Series:
    """Resume a cauda de uma distribuição por percentis.

    Args:
        values: Série numérica.
        quantiles: Percentis (0–1) a reportar.

    Returns:
        Série indexada por rótulos ``"p50"``, ``"p99"``, ``"max"`` etc.
    """
    result = values.quantile(list(quantiles))
    result.index = ["max" if q == 1.0 else f"p{q * 100:g}" for q in quantiles]
    return result


def _summarize_wins(target: pd.Series, groups: pd.Series, alpha: float) -> pd.DataFrame:
    grouped = target.groupby(groups, observed=True).agg(wins="sum", n="count")
    low, high = proportion_confint(grouped["wins"], grouped["n"], alpha=alpha, method="wilson")
    return grouped.assign(win_rate=grouped["wins"] / grouped["n"], ci_low=low, ci_high=high)


def win_rate_by_group(
    df: pd.DataFrame, column: str, target: str = TARGET, alpha: float = 0.05
) -> pd.DataFrame:
    """Taxa de vitória por valor de uma variável discreta, com IC de Wilson.

    Args:
        df: DataFrame com a variável e o alvo.
        column: Variável discreta.
        target: Coluna binária do alvo.
        alpha: Nível de significância do IC.

    Returns:
        DataFrame indexado pelo valor, com ``wins, n, win_rate, ci_low, ci_high``.
    """
    return _summarize_wins(df[target], df[column], alpha)


def win_rate_by_bins(
    df: pd.DataFrame,
    column: str,
    bins: int | Sequence[float],
    target: str = TARGET,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Taxa de vitória por faixa de uma variável contínua, com IC de Wilson.

    Args:
        df: DataFrame com a variável e o alvo.
        column: Variável contínua.
        bins: Número de faixas ou bordas explícitas (``pd.cut``).
        target: Coluna binária do alvo.
        alpha: Nível de significância do IC.

    Returns:
        DataFrame indexado pela faixa, com ``center, wins, n, win_rate, ci_low, ci_high``.
    """
    intervals = pd.cut(df[column], bins=bins)
    summary = _summarize_wins(df[target], intervals, alpha)
    centers = [interval.mid for interval in summary.index]
    return summary.assign(center=centers)


def compute_vif(features: pd.DataFrame) -> pd.DataFrame:
    """Calcula o VIF de cada coluna (com intercepto no desenho).

    Args:
        features: Apenas colunas numéricas, sem o alvo.

    Returns:
        DataFrame ``feature, vif`` ordenado do maior para o menor.
    """
    design = add_constant(features.astype(float), has_constant="add").to_numpy()
    vifs = [variance_inflation_factor(design, i + 1) for i in range(features.shape[1])]
    return (
        pd.DataFrame({"feature": features.columns, "vif": vifs})
        .sort_values("vif", ascending=False)
        .reset_index(drop=True)
    )


def target_correlations(
    df: pd.DataFrame, columns: Sequence[str], target: str = TARGET
) -> pd.DataFrame:
    """Correlação ponto-bisserial de cada variável com o alvo binário.

    Args:
        df: DataFrame com as variáveis e o alvo.
        columns: Variáveis numéricas a avaliar.
        target: Coluna binária do alvo.

    Returns:
        DataFrame ``feature, r, p_value`` ordenado por ``|r|`` decrescente.
    """
    rows = []
    for column in columns:
        result = stats.pointbiserialr(df[target], df[column])
        rows.append({"feature": column, "r": result.statistic, "p_value": result.pvalue})
    out = pd.DataFrame(rows)
    return out.reindex(out["r"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def equality_rate(left: pd.Series, right: pd.Series, atol: float = 1e-6) -> float:
    """Fração de linhas em que duas séries são iguais (com tolerância para floats).

    Args:
        left: Primeira série.
        right: Segunda série.
        atol: Tolerância absoluta.

    Returns:
        Proporção entre 0 e 1.
    """
    return float(np.isclose(left, right, atol=atol).mean())
