"""Testes de hipótese e modelos inferenciais do Notebook 02."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.discrete.discrete_model import BinaryResultsWrapper
from statsmodels.stats.proportion import proportion_confint

from lol_win.config import RANDOM_STATE, TARGET


@dataclass(frozen=True)
class ProportionTestResult:
    """Resultado do teste de uma proporção contra um valor de referência."""

    n: int
    successes: int
    rate: float
    null_rate: float
    p_value: float
    ci_low: float
    ci_high: float
    cohens_h: float


def cohens_h(rate: float, null_rate: float) -> float:
    """Tamanho de efeito de Cohen para diferença entre proporções.

    Args:
        rate: Proporção observada.
        null_rate: Proporção de referência.

    Returns:
        ``2·asin(√rate) − 2·asin(√null_rate)``; |h| < 0,2 é considerado pequeno.
    """
    return float(2 * np.arcsin(np.sqrt(rate)) - 2 * np.arcsin(np.sqrt(null_rate)))


def proportion_test(
    outcomes: pd.Series, null_rate: float = 0.5, alpha: float = 0.05
) -> ProportionTestResult:
    """Teste binomial exato bicaudal de uma proporção.

    Args:
        outcomes: Série binária (0/1).
        null_rate: Proporção sob H0.
        alpha: Nível de significância do IC (Clopper-Pearson).

    Returns:
        Estatísticas do teste, IC e tamanho de efeito.
    """
    n = int(outcomes.size)
    successes = int(outcomes.sum())
    rate = successes / n
    p_value = stats.binomtest(successes, n, null_rate).pvalue
    ci_low, ci_high = proportion_confint(successes, n, alpha=alpha, method="beta")
    return ProportionTestResult(
        n=n,
        successes=successes,
        rate=rate,
        null_rate=null_rate,
        p_value=float(p_value),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        cohens_h=cohens_h(rate, null_rate),
    )


def minimum_detectable_effect(
    n: int, null_rate: float = 0.5, alpha: float = 0.05, power: float = 0.8
) -> float:
    """Menor desvio de proporção detectável com o tamanho de amostra dado.

    Usa a aproximação normal na escala arco-seno (h de Cohen).

    Args:
        n: Tamanho da amostra.
        null_rate: Proporção sob H0.
        alpha: Nível de significância bicaudal.
        power: Poder desejado.

    Returns:
        Diferença mínima (em pontos de proporção, ex.: 0.014) acima de ``null_rate``.
    """
    h = (stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)) / np.sqrt(n)
    detectable = np.sin((h + 2 * np.arcsin(np.sqrt(null_rate))) / 2) ** 2
    return float(detectable - null_rate)


@dataclass(frozen=True)
class AssociationTestResult:
    """Resultado de um teste qui-quadrado 2x2 entre duas variáveis binárias."""

    chi2: float
    p_value: float
    cramers_v: float
    contingency: pd.DataFrame


def chi2_association(df: pd.DataFrame, column: str, target: str = TARGET) -> AssociationTestResult:
    """Qui-quadrado de independência entre uma variável binária e o alvo.

    Args:
        df: DataFrame com as duas variáveis.
        column: Variável explicativa binária.
        target: Alvo binário.

    Returns:
        Estatística, p-valor, V de Cramér e tabela de contingência.
    """
    table = pd.crosstab(df[column], df[target])
    chi2, p_value, _, _ = stats.chi2_contingency(table, correction=False)
    cramers_v = np.sqrt(chi2 / (table.to_numpy().sum() * (min(table.shape) - 1)))
    return AssociationTestResult(
        chi2=float(chi2), p_value=float(p_value), cramers_v=float(cramers_v), contingency=table
    )


def fit_logit(
    df: pd.DataFrame, features: Sequence[str], target: str = TARGET
) -> BinaryResultsWrapper:
    """Ajusta uma regressão logística (statsmodels) com intercepto.

    Args:
        df: DataFrame com features e alvo.
        features: Variáveis explicativas.
        target: Alvo binário.

    Returns:
        Resultado ajustado (coeficientes, erros-padrão, p-valores).
    """
    design = sm.add_constant(df[list(features)].astype(float), has_constant="add")
    return sm.Logit(df[target].astype(float), design).fit(disp=0)


def odds_ratio_table(
    result: BinaryResultsWrapper,
    per: Mapping[str, float] | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Converte coeficientes de um logit em odds ratios com IC.

    Args:
        result: Logit ajustado.
        per: Unidade de leitura por variável (ex.: ``{"total_gold_diff": 1000}`` reporta o
            odds ratio por +1.000 de gold). Padrão: 1 para todas.
        alpha: Nível de significância do IC.

    Returns:
        DataFrame indexado pela variável com ``per, coef, odds_ratio, or_ci_low, or_ci_high,
        p_value`` (sem o intercepto).
    """
    per = per or {}
    params = result.params.drop("const")
    conf = result.conf_int(alpha=alpha).drop("const")
    scale = pd.Series({name: per.get(name, 1.0) for name in params.index})
    return pd.DataFrame(
        {
            "per": scale,
            "coef": params * scale,
            "odds_ratio": np.exp(params * scale),
            "or_ci_low": np.exp(conf[0] * scale),
            "or_ci_high": np.exp(conf[1] * scale),
            "p_value": result.pvalues.drop("const"),
        }
    )


def gold_equivalent(
    params: pd.Series | Mapping[str, float], objective: str, gold_column: str
) -> float:
    """Valor de um objetivo em gold: razão entre coeficientes do logit.

    Args:
        params: Coeficientes do logit (gold na unidade original, 1 = 1 de gold).
        objective: Nome da variável do objetivo.
        gold_column: Nome da variável de gold diff.

    Returns:
        Quantidade de gold diff com o mesmo efeito no log-odds que +1 do objetivo.
    """
    return float(params[objective] / params[gold_column])


def bootstrap_gold_equivalent(
    df: pd.DataFrame,
    objectives: Sequence[str],
    gold_column: str,
    controls: Sequence[str] = (),
    target: str = TARGET,
    n_boot: int = 1000,
    alpha: float = 0.05,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Estima o valor em gold de cada objetivo com IC por bootstrap (percentil).

    Ajusta um único logit ``target ~ gold + objectives + controls`` por reamostragem.

    Args:
        df: DataFrame com as variáveis.
        objectives: Variáveis cujo valor em gold será estimado.
        gold_column: Variável de gold diff (unidade original).
        controls: Variáveis extras de controle.
        target: Alvo binário.
        n_boot: Número de reamostragens.
        alpha: Nível de significância do IC.
        random_state: Semente.

    Returns:
        DataFrame indexado pelo objetivo com ``gold_value, ci_low, ci_high, n_boot``.
    """
    features = [gold_column, *objectives, *controls]
    point = fit_logit(df, features, target).params
    design = sm.add_constant(df[features].astype(float), has_constant="add").to_numpy()
    y = df[target].to_numpy(dtype=float)
    gold_idx = 1
    objective_idx = [features.index(obj) + 1 for obj in objectives]

    rng = np.random.default_rng(random_state)
    samples = np.empty((n_boot, len(objectives)))
    for b in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        params = sm.Logit(y[idx], design[idx]).fit(disp=0).params
        samples[b] = params[objective_idx] / params[gold_idx]

    low, high = np.quantile(samples, [alpha / 2, 1 - alpha / 2], axis=0)
    return pd.DataFrame(
        {
            "gold_value": [gold_equivalent(point, obj, gold_column) for obj in objectives],
            "ci_low": low,
            "ci_high": high,
            "n_boot": n_boot,
        },
        index=pd.Index(objectives, name="objective"),
    )


def raw_vs_adjusted_odds(
    df: pd.DataFrame,
    variables: Sequence[str],
    controls: Sequence[str],
    per: Mapping[str, float] | None = None,
    target: str = TARGET,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Compara o odds ratio bruto e o ajustado por controles para cada variável.

    Cada variável é ajustada em dois logits: sozinha e junto dos ``controls``.

    Args:
        df: DataFrame com variáveis, controles e alvo.
        variables: Variáveis de interesse (uma de cada vez).
        controls: Variáveis de controle do modelo ajustado.
        per: Unidade de leitura por variável (ver ``odds_ratio_table``).
        target: Alvo binário.
        alpha: Nível de significância do IC.

    Returns:
        DataFrame com MultiIndex ``(variable, model)``, ``model`` em ``{"bruto", "ajustado"}``.
    """
    frames = {}
    for variable in variables:
        for model, features in (("bruto", [variable]), ("ajustado", [*controls, variable])):
            table = odds_ratio_table(fit_logit(df, features, target), per=per, alpha=alpha)
            frames[(variable, model)] = table.loc[variable]
    out = pd.DataFrame(frames).T
    out.index.names = ["variable", "model"]
    return out
