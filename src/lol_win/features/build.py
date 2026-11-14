"""Construção de features de modelagem (fonte única para treino, API e app).

O modelo recebe as estatísticas brutas dos dois times aos 10 min (``INPUT_COLUMNS``) e o
``Pipeline`` calcula as features internamente: ``WardsClipper`` → ``FeatureBuilder`` → modelo.
"""

from collections.abc import Sequence
from typing import Final, Literal, get_args

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from lol_win.config import INPUT_COLUMNS, SIDES, TEAM_STATS, WARDS_CAP_QUANTILE

FeatureSet = Literal["diffs", "diffs_abs", "diffs_core"]
FEATURE_SETS: Final[tuple[str, ...]] = get_args(FeatureSet)
WARDS_COLUMNS: Final[tuple[str, str]] = ("blue_wards_placed", "red_wards_placed")
CORE_EXCLUDED_STATS: Final[tuple[str, ...]] = ("wards_placed", "wards_destroyed")


def side_diffs(df: pd.DataFrame, stats: Sequence[str] = TEAM_STATS) -> pd.DataFrame:
    """Calcula diferenças azul − vermelho para cada estatística.

    Args:
        df: DataFrame com colunas ``blue_<stat>`` e ``red_<stat>``.
        stats: Estatísticas a diferenciar.

    Returns:
        DataFrame com colunas ``<stat>_diff``, mesmo índice de ``df``.
    """
    return pd.DataFrame({f"{stat}_diff": df[f"blue_{stat}"] - df[f"red_{stat}"] for stat in stats})


def clip_upper(values: pd.Series, upper: float) -> pd.Series:
    """Limita valores a um teto (winsorização superior).

    Args:
        values: Série numérica.
        upper: Teto; valores acima viram ``upper``.

    Returns:
        Série com o mesmo índice e valores ``<= upper``.
    """
    return values.clip(upper=upper)


def build_features(stats: pd.DataFrame, feature_set: FeatureSet = "diffs") -> pd.DataFrame:
    """Transforma estatísticas brutas dos dois times em features de modelagem.

    Args:
        stats: DataFrame com ``INPUT_COLUMNS`` (colunas extras são ignoradas).
        feature_set: ``"diffs"`` (diferenças azul − vermelho + ``blue_first_blood``),
            ``"diffs_abs"`` (as mesmas diffs + estatísticas absolutas de cada time) ou
            ``"diffs_core"`` (diffs sem visão e sem first blood, que não tiveram efeito além do
            gold no Notebook 02).

    Returns:
        DataFrame de features com o mesmo índice de ``stats``.

    Raises:
        ValueError: Se ``feature_set`` for desconhecido.
        KeyError: Se faltar alguma coluna de ``INPUT_COLUMNS``.
    """
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"feature_set deve ser um de {FEATURE_SETS}, recebido {feature_set!r}")
    missing = set(INPUT_COLUMNS) - set(stats.columns)
    if missing:
        raise KeyError(f"Colunas de entrada ausentes: {sorted(missing)}")

    if feature_set == "diffs_core":
        return side_diffs(stats, [s for s in TEAM_STATS if s not in CORE_EXCLUDED_STATS])
    features = side_diffs(stats).assign(blue_first_blood=stats["blue_first_blood"])
    if feature_set == "diffs":
        return features
    absolutes = stats[[f"{side}_{stat}" for side in SIDES for stat in TEAM_STATS]]
    return pd.concat([features, absolutes], axis=1)


class WardsClipper(TransformerMixin, BaseEstimator):
    """Aplica um teto comum às wards colocadas dos dois times, aprendido só no ``fit``.

    O teto é o quantil das wards dos **dois lados empilhados**, então azul e vermelho recebem o
    mesmo limite e a simetria entre lados é preservada.

    Args:
        quantile: Quantil do teto (ex.: 0.99). ``None`` desliga o cap (passthrough), útil para
            comparar com/sem cap na validação cruzada.
        columns: Colunas que recebem o teto.
    """

    def __init__(
        self, quantile: float | None = WARDS_CAP_QUANTILE, columns: Sequence[str] = WARDS_COLUMNS
    ) -> None:
        self.quantile = quantile
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "WardsClipper":
        """Aprende o teto a partir dos dados de treino.

        Args:
            X: Estatísticas brutas.
            y: Ignorado.

        Returns:
            O próprio transformer ajustado.
        """
        pooled = pd.concat([X[column] for column in self.columns])
        self.cap_ = np.inf if self.quantile is None else float(pooled.quantile(self.quantile))
        self.feature_names_in_ = X.columns.to_numpy()
        return self

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:
        """Nomes das colunas de saída (iguais às de entrada).

        Args:
            input_features: Ignorado.

        Returns:
            Array com os nomes das colunas vistas no ``fit``.
        """
        check_is_fitted(self, "feature_names_in_")
        return self.feature_names_in_

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Aplica o teto aprendido.

        Args:
            X: Estatísticas brutas.

        Returns:
            Cópia de ``X`` com as colunas de wards limitadas a ``cap_``.
        """
        check_is_fitted(self, "cap_")
        clipped = X.copy()
        for column in self.columns:
            clipped[column] = clip_upper(clipped[column], self.cap_)
        return clipped


class FeatureBuilder(TransformerMixin, BaseEstimator):
    """Adapta ``build_features`` à interface de transformer do scikit-learn.

    Args:
        feature_set: Conjunto de features (ver ``build_features``).
    """

    def __init__(self, feature_set: FeatureSet = "diffs") -> None:
        self.feature_set = feature_set

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "FeatureBuilder":
        """Registra os nomes das features geradas.

        Args:
            X: Estatísticas brutas.
            y: Ignorado.

        Returns:
            O próprio transformer ajustado.
        """
        self.feature_names_out_ = build_features(X.head(1), self.feature_set).columns.to_numpy()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Calcula as features.

        Args:
            X: Estatísticas brutas.

        Returns:
            DataFrame de features.
        """
        check_is_fitted(self, "feature_names_out_")
        return build_features(X, self.feature_set)

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:
        """Nomes das features geradas (compatível com ``Pipeline.get_feature_names_out``).

        Args:
            input_features: Ignorado.

        Returns:
            Array com os nomes das colunas de saída.
        """
        check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_


def make_feature_pipeline(
    feature_set: FeatureSet = "diffs", wards_quantile: float | None = WARDS_CAP_QUANTILE
) -> Pipeline:
    """Monta a etapa de preparação (clip de wards → features) para ser seguida por um modelo.

    Args:
        feature_set: Conjunto de features.
        wards_quantile: Quantil do teto de wards (``None`` desliga o cap).

    Returns:
        ``Pipeline`` sklearn que recebe estatísticas brutas e devolve features.
    """
    return Pipeline(
        [
            ("clip_wards", WardsClipper(quantile=wards_quantile)),
            ("features", FeatureBuilder(feature_set=feature_set)),
        ]
    )
