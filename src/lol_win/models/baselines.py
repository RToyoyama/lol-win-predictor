"""Modelos de referência: o mínimo que um modelo de verdade precisa superar."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


class GoldRuleClassifier(ClassifierMixin, BaseEstimator):
    """Regra "quem tem mais gold vence".

    Probabilidade de vitória azul = 1 se ``gold_diff > 0``, 0 se ``< 0`` e 0,5 se empatado.
    Não aprende nada: o ``fit`` só registra as classes.

    Args:
        column: Coluna com a diferença de gold (azul − vermelho).
    """

    def __init__(self, column: str = "total_gold_diff") -> None:
        self.column = column

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GoldRuleClassifier":
        """Registra as classes do alvo.

        Args:
            X: Features (precisa conter ``column``).
            y: Alvo binário.

        Returns:
            O próprio classificador.
        """
        if self.column not in X.columns:
            raise KeyError(f"Coluna {self.column!r} ausente nas features")
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probabilidades ``[P(vermelho vence), P(azul vence)]``.

        Args:
            X: Features.

        Returns:
            Array ``(n, 2)``.
        """
        check_is_fitted(self, "classes_")
        gold = X[self.column].to_numpy()
        p_blue = np.select([gold > 0, gold < 0], [1.0, 0.0], default=0.5)
        return np.column_stack([1 - p_blue, p_blue])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Classe prevista (1 = azul vence).

        Args:
            X: Features.

        Returns:
            Array ``(n,)`` de 0/1.
        """
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)
