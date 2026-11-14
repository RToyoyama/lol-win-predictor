"""Split silver -> gold: holdout de teste estratificado, criado uma única vez."""

import argparse
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from lol_win.config import GOLD_TEST, GOLD_TRAIN, RANDOM_STATE, SILVER_PARQUET, TARGET, TEST_SIZE

logger = logging.getLogger(__name__)


def split_train_test(
    silver: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    target: str = TARGET,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide em treino e teste estratificando pelo alvo.

    Args:
        silver: DataFrame silver com o alvo.
        test_size: Fração do teste.
        random_state: Semente.
        target: Coluna do alvo.

    Returns:
        Tupla ``(train, test)`` com índices reiniciados.
    """
    train, test = train_test_split(
        silver, test_size=test_size, stratify=silver[target], random_state=random_state
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def run(
    silver_path: Path = SILVER_PARQUET,
    train_path: Path = GOLD_TRAIN,
    test_path: Path = GOLD_TEST,
    force: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cria a camada gold. Não sobrescreve um split existente sem ``force``.

    Args:
        silver_path: Parquet silver de entrada.
        train_path: Parquet de treino.
        test_path: Parquet de teste.
        force: Recria o split mesmo que ele já exista.

    Returns:
        Tupla ``(train, test)``.
    """
    if train_path.exists() and test_path.exists() and not force:
        logger.info("gold já existe; mantendo split fixo (use --force para recriar)")
        return pd.read_parquet(train_path), pd.read_parquet(test_path)

    train, test = split_train_test(pd.read_parquet(silver_path))
    train_path.parent.mkdir(parents=True, exist_ok=True)
    train.to_parquet(train_path, index=False)
    test.to_parquet(test_path, index=False)
    logger.info("gold: treino %d · teste %d -> %s", len(train), len(test), train_path.parent)
    return train, test


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="recria o split existente")
    run(force=parser.parse_args().force)
