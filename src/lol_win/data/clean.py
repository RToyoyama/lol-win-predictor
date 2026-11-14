"""Limpeza bronze -> silver: remove colunas redundantes e valida o contrato silver."""

import logging
from pathlib import Path

import pandas as pd

from lol_win.config import BRONZE_PARQUET, SILVER_PARQUET
from lol_win.data.schemas import REDUNDANT_COLUMNS, SILVER_SCHEMA

logger = logging.getLogger(__name__)


def to_silver(bronze: pd.DataFrame) -> pd.DataFrame:
    """Remove as colunas determinísticas redundantes e valida o resultado.

    Args:
        bronze: DataFrame da camada bronze.

    Returns:
        DataFrame silver validado.

    Raises:
        pandera.errors.SchemaErrors: Se o resultado violar o contrato silver.
    """
    silver = bronze.drop(columns=list(REDUNDANT_COLUMNS))
    return SILVER_SCHEMA.validate(silver, lazy=True)


def run(bronze_path: Path = BRONZE_PARQUET, silver_path: Path = SILVER_PARQUET) -> pd.DataFrame:
    """Executa bronze -> silver e grava em Parquet.

    Args:
        bronze_path: Parquet de entrada.
        silver_path: Parquet de saída.

    Returns:
        DataFrame silver gravado.
    """
    silver = to_silver(pd.read_parquet(bronze_path))
    silver_path.parent.mkdir(parents=True, exist_ok=True)
    silver.to_parquet(silver_path, index=False)
    logger.info("silver: %d linhas x %d colunas -> %s", *silver.shape, silver_path)
    return silver


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run()
