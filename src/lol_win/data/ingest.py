"""Ingestão raw -> bronze: renomeia para snake_case, valida o contrato e grava Parquet."""

import logging
import re
from pathlib import Path

import pandas as pd

from lol_win.config import BRONZE_PARQUET, RAW_CSV
from lol_win.data.schemas import BRONZE_SCHEMA

logger = logging.getLogger(__name__)

_ACRONYM_BOUNDARY = re.compile(r"(.)([A-Z][a-z]+)")
_LOWER_UPPER_BOUNDARY = re.compile(r"([a-z0-9])([A-Z])")


def to_snake_case(name: str) -> str:
    """Converte um nome camelCase (com siglas) para snake_case.

    Args:
        name: Nome original, ex.: ``"blueCSPerMin"``.

    Returns:
        Nome em snake_case, ex.: ``"blue_cs_per_min"``.
    """
    name = _ACRONYM_BOUNDARY.sub(r"\1_\2", name)
    return _LOWER_UPPER_BOUNDARY.sub(r"\1_\2", name).lower()


def read_raw(path: Path = RAW_CSV) -> pd.DataFrame:
    """Lê o CSV bruto sem nenhuma transformação.

    Args:
        path: Caminho do CSV.

    Returns:
        DataFrame com as colunas originais.
    """
    return pd.read_csv(path)


def to_bronze(raw: pd.DataFrame) -> pd.DataFrame:
    """Padroniza nomes e valida o contrato da camada bronze.

    Args:
        raw: DataFrame como lido do CSV.

    Returns:
        DataFrame validado, com colunas em snake_case.

    Raises:
        pandera.errors.SchemaErrors: Se qualquer regra do contrato for violada.
    """
    bronze = raw.rename(columns=to_snake_case)
    return BRONZE_SCHEMA.validate(bronze, lazy=True)


def run(raw_path: Path = RAW_CSV, bronze_path: Path = BRONZE_PARQUET) -> pd.DataFrame:
    """Executa raw -> bronze e grava o resultado em Parquet.

    Args:
        raw_path: CSV de entrada.
        bronze_path: Parquet de saída.

    Returns:
        DataFrame bronze gravado.
    """
    bronze = to_bronze(read_raw(raw_path))
    bronze_path.parent.mkdir(parents=True, exist_ok=True)
    bronze.to_parquet(bronze_path, index=False)
    logger.info("bronze: %d linhas x %d colunas -> %s", *bronze.shape, bronze_path)
    return bronze


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run()
