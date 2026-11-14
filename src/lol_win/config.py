"""Constantes e caminhos centrais do projeto."""

from pathlib import Path
from typing import Final

RANDOM_STATE: Final[int] = 42
TEST_SIZE: Final[float] = 0.2
CV_FOLDS: Final[int] = 5
TARGET: Final[str] = "blue_wins"

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
RAW_DIR: Final[Path] = DATA_DIR / "raw"
BRONZE_DIR: Final[Path] = DATA_DIR / "bronze"
SILVER_DIR: Final[Path] = DATA_DIR / "silver"
GOLD_DIR: Final[Path] = DATA_DIR / "gold"

RAW_CSV: Final[Path] = RAW_DIR / "Base_M43_Pratique_LOL_RANKED_WIN.csv"
BRONZE_PARQUET: Final[Path] = BRONZE_DIR / "matches.parquet"
SILVER_PARQUET: Final[Path] = SILVER_DIR / "matches.parquet"
GOLD_TRAIN: Final[Path] = GOLD_DIR / "train.parquet"
GOLD_TEST: Final[Path] = GOLD_DIR / "test.parquet"

MODELS_DIR: Final[Path] = PROJECT_ROOT / "models"
MLRUNS_DIR: Final[Path] = PROJECT_ROOT / "mlruns"
REPORTS_DIR: Final[Path] = PROJECT_ROOT / "reports"
FIGURES_DIR: Final[Path] = REPORTS_DIR / "figures"
METRICS_DIR: Final[Path] = REPORTS_DIR / "metrics"

MLFLOW_TRACKING_URI: Final[str] = MLRUNS_DIR.as_uri()
MLFLOW_EXPERIMENT: Final[str] = "lol-win-predictor"
