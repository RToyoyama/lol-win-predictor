"""Constantes e caminhos centrais do projeto."""

import os
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

MLFLOW_DB: Final[Path] = PROJECT_ROOT / "mlflow.db"
MLFLOW_TRACKING_URI: Final[str] = f"sqlite:///{MLFLOW_DB}"
MLFLOW_ARTIFACTS_DIR: Final[Path] = MLRUNS_DIR
MLFLOW_EXPERIMENT: Final[str] = "lol-win-predictor"

SIDES: Final[tuple[str, str]] = ("blue", "red")
TEAM_STATS: Final[tuple[str, ...]] = (
    "wards_placed",
    "wards_destroyed",
    "kills",
    "assists",
    "dragons",
    "heralds",
    "towers_destroyed",
    "total_minions_killed",
    "total_jungle_minions_killed",
    "total_gold",
    "total_experience",
    "avg_level",
)
INPUT_COLUMNS: Final[tuple[str, ...]] = (
    "blue_first_blood",
    *(f"{side}_{stat}" for side in SIDES for stat in TEAM_STATS),
)
WARDS_CAP_QUANTILE: Final[float] = 0.99

MODEL_PATH: Final[Path] = Path(os.environ.get("LOL_WIN_MODEL_PATH", MODELS_DIR / "model.joblib"))
MODEL_METADATA_PATH: Final[Path] = MODEL_PATH.with_name("metadata.json")
FINAL_METRICS_PATH: Final[Path] = METRICS_DIR / "final.json"
MLFLOW_MODEL_NAME: Final[str] = "lol-win-predictor"
MLFLOW_MODEL_ALIAS: Final[str] = "champion"
AUC_LEAKAGE_ALARM: Final[float] = 0.85
