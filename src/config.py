"""Central configuration: paths, constants and seeds.

All pipeline steps (Methodology Steps 1-7) resolve their inputs and outputs
from here so that every module agrees on file locations and feature names.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_CSV = RAW_DIR / "motor_data.csv"
CLEAN_CSV = PROCESSED_DIR / "clean_data.csv"
TEST_SPLIT_CSV = PROCESSED_DIR / "test_split.csv"
MODEL_PATH = MODELS_DIR / "rf_model.joblib"
COMPARISON_PATH = MODELS_DIR / "model_comparison.json"
METRICS_PATH = REPORTS_DIR / "metrics.json"

# Feature contract shared by training, evaluation, CLI and dashboard.
FEATURES = ["temperature", "vibration", "speed"]
TARGET = "condition"
CLASSES = ["NORMAL", "WARNING", "FAILURE RISK"]
FEATURE_UNITS = {"temperature": "°C", "vibration": "mm/s", "speed": "RPM"}

RANDOM_SEED = 42
N_SAMPLES = 5000
TEST_SIZE = 0.2


def ensure_dirs() -> None:
    """Create every output directory the pipeline writes to."""
    for directory in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)
