"""Step 2 - Data Preprocessing (Methodology).

Cleans the raw dataset: checks for missing values, removes duplicate and
out-of-range entries, keeps only the selected features (temperature,
vibration, speed) and the valid condition labels, and writes a clean CSV
ready for training.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import (
    CLASSES,
    CLEAN_CSV,
    FEATURES,
    RAW_CSV,
    TARGET,
    ensure_dirs,
)

# Plausible operating ranges; readings outside them are invalid entries.
VALID_RANGES: dict[str, tuple[float, float]] = {
    "temperature": (0.0, 150.0),
    "vibration": (0.0, 25.0),
    "speed": (500.0, 4000.0),
}


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return (clean dataframe, summary of every cleaning action)."""
    if missing_cols := [c for c in FEATURES + [TARGET] if c not in raw.columns]:
        raise ValueError(f"Raw data is missing required columns: {missing_cols}")

    summary: dict = {"rows_in": int(len(raw))}
    df = raw.copy()

    # Coerce numeric features; non-numeric entries become NaN.
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 1. Missing values: report them, fill numerics with the column median,
    #    and drop rows whose condition label is missing.
    missing_before = int(df[FEATURES + [TARGET]].isna().sum().sum())
    summary["missing_before"] = missing_before
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(str).str.strip().str.upper()
    df[FEATURES] = df[FEATURES].fillna(df[FEATURES].median(numeric_only=True))
    summary["missing_after"] = int(df[FEATURES + [TARGET]].isna().sum().sum())

    # 2. Incorrect entries: duplicate rows.
    before = len(df)
    df = df.drop_duplicates()
    summary["duplicates_removed"] = int(before - len(df))

    # 3. Unnecessary/invalid entries: labels outside the three conditions.
    valid_labels = df[TARGET].isin(CLASSES)
    summary["invalid_labels_removed"] = int((~valid_labels).sum())
    df = df[valid_labels]

    # 4. Out-of-range readings.
    in_range = pd.Series(True, index=df.index)
    for col, (low, high) in VALID_RANGES.items():
        in_range &= df[col].between(low, high)
    summary["out_of_range_removed"] = int((~in_range).sum())
    df = df[in_range]

    df = df[FEATURES + [TARGET]].reset_index(drop=True)
    summary["rows_out"] = int(len(df))
    summary["class_distribution"] = {
        cls: int((df[TARGET] == cls).sum()) for cls in CLASSES
    }
    return df, summary


def run(raw_path: Path = RAW_CSV, clean_path: Path = CLEAN_CSV) -> dict:
    """Load raw CSV, clean it, write the clean CSV (Step 2 entry point)."""
    ensure_dirs()
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {raw_path}. Run 'python -m src.data_collection' first."
        )
    raw = pd.read_csv(raw_path)
    df, summary = clean(raw)
    df.to_csv(clean_path, index=False)

    print(f"\n[Step 2] Preprocessing: {raw_path.name} -> {clean_path}")
    print(f"  rows in/out      : {summary['rows_in']} -> {summary['rows_out']}")
    print(f"  missing values   : {summary['missing_before']} before, {summary['missing_after']} after")
    print(f"  duplicates removed: {summary['duplicates_removed']}")
    print(f"  invalid labels removed: {summary['invalid_labels_removed']}")
    print(f"  out-of-range removed   : {summary['out_of_range_removed']}")
    for cls, count in summary["class_distribution"].items():
        print(f"  {cls:<14}: {count}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 2: clean the raw motor dataset.")
    parser.add_argument("--input", type=Path, default=RAW_CSV)
    parser.add_argument("--output", type=Path, default=CLEAN_CSV)
    args = parser.parse_args()
    run(args.input, args.output)


if __name__ == "__main__":
    main()
