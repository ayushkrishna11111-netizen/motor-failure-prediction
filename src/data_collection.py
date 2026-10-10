"""Step 1 - Data Collection (Methodology).

Generates a deterministic synthetic dataset of motor operating parameters:
temperature (deg C), vibration (mm/s) and rotational speed (RPM), with every
row labelled NORMAL / WARNING / FAILURE RISK.

Class-conditional distributions mimic real motor behaviour: healthy motors
run cool and smooth with stable speed, failing motors run hot, vibrate
heavily and lose speed. A fixed random seed keeps regeneration reproducible.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    CLASSES,
    FEATURES,
    N_SAMPLES,
    RANDOM_SEED,
    RAW_CSV,
    TARGET,
    ensure_dirs,
)

# Per-condition sampling profile: class weight and (mean, std) per feature.
PROFILES: dict[str, dict] = {
    "NORMAL": {
        "weight": 0.70,
        "temperature": (42.0, 4.0),
        "vibration": (1.2, 0.4),
        "speed": (1490.0, 25.0),
    },
    "WARNING": {
        "weight": 0.20,
        "temperature": (65.0, 6.0),
        "vibration": (3.6, 0.8),
        "speed": (1440.0, 45.0),
    },
    "FAILURE RISK": {
        "weight": 0.10,
        "temperature": (85.0, 7.0),
        "vibration": (6.5, 1.4),
        "speed": (1350.0, 70.0),
    },
}

# Physical sensor limits used to clip generated readings.
CLIPS: dict[str, tuple[float, float]] = {
    "temperature": (30.0, 115.0),
    "vibration": (0.1, 12.0),
    "speed": (1100.0, 1600.0),
}


def generate_dataset(n_samples: int = N_SAMPLES, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Return a synthetic motor dataset with `n_samples` rows."""
    rng = np.random.default_rng(seed)
    conditions = rng.choice(
        CLASSES, size=n_samples, p=[PROFILES[c]["weight"] for c in CLASSES]
    )

    columns: dict[str, np.ndarray] = {}
    for feature in FEATURES:
        values = np.empty(n_samples, dtype=float)
        for condition in CLASSES:
            mask = conditions == condition
            mean, std = PROFILES[condition][feature]
            values[mask] = rng.normal(mean, std, size=int(mask.sum()))
        low, high = CLIPS[feature]
        columns[feature] = np.clip(values, low, high)

    df = pd.DataFrame(columns)
    df["temperature"] = df["temperature"].round(1)
    df["vibration"] = df["vibration"].round(2)
    df["speed"] = df["speed"].round(0).astype(int)
    df[TARGET] = conditions
    return df[FEATURES + [TARGET]]


def run(
    n_samples: int = N_SAMPLES,
    seed: int = RANDOM_SEED,
    output: Path = RAW_CSV,
) -> pd.DataFrame:
    """Generate the dataset and write it to `output` (Step 1 entry point)."""
    ensure_dirs()
    df = generate_dataset(n_samples, seed)
    df.to_csv(output, index=False)
    print(f"\n[Step 1] Data collection: wrote {len(df)} rows -> {output}")
    print(df[TARGET].value_counts().reindex(CLASSES).to_string())
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 1: generate the raw motor dataset.")
    parser.add_argument("--samples", type=int, default=N_SAMPLES, help="number of rows")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="random seed")
    parser.add_argument("--output", type=Path, default=RAW_CSV, help="output CSV path")
    args = parser.parse_args()
    run(args.samples, args.seed, args.output)


if __name__ == "__main__":
    main()
