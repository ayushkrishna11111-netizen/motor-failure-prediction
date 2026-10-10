"""Step 6 - Prediction (Methodology).

Classifies a single set of motor readings using the saved Random Forest
model and prints the motor condition (NORMAL / WARNING / FAILURE RISK)
with class probabilities, in the MOTOR MONITORING SYSTEM format shown in
the project proposal.

Example:
    python -m src.predict --temperature 45 --vibration 1.2 --speed 1490
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

from src.config import CLASSES, FEATURES, FEATURE_UNITS, MODEL_PATH


def predict(
    temperature: float,
    vibration: float,
    speed: float,
    model_path: Path = MODEL_PATH,
) -> tuple[str, dict[str, float]]:
    """Return (predicted condition, {condition: probability}) for one reading."""
    if not model_path.exists():
        sys.exit(
            f"Model not found at {model_path}. Run 'python -m src.pipeline' first."
        )
    model = joblib.load(model_path)
    X = pd.DataFrame([[temperature, vibration, speed]], columns=FEATURES)
    label = str(model.predict(X)[0])
    probabilities = {
        str(cls): float(prob)
        for cls, prob in zip(model.classes_, model.predict_proba(X)[0])
    }
    return label, probabilities


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict motor condition from temperature, vibration and speed."
    )
    parser.add_argument("--temperature", type=float, required=True,
                        help="motor temperature in °C (e.g. 45)")
    parser.add_argument("--vibration", type=float, required=True,
                        help="vibration in mm/s (e.g. 1.2)")
    parser.add_argument("--speed", type=float, required=True,
                        help="rotational speed in RPM (e.g. 1490)")
    parser.add_argument("--model", type=Path, default=MODEL_PATH,
                        help="path to a trained model file")
    args = parser.parse_args()

    label, probabilities = predict(
        args.temperature, args.vibration, args.speed, args.model
    )

    print()
    print("       MOTOR MONITORING SYSTEM")
    print(f"Temperature : {args.temperature:g} {FEATURE_UNITS['temperature']}")
    print(f"Vibration   : {args.vibration:g} {FEATURE_UNITS['vibration']}")
    print(f"Speed       : {args.speed:g} {FEATURE_UNITS['speed']}")
    print("Predicted Condition:")
    print(f"  {label}")
    print("Probabilities:")
    for cls in CLASSES:
        prob = probabilities.get(cls, 0.0)
        print(f"  {cls:<14} {prob * 100:5.1f}%")
    print()


if __name__ == "__main__":
    main()
