"""Runs the full methodology pipeline (Steps 1-5) in order.

    python -m src.pipeline

Step 1  Data collection      -> data/raw/motor_data.csv
Step 2  Preprocessing        -> data/processed/clean_data.csv
Step 3  Exploratory analysis -> reports/figures/*.png
Step 4  Model development    -> models/rf_model.joblib
Step 5  Model evaluation     -> reports/metrics.json

After it finishes, Step 6 (CLI) and Step 7 (dashboard) are available:
    python -m src.predict --temperature 45 --vibration 1.2 --speed 1490
    streamlit run dashboard/app.py
"""

from __future__ import annotations

from src import data_collection, eda, evaluate, preprocessing, train
from src.config import ensure_dirs


def main() -> None:
    ensure_dirs()
    print("=" * 62)
    print("Motor Failure Prediction - full pipeline (Steps 1-5)")
    print("=" * 62)

    data_collection.run()
    preprocessing.run()
    eda.run()
    comparison = train.run()
    metrics = evaluate.run()

    print("\n" + "=" * 62)
    print("Pipeline complete.")
    print(f"  Random Forest test accuracy : {metrics['accuracy']:.4f}")
    print(f"  Macro F1                    : {metrics['macro_f1']:.4f}")
    best = max(comparison, key=lambda name: comparison[name]["macro_f1"])
    print(f"  Best model by macro F1      : {best} "
          f"({comparison[best]['macro_f1']:.4f})")
    print("Next steps:")
    print("  - Predict    : python -m src.predict --temperature 45 "
          "--vibration 1.2 --speed 1490")
    print("  - Dashboard  : streamlit run dashboard/app.py")
    print("=" * 62)


if __name__ == "__main__":
    main()
