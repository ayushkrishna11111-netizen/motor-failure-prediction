"""Step 4 - Model Development (Methodology).

Splits the clean data into stratified training and testing sets, trains a
Random Forest classifier as the primary model, and compares it against
Decision Tree, Logistic Regression and SVM (the optional comparison models
named in the methodology). The fitted Random Forest is saved for
evaluation and prediction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.config import (
    CLEAN_CSV,
    COMPARISON_PATH,
    FEATURES,
    MODEL_PATH,
    RANDOM_SEED,
    TARGET,
    TEST_SIZE,
    TEST_SPLIT_CSV,
    ensure_dirs,
)


def build_candidates() -> dict:
    """The primary model plus the comparison models from the methodology."""
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_SEED, n_jobs=-1
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_SEED),
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_SEED)),
            ]
        ),
        "SVM (RBF)": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", random_state=RANDOM_SEED)),
            ]
        ),
    }


def run(clean_path: Path = CLEAN_CSV) -> dict:
    """Train all candidates and save the Random Forest (Step 4 entry point)."""
    ensure_dirs()
    if not clean_path.exists():
        raise FileNotFoundError(
            f"Clean data not found at {clean_path}. Run 'python -m src.preprocessing' first."
        )

    df = pd.read_csv(clean_path)
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    print(f"\n[Step 4] Model development")
    print(f"  train/test split: {len(X_train)} / {len(X_test)} rows (stratified, seed={RANDOM_SEED})")

    comparison: dict[str, dict] = {}
    fitted: dict = {}
    for name, model in build_candidates().items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        comparison[name] = {
            "accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "macro_f1": round(float(f1_score(y_test, pred, average="macro")), 4),
        }
        fitted[name] = model
        print(
            f"  {name:<20} accuracy={comparison[name]['accuracy']:.4f}  "
            f"macro_f1={comparison[name]['macro_f1']:.4f}"
        )

    # Persist the primary model and the held-out test split for evaluation.
    joblib.dump(fitted["Random Forest"], MODEL_PATH)
    COMPARISON_PATH.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    test_split = X_test.copy()
    test_split[TARGET] = y_test
    test_split.to_csv(TEST_SPLIT_CSV, index=False)

    print(f"  saved model      -> {MODEL_PATH}")
    print(f"  saved comparison -> {COMPARISON_PATH}")
    print(f"  saved test split -> {TEST_SPLIT_CSV}")
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 4: train the classification models.")
    parser.add_argument("--input", default=CLEAN_CSV, help="clean data CSV")
    args = parser.parse_args()
    run(Path(args.input))


if __name__ == "__main__":
    main()
