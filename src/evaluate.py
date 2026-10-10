"""Step 5 - Model Evaluation (Methodology).

Evaluates the saved Random Forest on the held-out test split using the
measures from the methodology: accuracy, precision, recall, F1-score and
the confusion matrix. Writes reports/metrics.json and a confusion-matrix
figure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from src.config import (
    CLASSES,
    FEATURES,
    FIGURES_DIR,
    METRICS_PATH,
    MODEL_PATH,
    TARGET,
    TEST_SPLIT_CSV,
    ensure_dirs,
)


def _plot_confusion_matrix(
    cm: list[list[int]], labels: list[str], out_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix - Random Forest")
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = cm[i][j]
            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                color="white" if value > max(map(max, cm)) / 2 else "black",
                fontsize=11,
            )
    fig.colorbar(im, ax=ax, shrink=0.8, label="Count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run(
    model_path: Path = MODEL_PATH,
    test_path: Path = TEST_SPLIT_CSV,
    metrics_path: Path = METRICS_PATH,
    figures_dir: Path = FIGURES_DIR,
) -> dict:
    """Evaluate the saved model on the held-out split (Step 5 entry point)."""
    ensure_dirs()
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run 'python -m src.train' first."
        )
    if not test_path.exists():
        raise FileNotFoundError(
            f"Test split not found at {test_path}. Run 'python -m src.train' first."
        )

    model = joblib.load(model_path)
    test = pd.read_csv(test_path)
    X = test[FEATURES]
    y = test[TARGET]
    y_pred = model.predict(X)

    # Report in the documented condition order.
    labels = [c for c in CLASSES if c in set(y) | set(y_pred)]
    report = classification_report(y, y_pred, labels=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y, y_pred, labels=labels)

    per_class = {}
    for label in labels:
        precision, recall, f1, support = precision_recall_fscore_support(
            y, y_pred, labels=[label], zero_division=0
        )
        per_class[label] = {
            "precision": round(float(precision[0]), 4),
            "recall": round(float(recall[0]), 4),
            "f1_score": round(float(f1[0]), 4),
            "support": int(support[0]),
        }

    metrics = {
        "model": "Random Forest Classifier",
        "test_samples": int(len(y)),
        "accuracy": round(float(accuracy_score(y, y_pred)), 4),
        "macro_precision": round(float(report["macro avg"]["precision"]), 4),
        "macro_recall": round(float(report["macro avg"]["recall"]), 4),
        "macro_f1": round(float(f1_score(y, y_pred, average="macro")), 4),
        "weighted_f1": round(float(report["weighted avg"]["f1-score"]), 4),
        "per_class": per_class,
        "confusion_matrix": {"labels": labels, "matrix": cm.tolist()},
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    cm_path = figures_dir / "confusion_matrix.png"
    _plot_confusion_matrix(cm.tolist(), labels, cm_path)

    print(f"\n[Step 5] Model evaluation on {len(y)} held-out samples")
    print(f"  accuracy : {metrics['accuracy']:.4f}")
    print(f"  macro precision/recall/F1 : {metrics['macro_precision']:.4f} / "
          f"{metrics['macro_recall']:.4f} / {metrics['macro_f1']:.4f}")
    print("  per-class:")
    for label in labels:
        m = per_class[label]
        print(f"    {label:<14} P={m['precision']:.3f} R={m['recall']:.3f} "
              f"F1={m['f1_score']:.3f} n={m['support']}")
    print(f"  saved metrics -> {metrics_path}")
    print(f"  saved figure  -> {cm_path}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 5: evaluate the trained model.")
    parser.add_argument("--model", default=MODEL_PATH, type=Path)
    parser.add_argument("--test", default=TEST_SPLIT_CSV, type=Path)
    args = parser.parse_args()
    run(args.model, args.test)


if __name__ == "__main__":
    main()
