"""Step 3 - Exploratory Data Analysis (Methodology).

Plots class balance, per-feature distributions split by condition, a
correlation heatmap and pairwise scatter plots so the relationships between
temperature, vibration, speed and motor condition are visible. All figures
are saved as PNGs in reports/figures/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe: render to files, not a GUI window

import matplotlib.pyplot as plt
import pandas as pd

from src.config import (
    CLASSES,
    CLEAN_CSV,
    FEATURES,
    FIGURES_DIR,
    FEATURE_UNITS,
    TARGET,
    ensure_dirs,
)

COLORS = {"NORMAL": "#2e7d32", "WARNING": "#f9a825", "FAILURE RISK": "#c62828"}


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")
    return path


def make_figures(df: pd.DataFrame, out_dir: Path = FIGURES_DIR) -> list[Path]:
    """Generate all EDA figures; returns the list of PNG paths."""
    ensure_dirs()
    paths: list[Path] = []

    # Figure 1: class distribution.
    counts = df[TARGET].value_counts().reindex(CLASSES).fillna(0)
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(counts.index, counts.values, color=[COLORS[c] for c in counts.index])
    ax.bar_label(bars, padding=3)
    ax.set_title("Class distribution of motor conditions")
    ax.set_ylabel("Number of readings")
    ax.set_ylim(0, counts.max() * 1.15)
    paths.append(_save(fig, out_dir / "class_distribution.png"))

    # Figure 2: per-feature distributions split by condition.
    fig, axes = plt.subplots(1, len(FEATURES), figsize=(15, 4))
    for ax, feature in zip(axes, FEATURES):
        for cls in CLASSES:
            subset = df.loc[df[TARGET] == cls, feature]
            ax.hist(subset, bins=30, alpha=0.55, density=True,
                    label=cls, color=COLORS[cls])
        ax.set_xlabel(f"{feature} ({FEATURE_UNITS[feature]})")
        ax.set_title(f"{feature} distribution by condition")
        ax.legend(fontsize=8)
    fig.suptitle("Parameter distributions for each motor condition", y=1.02)
    paths.append(_save(fig, out_dir / "feature_distributions.png"))

    # Figure 3: correlation heatmap (condition encoded as an ordinal code).
    encoded = df.copy()
    encoded["condition_code"] = pd.Categorical(
        encoded[TARGET], categories=CLASSES
    ).codes
    corr = encoded[FEATURES + ["condition_code"]].corr()

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    labels = FEATURES + ["condition"]
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}",
                    ha="center", va="center", color="black", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    ax.set_title("Correlation between parameters and condition\n(0=Normal, 1=Warning, 2=Failure Risk)")
    paths.append(_save(fig, out_dir / "correlation_heatmap.png"))

    # Figure 4: pairwise scatter plots coloured by condition.
    pairs = [(0, 1), (0, 2), (1, 2)]
    fig, axes = plt.subplots(1, len(pairs), figsize=(15, 4.5))
    for ax, (i, j) in zip(axes, pairs):
        for cls in CLASSES:
            subset = df[df[TARGET] == cls]
            ax.scatter(subset[FEATURES[i]], subset[FEATURES[j]],
                       s=8, alpha=0.5, color=COLORS[cls], label=cls)
        ax.set_xlabel(f"{FEATURES[i]} ({FEATURE_UNITS[FEATURES[i]]})")
        ax.set_ylabel(f"{FEATURES[j]} ({FEATURE_UNITS[FEATURES[j]]})")
        ax.set_title(f"{FEATURES[i]} vs {FEATURES[j]}")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_dir / "parameter_scatter.png", dpi=150)
    plt.close(fig)
    print(f"  saved {out_dir / 'parameter_scatter.png'}")
    paths.append(out_dir / "parameter_scatter.png")

    return paths


def run(clean_path: Path = CLEAN_CSV, out_dir: Path = FIGURES_DIR) -> list[Path]:
    """Load the clean data and produce all EDA figures (Step 3 entry point)."""
    if not clean_path.exists():
        raise FileNotFoundError(
            f"Clean data not found at {clean_path}. Run 'python -m src.preprocessing' first."
        )
    df = pd.read_csv(clean_path)
    print(f"\n[Step 3] Exploratory analysis on {clean_path}")
    paths = make_figures(df, out_dir)
    print(f"  {len(paths)} figures written to {out_dir}")
    return paths


def main() -> None:
    run()


if __name__ == "__main__":
    main()
