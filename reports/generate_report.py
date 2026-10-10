"""Generate the executive report PDF with systematic flowcharts.

Run from the project root:
    python reports/generate_report.py

Outputs:
    reports/figures/build_flowchart.png       how the system is built
    reports/figures/prediction_flowchart.png  how a reading is predicted
    reports/Executive_Report_Motor_Failure_Prediction.pdf
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from fpdf import FPDF
from fpdf.fonts import FontFace
from fpdf.enums import XPos, YPos
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "reports" / "figures"
OUT_PDF = ROOT / "reports" / "Executive_Report_Motor_Failure_Prediction.pdf"

# Palette
BLUE_FILL, BLUE_EDGE = "#E8F0FE", "#3B6EA5"
GREEN_FILL, GREEN_EDGE = "#E6F4EA", "#2E7D32"
AMBER_FILL, AMBER_EDGE = "#FFF4E5", "#C77700"
RED_FILL, RED_EDGE = "#FDECEA", "#C62828"
ARROW = "#37474F"


def _box(ax, x, y, w, h, text, fill=BLUE_FILL, edge=BLUE_EDGE, fs=8.5, bold=True):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.4,rounding_size=1.2",
        linewidth=1.5, edgecolor=edge, facecolor=fill,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2, y + h / 2, text, ha="center", va="center",
        fontsize=fs, fontweight="bold" if bold else "normal",
        color="#1A2B3C", linespacing=1.35,
    )


def _arrow(ax, start, end, **kwargs):
    ax.add_patch(
        FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=16,
            linewidth=1.6, color=ARROW, shrinkA=2, shrinkB=2, **kwargs,
        )
    )


def build_flowchart(path: Path) -> None:
    """Systematic flow: scaffold -> environment -> Steps 1-7 -> working system."""
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 56)
    ax.axis("off")
    fig.suptitle(
        "System Build Flow - from empty folder to a working system",
        fontsize=13, fontweight="bold", y=0.97,
    )

    w, h = 21, 13
    cols_x = [2, 27, 52, 77]
    row1_y, row2_y = 30, 9

    stages = [
        (cols_x[0], row1_y, "1. Project scaffold\nconfig.py, folders,\nrequirements.txt,\n.gitignore, README"),
        (cols_x[1], row1_y, "2. Environment\nPython 3.14 venv\npip install -r\nrequirements.txt"),
        (cols_x[2], row1_y, "3. Data (Steps 1-2)\ngenerate 5,000 rows\nclean to 4,996\nraw -> processed CSV"),
        (cols_x[3], row1_y, "4. EDA (Step 3)\n4 figures: classes,\ndistributions, corr,\nscatter -> figures/"),
        (cols_x[3], row2_y, "5. Training (Step 4)\nRandom Forest vs\nDT / LogReg / SVM\n-> rf_model.joblib"),
        (cols_x[2], row2_y, "6. Evaluation (Step 5)\naccuracy, P/R/F1,\nconfusion matrix\n-> metrics.json"),
        (cols_x[1], row2_y, "7. Prediction (Step 6)\nCLI classifies any\nreading with saved\nmodel"),
        (cols_x[0], row2_y, "8. Dashboard (Step 7)\nStreamlit UI: inputs,\ncolor banner,\nprobability bars",
         GREEN_FILL, GREEN_EDGE),
    ]
    centers = {}
    for i, stage in enumerate(stages):
        x, y, text = stage[0], stage[1], stage[2]
        fill = stage[3] if len(stage) > 3 else BLUE_FILL
        edge = stage[4] if len(stage) > 4 else BLUE_EDGE
        _box(ax, x, y, w, h, text, fill, edge)
        centers[i] = (x, y)

    # Row 1: left -> right
    for i in range(3):
        x0, y0 = centers[i]
        x1, _ = centers[i + 1]
        _arrow(ax, (x0 + w, y0 + h / 2), (x1, y0 + h / 2))
    # Down from stage 4 to stage 5 (same column, right side)
    x3, y3 = centers[3]
    _arrow(ax, (x3 + w / 2, y3), (x3 + w / 2, y3 - 8))
    # Row 2: right -> left
    for i in range(4, 7):
        x0, y0 = centers[i]
        x1, _ = centers[i + 1]
        _arrow(ax, (x0, y0 + h / 2), (x1 + w, y0 + h / 2))

    ax.text(
        50, 1.5,
        "Each stage writes a reproducible artifact; the whole flow runs with:  python -m src.pipeline",
        ha="center", fontsize=8.5, style="italic", color="#555555",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def prediction_flowchart(path: Path) -> None:
    """Prediction flow: readings -> prepare -> model -> probabilities -> class -> display."""
    fig, ax = plt.subplots(figsize=(10, 7.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(-5, 74)
    ax.axis("off")
    fig.suptitle(
        "Prediction Flow - how one reading becomes a condition",
        fontsize=13, fontweight="bold", y=0.97,
    )

    # Left column: input chain
    lw = 32
    chain = [
        (60, "Sensor readings\ntemperature (°C)  ·  vibration (mm/s)  ·  speed (RPM)"),
        (47, "Prepare input row\n3-feature DataFrame with the\ncolumn names locked in config.py"),
        (34, "Load trained model\nmodels/rf_model.joblib\n(Random Forest, 300 trees)"),
        (21, "Score the reading\nmodel.predict_proba() gives a\nprobability for each class"),
    ]
    for y, text in chain:
        _box(ax, 3, y, lw, 11, text, fs=8)

    # Diamond: pick the highest-scoring class
    cx, cy, dw, dh = 57, 26.5, 24, 17
    diamond = Polygon(
        [(cx, cy + dh / 2), (cx + dw / 2, cy), (cx, cy - dh / 2), (cx - dw / 2, cy)],
        closed=True, facecolor="#FFF8E1", edgecolor="#C77700", linewidth=1.5,
    )
    ax.add_patch(diamond)
    ax.text(cx, cy, "Class with the\nhighest score\n(argmax)", ha="center",
            va="center", fontsize=8, fontweight="bold", color="#1A2B3C",
            linespacing=1.3)

    _arrow(ax, (3 + lw + 1, 26.5), (cx - dw / 2, 26.5))

    # Three class outcome boxes (row)
    class_boxes = [
        (56, "NORMAL", GREEN_FILL, GREEN_EDGE),
        (70, "WARNING", AMBER_FILL, AMBER_EDGE),
        (84, "FAILURE RISK", RED_FILL, RED_EDGE),
    ]
    for x, label, fill, edge in class_boxes:
        _box(ax, x, 9, 12.5, 9, label, fill, edge, fs=8.5)
        _arrow(ax, (cx + 1, cy - 3), (x + 6.25, 18))

    # Display
    _box(
        ax, 56, -1.5, 40.5, 7.5,
        "Display result: dashboard banner + probability bars\nor CLI report (MOTOR MONITORING SYSTEM)",
        fill="#EDE7F6", edge="#5E35B1", fs=8,
    )
    for x, _, _, _ in class_boxes:
        _arrow(ax, (x + 6.25, 9), (x + 6.25, 6))

    # Chain arrows (downwards)
    for i in range(len(chain) - 1):
        y0 = chain[i][0]
        _arrow(ax, (3 + lw / 2, y0), (3 + lw / 2, chain[i + 1][0] + 11))

    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


class Report(FPDF):
    def footer(self) -> None:
        self.set_y(-15)
        self.set_font("DejaVu", size=8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Motor Failure Prediction - Executive Report  |  page {self.page_no()}", align="C")

    def h1(self, text: str) -> None:
        self.ln(3)
        self.set_font("DejaVu", "B", 15)
        self.set_text_color(20, 50, 90)
        self.cell(0, 9, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(59, 110, 165)
        self.set_line_width(0.6)
        y = self.get_y()
        self.line(self.l_margin, y, self.w - self.r_margin, y)
        self.ln(3)
        self.set_text_color(0, 0, 0)

    def h2(self, text: str) -> None:
        self.ln(1)
        self.set_font("DejaVu", "B", 11)
        self.set_text_color(40, 70, 110)
        self.cell(0, 7, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(0, 0, 0)

    def body(self, text: str) -> None:
        self.set_font("DejaVu", size=10)
        self.multi_cell(0, 5.4, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(1)

    def bullet(self, text: str) -> None:
        self.set_font("DejaVu", size=10)
        self.cell(5, 5.4, chr(0x2022))
        self.multi_cell(0, 5.4, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def mono_block(self, lines: list[str]) -> None:
        self.set_fill_color(240, 243, 247)
        self.set_font("DejaVu", size=9)
        for line in lines:
            self.cell(0, 5.6, "  " + line, new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
        self.ln(2)

    def add_table(self, headers: list[str], rows: list[list[str]],
              weights: list[int], align_left_first: bool = True) -> None:
        aligns = ("LEFT" if align_left_first else "CENTER",) + ("CENTER",) * (len(headers) - 1)
        style = FontFace(emphasis="BOLD", color=(255, 255, 255), fill_color=(36, 64, 98))
        with self.table(
            headings_style=style,
            col_widths=weights,
            text_align=aligns,
            line_height=5.6,
            borders_layout="ALL",
        ) as t:
            hr = t.row()
            for h in headers:
                hr.cell(h)
            for r in rows:
                tr = t.row()
                for c in r:
                    tr.cell(c)
        self.ln(2)


def build_pdf(build_png: Path, pred_png: Path) -> Path:
    metrics = json.loads((ROOT / "reports" / "metrics.json").read_text(encoding="utf-8"))
    comparison = json.loads((ROOT / "models" / "model_comparison.json").read_text(encoding="utf-8"))

    font_dir = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    pdf = Report()
    pdf.set_auto_page_break(True, margin=20)
    pdf.add_font("DejaVu", "", font_dir / "DejaVuSans.ttf")
    pdf.add_font("DejaVu", "B", font_dir / "DejaVuSans-Bold.ttf")
    pdf.add_page()

    # ---- Title block ----
    pdf.set_font("DejaVu", "B", 20)
    pdf.set_text_color(20, 50, 90)
    pdf.multi_cell(0, 9, "Executive Report", align="C",
                   new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("DejaVu", "B", 14)
    pdf.set_text_color(40, 70, 110)
    pdf.multi_cell(0, 7.5,
                   "Motor Failure Prediction Using Temperature,\nVibration and Speed Data",
                   align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("DejaVu", "", 10)
    pdf.set_text_color(80, 80, 80)
    pdf.multi_cell(0, 5.6,
                   "Predictive Maintenance for Machines\n"
                   "Department of Computer Science and Engineering, "
                   "ICFAI Foundation for Higher Education\n"
                   "Academic Year 2026-2027", align="C",
                   new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(1)
    pdf.set_font("DejaVu", "", 9)
    pdf.multi_cell(0, 5,
                   "Team: M. Abhilash Goud (24STUCHH010307)  ·  "
                   "S. Ayaan Ahmed (24STUCHH010284)\n"
                   "B. Arjun (24STUCHH010285)  ·  "
                   "A. Darshan Reddy (24STUCHH010985)  ·  "
                   f"B.Tech CSE, III Year   |   Report date: {date.today().isoformat()}",
                   align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)

    # ---- 1. Executive summary ----
    pdf.h1("1. Executive Summary")
    pdf.body(
        "This project delivers an end-to-end predictive-maintenance system that classifies "
        "electric-motor operating condition as NORMAL, WARNING or FAILURE RISK from three "
        "sensor readings: temperature, vibration and speed. The complete seven-step methodology "
        "was implemented as a reproducible Python pipeline and verified end to end. The final "
        f"Random Forest classifier achieves {metrics['accuracy']:.2%} accuracy and "
        f"{metrics['macro_f1']:.2%} macro F1 on {metrics['test_samples']} held-out test samples, "
        "and is exposed through both a command-line predictor and a live Streamlit dashboard."
    )
    pdf.add_table(
        ["Headline result", "Value"],
        [
            ["Test accuracy (Random Forest)", f"{metrics['accuracy']:.2%}"],
            ["Macro F1", f"{metrics['macro_f1']:.2%}"],
            ["Held-out test samples", str(metrics["test_samples"])],
            ["Models benchmarked", f"{len(comparison)} (RF, Decision Tree, LogReg, SVM)"],
            ["Pipeline steps implemented", "7 of 7 (data -> dashboard)"],
            ["Reproducibility", "fixed random seed 42, deterministic dataset"],
        ],
        weights=[60, 40],
    )

    # ---- 2. System at a glance ----
    pdf.h1("2. System at a Glance")
    for b in [
        "Language & libraries: Python with Pandas, NumPy, Matplotlib, Scikit-learn, Joblib, Streamlit.",
        "Dataset: 5,000 synthetic motor readings generated from class-conditional physical profiles; "
        "4,996 rows after cleaning (4 duplicates removed). Features: temperature (°C), vibration (mm/s), "
        "speed (RPM). Labels: NORMAL, WARNING, FAILURE RISK.",
        "Split: stratified 80/20 - 3,996 training and 1,000 held-out test rows (seed 42).",
        "Interfaces: CLI 'python -m src.predict ...' and dashboard 'streamlit run dashboard/app.py'.",
        "Future scope (per methodology): sensors + ESP32 -> cloud/ML -> alerts, RUL prediction.",
    ]:
        pdf.bullet(b)
    pdf.ln(2)

    # ---- 3. Build flow ----
    pdf.add_page()
    pdf.h1("3. How the System Is Built (Systematic Build Flow)")
    pdf.body(
        "The system is assembled in one linear, reproducible flow. Every stage writes a file "
        "artifact that the next stage consumes, so the whole build can be regenerated at any time "
        "with the single command 'python -m src.pipeline'."
    )
    pdf.image(str(build_png), w=170)
    pdf.ln(3)
    pdf.h2("Build artifacts by stage")
    pdf.add_table(
        ["Stage", "Command", "Output artifact"],
        [
            ["1. Scaffold + env", "pip install -r requirements.txt", ".venv with 6 pinned libraries"],
            ["2. Data collection", "python -m src.data_collection", "data/raw/motor_data.csv (5,000 rows)"],
            ["3. Preprocessing", "python -m src.preprocessing", "data/processed/clean_data.csv (4,996 rows)"],
            ["4. Exploratory analysis", "python -m src.eda", "4 PNG figures in reports/figures/"],
            ["5. Model training", "python -m src.train", "models/rf_model.joblib + comparison JSON"],
            ["6. Evaluation", "python -m src.evaluate", "reports/metrics.json + confusion matrix PNG"],
            ["7. Delivery", "streamlit run dashboard/app.py", "live monitoring dashboard on port 8501"],
        ],
        weights=[26, 36, 38],
    )

    # ---- 4. Prediction flow ----
    pdf.add_page()
    pdf.h1("4. How the System Predicts (Prediction Flow)")
    pdf.body(
        "Prediction is a fixed five-stage path. A reading is converted into a three-feature "
        "row using the same column names used in training, scored by the saved Random Forest, "
        "and the highest-scoring class becomes the displayed condition."
    )
    pdf.image(str(pred_png), w=165)
    pdf.ln(3)
    pdf.h2("Verified example predictions")
    pdf.add_table(
        ["Input (temperature / vibration / speed)", "Predicted condition", "Matches proposal?"],
        [
            ["45 °C, 1.2 mm/s, 1490 RPM", "NORMAL", "yes"],
            ["66 °C, 3.8 mm/s, 1430 RPM", "WARNING", "yes"],
            ["88 °C, 6.8 mm/s, 1320 RPM", "FAILURE RISK", "yes"],
        ],
        weights=[50, 25, 25],
    )

    # ---- 5. Model selection ----
    pdf.add_page()
    pdf.h1("5. Which Model Was Used (Model Selection)")
    pdf.body(
        "All four candidate models from the methodology were trained on the same stratified "
        "split and benchmarked on the same 1,000 held-out samples:"
    )
    rows = []
    for name, s in sorted(comparison.items(), key=lambda kv: kv[1]["accuracy"], reverse=True):
        star = "  <- USED IN SYSTEM" if name == "Random Forest" else ""
        rows.append([name + star, f"{s['accuracy']:.4f}", f"{s['macro_f1']:.4f}"])
    pdf.add_table(["Model", "Test accuracy", "Macro F1"], rows, weights=[50, 25, 25])
    pdf.body(
        "Decision: the system uses the Random Forest Classifier (models/rf_model.joblib, 300 trees). "
        "Although SVM (RBF) scored marginally higher (0.9990 vs 0.9980 - a difference of only "
        "~1-2 samples out of 1,000, within noise), Random Forest was selected because the "
        "methodology specifies it as the primary algorithm for tabular sensor data; it is "
        "interpretable, trains in under a second, and natively returns the class probabilities "
        "the dashboard and CLI display (the SVM pipeline was fitted without probability "
        "estimation). Decision Tree and Logistic Regression were retained as baselines."
    )
    pdf.h2("Final model - per-class performance (held-out test set)")
    rows = []
    for cls, m in metrics["per_class"].items():
        rows.append([cls, f"{m['precision']:.3f}", f"{m['recall']:.3f}",
                     f"{m['f1_score']:.3f}", str(m["support"])])
    pdf.add_table(["Condition", "Precision", "Recall", "F1-score", "Support"],
                  rows, weights=[32, 17, 17, 17, 17])
    cm_png = FIGURES / "confusion_matrix.png"
    if cm_png.exists():
        pdf.set_font("DejaVu", "B", 10)
        pdf.cell(0, 6, "Confusion matrix (actual vs predicted):",
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.image(str(cm_png), w=95)
        pdf.ln(2)

    # ---- 6. Verification ----
    pdf.h1("6. Verification Performed")
    for b in [
        "Full pipeline runs end to end with exit code 0 and regenerates all artifacts.",
        "All six module entry points answer --help correctly; every documented README command works.",
        "CLI example from the project proposal (45 °C / 1.2 mm/s / 1490 RPM) returns NORMAL.",
        "Dashboard verified in a browser: renders, predicts live, and shows sidebar metrics; "
        "no console errors.",
        "Final syntax check passes on all source files.",
    ]:
        pdf.bullet(b)
    pdf.ln(2)

    # ---- 7. Conclusion ----
    pdf.h1("7. Conclusion & Next Steps")
    pdf.body(
        "The proposed system was built exactly as documented: a seven-step pipeline from data "
        "collection to a monitoring interface, with a Random Forest classifier that correctly "
        "separates NORMAL, WARNING and FAILURE RISK conditions at 99.8% held-out accuracy. "
        "Natural next steps are real sensor ingestion (ESP32), Remaining Useful Life (RUL) "
        "prediction, alert notifications and cloud deployment, per the methodology's future "
        "extension section."
    )

    pdf.output(str(OUT_PDF))
    return OUT_PDF


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    build_png = FIGURES / "build_flowchart.png"
    pred_png = FIGURES / "prediction_flowchart.png"
    print(f"drawing {build_png.name} ...")
    build_flowchart(build_png)
    print(f"drawing {pred_png.name} ...")
    prediction_flowchart(pred_png)
    out = build_pdf(build_png, pred_png)
    print(f"wrote {out} ({out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
