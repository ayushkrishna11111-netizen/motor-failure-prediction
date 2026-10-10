"""Step 7 - Monitoring Interface (Methodology).

Streamlit dashboard showing live motor readings and the predicted motor
condition (NORMAL / WARNING / FAILURE RISK) with class probabilities.

Run from the project root:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

# Allow `src.*` imports when Streamlit runs this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd

from src.config import (
    CLASSES,
    FEATURES,
    FEATURE_UNITS,
    METRICS_PATH,
    MODEL_PATH,
)

CONDITION_STYLE = {
    "NORMAL": "success",
    "WARNING": "warning",
    "FAILURE RISK": "error",
}
PRESETS = {
    "Normal example (45 °C, 1.2 mm/s, 1490 RPM)": (45.0, 1.2, 1490),
    "Warning example (66 °C, 3.8 mm/s, 1430 RPM)": (66.0, 3.8, 1430),
    "Failure example (88 °C, 6.8 mm/s, 1320 RPM)": (88.0, 6.8, 1320),
}


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics() -> dict | None:
    if not METRICS_PATH.exists():
        return None
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def main() -> None:
    st.set_page_config(page_title="Motor Monitoring System", page_icon="⚙️")
    st.title("⚙️ Motor Monitoring System")
    st.caption(
        "Predictive maintenance dashboard - motor condition from "
        "temperature, vibration and speed (Methodology Step 7)."
    )

    model = load_model()
    if model is None:
        st.error(
            "Trained model not found. Run the pipeline first:\n\n"
            "`python -m src.pipeline`"
        )
        st.stop()

    metrics = load_metrics()
    if metrics:
        with st.sidebar:
            st.header("Model performance")
            st.metric("Test accuracy", f"{metrics['accuracy']:.2%}")
            st.metric("Macro F1", f"{metrics['macro_f1']:.2%}")
            st.caption(
                f"Held-out test samples: {metrics['test_samples']}  \n"
                "Full report: `reports/metrics.json`"
            )
            st.divider()
            st.caption(
                "Future extension: real-time readings via sensors + "
                "ESP32 → cloud (see Methodology §6)."
            )

    st.subheader("Sensor readings")
    preset = st.radio("Load a preset reading:", list(PRESETS), horizontal=True)
    default_temp, default_vib, default_speed = PRESETS[preset]

    col1, col2, col3 = st.columns(3)
    temperature = col1.number_input(
        f"Temperature ({FEATURE_UNITS['temperature']})",
        0.0, 150.0, default_temp, 0.5,
    )
    vibration = col2.number_input(
        f"Vibration ({FEATURE_UNITS['vibration']})",
        0.0, 25.0, default_vib, 0.1,
    )
    speed = col3.number_input(
        f"Speed ({FEATURE_UNITS['speed']})",
        500.0, 4000.0, float(default_speed), 10.0,
    )

    X = pd.DataFrame([[temperature, vibration, speed]], columns=FEATURES)
    label = str(model.predict(X)[0])
    probabilities = {
        str(cls): float(prob)
        for cls, prob in zip(model.classes_, model.predict_proba(X)[0])
    }

    st.subheader("Predicted condition")
    banner = (
        f"**Temperature:** {temperature:g} °C   |   "
        f"**Vibration:** {vibration:g} mm/s   |   "
        f"**Speed:** {speed:g} RPM"
    )
    st.write(banner)
    style = CONDITION_STYLE.get(label, "info")
    headline = {
        "NORMAL": "✅ NORMAL - motor operating within the learned normal pattern",
        "WARNING": "⚠️ WARNING - abnormal behaviour detected, monitor closely",
        "FAILURE RISK": "🛑 FAILURE RISK - high likelihood of failure, act now",
    }.get(label, label)
    getattr(st, style)(headline)

    st.subheader("Class probabilities")
    prob_df = pd.DataFrame(
        {"condition": CLASSES,
         "probability": [probabilities.get(c, 0.0) for c in CLASSES]}
    ).set_index("condition")
    st.bar_chart(prob_df)
    cols = st.columns(len(CLASSES))
    for col, cls in zip(cols, CLASSES):
        col.metric(cls, f"{probabilities.get(cls, 0.0):.1%}")


if __name__ == "__main__":
    main()
