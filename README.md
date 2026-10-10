# Motor Failure Prediction Using Temperature, Vibration and Speed Data

Predictive maintenance for machines — a machine-learning system that reads motor
**temperature (°C)**, **vibration (mm/s)** and **speed (RPM)** and predicts the
operating condition: `NORMAL`, `WARNING` or `FAILURE RISK`.

Implements all seven steps of the methodology document
(`../Methodology_Motor_Failure_Prediction.docx`) end-to-end:

| Step | Activity | File | Output |
| --- | --- | --- | --- |
| 1. Data collection | Synthetic motor sensor data | `src/data_collection.py` | `data/raw/motor_data.csv` |
| 2. Preprocessing | Missing values, duplicates, range checks | `src/preprocessing.py` | `data/processed/clean_data.csv` |
| 3. Exploratory analysis | Distributions, correlation, scatter plots | `src/eda.py` | `reports/figures/*.png` |
| 4. Model development | Random Forest + 3 comparison models | `src/train.py` | `models/rf_model.joblib` |
| 5. Model evaluation | Accuracy, precision, recall, F1, confusion matrix | `src/evaluate.py` | `reports/metrics.json`, `reports/figures/confusion_matrix.png` |
| 6. Prediction | Single-reading classifier CLI | `src/predict.py` | terminal report |
| 7. Monitoring interface | Streamlit dashboard | `dashboard/app.py` | live web UI |

## Setup (once)

Requires Python 3.10+ (developed and verified on Python 3.14, Windows).

```powershell
# From this folder (motor-failure-prediction/)
python -m venv .venv
.venv\Scripts\Activate.ps1        # PowerShell; use `source .venv/Scripts/activate` in Git Bash
pip install -r requirements.txt
```

## Run the pipeline (Steps 1–5)

```powershell
python -m src.pipeline
```

This regenerates the dataset, cleans it, writes the EDA figures, trains all four
models and evaluates the Random Forest. Every run is reproducible (seed = 42).

Run steps individually if preferred:

```powershell
python -m src.data_collection     # Step 1
python -m src.preprocessing       # Step 2
python -m src.eda                 # Step 3
python -m src.train               # Step 4
python -m src.evaluate            # Step 5
```

## Predict a reading (Step 6)

```powershell
python -m src.predict --temperature 45 --vibration 1.2 --speed 1490
```

Example output (matches the project proposal):

```
       MOTOR MONITORING SYSTEM
Temperature : 45 °C
Vibration   : 1.2 mm/s
Speed       : 1490 RPM
Predicted Condition:
  NORMAL
Probabilities:
  NORMAL         100.0%
  WARNING          0.0%
  FAILURE RISK     0.0%
```

## Dashboard (Step 7)

```powershell
streamlit run dashboard/app.py
```

Opens at `http://localhost:8501` with input sliders, a colour-coded condition
banner and class-probability bars. Run the pipeline first so the model exists.

## Verified results (2026-10-10, seed 42)

- Dataset: 5000 rows generated → 4996 after cleaning (4 duplicates removed)
- Split: 3996 train / 1000 held-out test (stratified)
- **Random Forest: test accuracy 0.9980, macro F1 0.9949**
- Per-class F1: NORMAL 1.000, WARNING 0.995, FAILURE RISK 0.990
- Model comparison (accuracy): SVM (RBF) 0.9990 · Random Forest 0.9980 ·
  Logistic Regression 0.9980 · Decision Tree 0.9970
- Prediction checks: 45 °C / 1.2 / 1490 → `NORMAL`;
  66 °C / 3.8 / 1430 → `WARNING`; 88 °C / 6.8 / 1320 → `FAILURE RISK`
- Dashboard verified rendering and live re-prediction on port 8501

## Technologies

Python · Pandas · NumPy · Matplotlib · Scikit-learn · Joblib · Streamlit
(primary model: RandomForestClassifier; Decision Tree, Logistic Regression and
SVM trained for comparison, per the methodology).

## Future extensions (per methodology §6)

Real-time sensors → ESP32/Arduino → internet → cloud/ML model → dashboard →
alert; Remaining Useful Life (RUL) prediction, automatic notifications, cloud
storage and advanced anomaly detection.
