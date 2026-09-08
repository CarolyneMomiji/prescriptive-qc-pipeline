import json
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict

app = FastAPI(title="Prescriptive QC Pipeline")

model = joblib.load("models/model.joblib")
with open("models/feature_cols.json") as f:
    FEATURE_COLS = json.load(f)
with open("models/run_id.txt") as f:
    run_id = f.read().strip()

ACTION_TEMPLATES = {
    "high_positive": "Sensor {sensor} reading is elevated and strongly associated with failures — flag for calibration check.",
    "high_negative": "Sensor {sensor} reading is low relative to passing runs — flag for inspection of related process step.",
}


class SensorReading(BaseModel):
    readings: Dict[str, float]  # e.g. {"0": 1.2, "1": 3.4, ...}


@app.get("/health")
def health():
    return {"status": "ok", "model_run_id": run_id, "n_features": len(FEATURE_COLS)}


@app.post("/predict")
def predict(payload: SensorReading):
    row = pd.DataFrame([payload.readings])
    row = row.reindex(columns=FEATURE_COLS)  # align + fill missing with NaN

    prediction = model.predict(row)[0]
    prob_fail = model.predict_proba(row)[0][1]

    imputed = model.named_steps["impute"].transform(row)
    scaled = model.named_steps["scale"].transform(imputed)
    coefs = model.named_steps["clf"].coef_[0]
    contributions = scaled[0] * coefs

    contrib_df = pd.DataFrame({
        "sensor": FEATURE_COLS,
        "contribution": contributions,
    }).sort_values("contribution", key=abs, ascending=False).head(5)

    recommendations = []
    for _, r in contrib_df.iterrows():
        key = "high_positive" if r["contribution"] > 0 else "high_negative"
        recommendations.append(ACTION_TEMPLATES[key].format(sensor=r["sensor"]))

    return {
        "prediction": "fail" if prediction == 1 else "pass",
        "fail_probability": round(float(prob_fail), 4),
        "recommendations": recommendations,
    }
