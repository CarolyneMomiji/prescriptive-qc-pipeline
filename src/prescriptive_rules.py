import pandas as pd
import numpy as np
import mlflow

X = pd.read_csv("data/secom_features.csv")
y = pd.read_csv("data/secom_labels.csv")["label"]

# Recreate the same cleaning as train_baseline.py
constant_cols = X.columns[X.nunique(dropna=True) <= 1]
missing_pct = X.isnull().mean()
high_missing_cols = X.columns[missing_pct > 0.5]
drop_cols = set(constant_cols) | set(high_missing_cols)
X_clean = X.drop(columns=list(drop_cols))

# Load the logistic regression run from MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name("secom-defect-detection")
runs = client.search_runs(
    experiment.experiment_id,
    filter_string="params.model_type = 'logistic_regression'",
    order_by=["start_time DESC"],
    max_results=1,
)
run_id = runs[0].info.run_id
print(f"Loading model from run: {run_id}")

model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

# The pipeline's classifier is the last step
clf = model.named_steps["clf"]
coefficients = clf.coef_[0]

feature_importance = pd.DataFrame({
    "sensor": X_clean.columns,
    "coefficient": coefficients,
}).sort_values("coefficient", key=abs, ascending=False)

print("\nTop 10 sensors driving a 'fail' prediction:")
print(feature_importance.head(10))

feature_importance.to_csv("data/feature_importance.csv", index=False)
print("\nSaved to data/feature_importance.csv")

# --- Recommendation engine ---

# Generic action templates keyed by contribution direction.
# In a real deployment, these would be populated by process engineers
# who know what each sensor physically measures. Here they're
# illustrative placeholders that demonstrate the mapping mechanism.
ACTION_TEMPLATES = {
    "high_positive": "Sensor {sensor} reading is elevated and strongly associated with failures — flag for calibration check.",
    "high_negative": "Sensor {sensor} reading is low relative to passing runs — flag for inspection of related process step.",
}

def generate_recommendation(row, pipeline, feature_cols, feature_importance, top_n=5):
    """
    row: a single-row DataFrame of raw (uncleaned) sensor readings
    pipeline: the fitted sklearn Pipeline (impute -> scale -> clf)
    feature_cols: the cleaned column list used at training time
    feature_importance: DataFrame with sensor/coefficient columns
    """
    row_clean = row[feature_cols]
    prediction = pipeline.predict(row_clean)[0]
    prob_fail = pipeline.predict_proba(row_clean)[0][1]

    # Transform the row the same way the pipeline does internally,
    # then multiply by coefficients to get each feature's contribution
    imputed = pipeline.named_steps["impute"].transform(row_clean)
    scaled = pipeline.named_steps["scale"].transform(imputed)
    coefs = pipeline.named_steps["clf"].coef_[0]
    contributions = scaled[0] * coefs

    contrib_df = pd.DataFrame({
        "sensor": feature_cols,
        "contribution": contributions,
    }).sort_values("contribution", key=abs, ascending=False).head(top_n)

    recommendations = []
    for _, r in contrib_df.iterrows():
        key = "high_positive" if r["contribution"] > 0 else "high_negative"
        recommendations.append(ACTION_TEMPLATES[key].format(sensor=r["sensor"]))

    return {
        "prediction": "fail" if prediction == 1 else "pass",
        "fail_probability": round(float(prob_fail), 4),
        "recommendations": recommendations,
    }


if __name__ == "__main__":
    # Quick manual test: grab one row known to be a real failure
    fail_idx = y[y == 1].index[0]
    test_row = X_clean.iloc[[fail_idx]]

    result = generate_recommendation(
        test_row, model, list(X_clean.columns), feature_importance
    )
    print("\n--- Test recommendation (known fail case) ---")
    print(result)
