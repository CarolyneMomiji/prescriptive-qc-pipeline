import json
import joblib
import mlflow

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
model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

import pandas as pd
X_full = pd.read_csv("data/secom_features.csv")
constant_cols = X_full.columns[X_full.nunique(dropna=True) <= 1]
missing_pct = X_full.isnull().mean()
high_missing_cols = X_full.columns[missing_pct > 0.5]
drop_cols = set(constant_cols) | set(high_missing_cols)
feature_cols = [c for c in X_full.columns if c not in drop_cols]

joblib.dump(model, "models/model.joblib")
with open("models/feature_cols.json", "w") as f:
    json.dump(feature_cols, f)
with open("models/run_id.txt", "w") as f:
    f.write(run_id)

print(f"Exported model from run {run_id} to models/model.joblib")
print(f"Exported {len(feature_cols)} feature columns to models/feature_cols.json")
