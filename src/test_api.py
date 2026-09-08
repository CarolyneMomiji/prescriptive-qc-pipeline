import pandas as pd
import requests
import json

X = pd.read_csv("data/secom_features.csv")
y = pd.read_csv("data/secom_labels.csv")["label"]

constant_cols = X.columns[X.nunique(dropna=True) <= 1]
missing_pct = X.isnull().mean()
high_missing_cols = X.columns[missing_pct > 0.5]
drop_cols = set(constant_cols) | set(high_missing_cols)
X_clean = X.drop(columns=list(drop_cols))

# Grab a known real failure case
fail_idx = y[y == 1].index[0]
row = X_clean.iloc[fail_idx]

# Build the payload, dropping any NaNs (the API will impute them same as training)
readings = row.dropna().to_dict()

response = requests.post(
    "http://127.0.0.1:8001/predict",
    json={"readings": readings},
)
print("Status:", response.status_code)
print(json.dumps(response.json(), indent=2))
