import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, recall_score, f1_score

X = pd.read_csv("data/secom_features.csv")
y = pd.read_csv("data/secom_labels.csv")["label"]

# Drop constant columns and columns with >50% missing
constant_cols = X.columns[X.nunique(dropna=True) <= 1]
missing_pct = X.isnull().mean()
high_missing_cols = X.columns[missing_pct > 0.5]
drop_cols = set(constant_cols) | set(high_missing_cols)

X_clean = X.drop(columns=list(drop_cols))
print(f"Dropped {len(drop_cols)} columns ({len(constant_cols)} constant, "
      f"{len(high_missing_cols)} high-missing, overlap accounted for)")
print(f"Remaining features: {X_clean.shape[1]}")

X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y, test_size=0.2, random_state=42, stratify=y
)

pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
    ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
])

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("secom-defect-detection")

with mlflow.start_run(run_name="baseline-logreg"):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    recall_fail = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)

    print("\nClassification report:")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    mlflow.log_param("model_type", "logistic_regression")
    mlflow.log_param("n_features", X_clean.shape[1])
    mlflow.log_metric("recall_fail_class", recall_fail)
    mlflow.log_metric("f1_fail_class", f1)
    mlflow.sklearn.log_model(pipeline, "model", serialization_format="cloudpickle")

print("\nRun logged to mlflow.db")

from sklearn.ensemble import RandomForestClassifier

with mlflow.start_run(run_name="random-forest"):
    rf_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=300, class_weight="balanced_subsample",
            max_depth=10, random_state=42
        )),
    ])
    rf_pipeline.fit(X_train, y_train)
    y_pred_rf = rf_pipeline.predict(X_test)

    recall_fail_rf = recall_score(y_test, y_pred_rf, pos_label=1)
    f1_rf = f1_score(y_test, y_pred_rf, pos_label=1)

    print("\n--- Random Forest ---")
    print(classification_report(y_test, y_pred_rf))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred_rf))

    mlflow.log_param("model_type", "random_forest")
    mlflow.log_param("n_features", X_clean.shape[1])
    mlflow.log_metric("recall_fail_class", recall_fail_rf)
    mlflow.log_metric("f1_fail_class", f1_rf)
    mlflow.sklearn.log_model(rf_pipeline, "model", serialization_format="cloudpickle")

print("\nBoth runs logged to mlflow.db")
