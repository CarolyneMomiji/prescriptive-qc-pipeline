import pandas as pd

X = pd.read_csv("data/secom_features.csv")

missing_pct = X.isnull().mean().sort_values(ascending=False) * 100

constant_cols = X.columns[X.nunique(dropna=True) <= 1]
print(f"\nConstant columns: {len(constant_cols)}")
print("Top 10 most-missing columns:")
print(missing_pct.head(10))

print(f"\nColumns with >50% missing: {(missing_pct > 50).sum()}")
print(f"Columns with any missing: {(missing_pct > 0).sum()} out of {X.shape[1]}")
print(f"Rows with any missing: {X.isnull().any(axis=1).sum()} out of {X.shape[0]}")
