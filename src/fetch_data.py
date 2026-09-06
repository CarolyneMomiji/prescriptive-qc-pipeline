from ucimlrepo import fetch_ucirepo
import pandas as pd

secom = fetch_ucirepo(id=179)
X = secom.data.features
y = secom.data.targets

print("Features shape:", X.shape)
print("Target shape:", y.shape)
print("\nTarget value counts:")
print(y.value_counts())

X.to_csv("data/secom_features.csv", index=False)
y.to_csv("data/secom_labels.csv", index=False)
print("\nSaved to data/secom_features.csv and data/secom_labels.csv")
