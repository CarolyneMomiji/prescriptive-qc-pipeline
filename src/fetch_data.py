import requests
import zipfile
import io
import pandas as pd

url = "https://archive.ics.uci.edu/static/public/179/secom.zip"
response = requests.get(url)
response.raise_for_status()

zf = zipfile.ZipFile(io.BytesIO(response.content))
print("Files in archive:", zf.namelist())

# secom.data: whitespace-separated, 591 feature columns, no header
with zf.open("secom.data") as f:
    X = pd.read_csv(f, sep=r"\s+", header=None)

# secom_labels.data: label (-1=pass, 1=fail) + timestamp, whitespace-separated
with zf.open("secom_labels.data") as f:
    y = pd.read_csv(f, sep=r"\s+", header=None, names=["label", "timestamp"])

print("Features shape:", X.shape)
print("Labels shape:", y.shape)
print("\nLabel value counts:")
print(y["label"].value_counts())

X.to_csv("data/secom_features.csv", index=False)
y.to_csv("data/secom_labels.csv", index=False)
print("\nSaved to data/secom_features.csv and data/secom_labels.csv")
