import duckdb
import pandas as pd

X = pd.read_csv("data/secom_features.csv")
y = pd.read_csv("data/secom_labels.csv")

con = duckdb.connect("data/secom.duckdb")
con.execute("CREATE OR REPLACE TABLE features AS SELECT * FROM X")
con.execute("CREATE OR REPLACE TABLE labels AS SELECT * FROM y")

print(con.execute("SELECT COUNT(*) FROM features").fetchone())
print(con.execute("SELECT COUNT(*) FROM labels").fetchone())
