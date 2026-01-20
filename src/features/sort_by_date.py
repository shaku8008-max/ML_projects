import pandas as pd

df = pd.read_csv("data/processed/epl_cl_2526_labeled.csv")

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

df.to_csv(
    "data/processed/epl_cl_2526_labeled_sorted.csv",
    index=False
)

print(df.head())
print(df.tail())
