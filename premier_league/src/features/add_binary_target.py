import pandas as pd

df = pd.read_csv("data/processed/epl_cl_2526_labeled_sorted.csv")

df["home_win"] = (df["home_goals"] > df["away_goals"]).astype(int)

df.to_csv(
    "data/processed/epl_cl_2526_binary.csv",
    index=False
)

print(df["home_win"].value_counts())
print(df.head())
