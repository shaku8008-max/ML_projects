import pandas as pd

IN_PATH = "data/processed/epl_2526_only.csv"
OUT_PATH = "data/processed/epl_2526_binary.csv"

df = pd.read_csv(IN_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

df["home_win"] = (df["home_goals"] > df["away_goals"]).astype(int)

df.to_csv(OUT_PATH, index=False)
print("Saved:", OUT_PATH)
print(df["home_win"].value_counts())
print(df.head())
