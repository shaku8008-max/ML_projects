import pandas as pd

df = pd.read_csv("data/processed/epl_cl_2526_all_matches.csv")

def get_winner(row):
    if row["home_goals"] > row["away_goals"]:
        return "home"
    elif row["home_goals"] < row["away_goals"]:
        return "away"
    else:
        return "draw"

df["winner"] = df.apply(get_winner, axis=1)

df.to_csv(
    "data/processed/epl_cl_2526_labeled.csv",
    index=False
)

print(df["winner"].value_counts())
print(df.head())
