import pandas as pd

df = pd.read_csv("data/processed/epl_cl_2526_binary.csv")
df["date"] = pd.to_datetime(df["date"])

df = df.sort_values("date").reset_index(drop=True)

# Initialize feature columns
df["home_gf_last5"] = 0.0
df["home_ga_last5"] = 0.0
df["away_gf_last5"] = 0.0
df["away_ga_last5"] = 0.0

# Store past matches per team
team_history = {}

def get_last_n(team, n=5):
    return team_history.get(team, [])[-n:]

for idx, row in df.iterrows():
    home = row["home_team"]
    away = row["away_team"]

    home_hist = get_last_n(home)
    away_hist = get_last_n(away)

    if home_hist:
        df.at[idx, "home_gf_last5"] = sum(m["gf"] for m in home_hist) / len(home_hist)
        df.at[idx, "home_ga_last5"] = sum(m["ga"] for m in home_hist) / len(home_hist)

    if away_hist:
        df.at[idx, "away_gf_last5"] = sum(m["gf"] for m in away_hist) / len(away_hist)
        df.at[idx, "away_ga_last5"] = sum(m["ga"] for m in away_hist) / len(away_hist)

    # Update history AFTER feature calculation (no leakage)
    team_history.setdefault(home, []).append({
        "gf": row["home_goals"],
        "ga": row["away_goals"]
    })
    team_history.setdefault(away, []).append({
        "gf": row["away_goals"],
        "ga": row["home_goals"]
    })

df.to_csv(
    "data/processed/epl_cl_2526_features_v1.csv",
    index=False
)

print("Feature build complete.")
print(df.head(10))
