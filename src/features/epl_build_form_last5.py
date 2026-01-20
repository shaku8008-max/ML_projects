import pandas as pd

IN_PATH = "data/processed/epl_2526_elo_homeadv.csv"
OUT_PATH = "data/processed/epl_2526_features_form_elo.csv"

df = pd.read_csv(IN_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

df["home_gf_last5"] = 0.0
df["home_ga_last5"] = 0.0
df["away_gf_last5"] = 0.0
df["away_ga_last5"] = 0.0

team_history = {}

def last_n(team, n=5):
    return team_history.get(team, [])[-n:]

for idx, row in df.iterrows():
    home = row["home_team"]
    away = row["away_team"]

    h = last_n(home)
    a = last_n(away)

    if h:
        df.at[idx, "home_gf_last5"] = sum(m["gf"] for m in h) / len(h)
        df.at[idx, "home_ga_last5"] = sum(m["ga"] for m in h) / len(h)
    if a:
        df.at[idx, "away_gf_last5"] = sum(m["gf"] for m in a) / len(a)
        df.at[idx, "away_ga_last5"] = sum(m["ga"] for m in a) / len(a)

    team_history.setdefault(home, []).append({"gf": row["home_goals"], "ga": row["away_goals"]})
    team_history.setdefault(away, []).append({"gf": row["away_goals"], "ga": row["home_goals"]})

df.to_csv(OUT_PATH, index=False)
print("Saved:", OUT_PATH)
print(df.head(10))
