import pandas as pd

IN_PATH = "data/processed/epl_cl_2526_features_v2_elo_homeadv.csv"
OUT_PATH = "data/processed/epl_cl_2526_features_v3_form_elo_homeadv.csv"

df = pd.read_csv(IN_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# Initialize rolling feature columns
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

    home_hist = last_n(home, 5)
    away_hist = last_n(away, 5)

    if home_hist:
        df.at[idx, "home_gf_last5"] = sum(m["gf"] for m in home_hist) / len(home_hist)
        df.at[idx, "home_ga_last5"] = sum(m["ga"] for m in home_hist) / len(home_hist)

    if away_hist:
        df.at[idx, "away_gf_last5"] = sum(m["gf"] for m in away_hist) / len(away_hist)
        df.at[idx, "away_ga_last5"] = sum(m["ga"] for m in away_hist) / len(away_hist)

    # Update history after computing features (no leakage)
    team_history.setdefault(home, []).append({"gf": row["home_goals"], "ga": row["away_goals"]})
    team_history.setdefault(away, []).append({"gf": row["away_goals"], "ga": row["home_goals"]})

df.to_csv(OUT_PATH, index=False)
print(f"Saved: {OUT_PATH}")
print(df[["date","competition","home_team","away_team","home_gf_last5","away_gf_last5","elo_diff_adv","home_win"]].head(10))
