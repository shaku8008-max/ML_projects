import pandas as pd

IN_PATH = "data/processed/epl_2526_elo_homeadv.csv"
OUT_PATH = "data/processed/epl_2526_features_form_elo_v2.csv"

df = pd.read_csv(IN_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# Initialize features
for col in [
    "home_gf_last5","home_ga_last5","away_gf_last5","away_ga_last5",
    "home_gf_last10","home_ga_last10","away_gf_last10","away_ga_last10",
]:
    df[col] = 0.0

team_history = {}

def last_n(team, n):
    return team_history.get(team, [])[-n:]

for idx, row in df.iterrows():
    home = row["home_team"]
    away = row["away_team"]

    h5 = last_n(home, 5)
    a5 = last_n(away, 5)
    h10 = last_n(home, 10)
    a10 = last_n(away, 10)

    def avg(hist, key):
        return sum(m[key] for m in hist) / len(hist) if hist else 0.0

    df.at[idx, "home_gf_last5"] = avg(h5, "gf")
    df.at[idx, "home_ga_last5"] = avg(h5, "ga")
    df.at[idx, "away_gf_last5"] = avg(a5, "gf")
    df.at[idx, "away_ga_last5"] = avg(a5, "ga")

    df.at[idx, "home_gf_last10"] = avg(h10, "gf")
    df.at[idx, "home_ga_last10"] = avg(h10, "ga")
    df.at[idx, "away_gf_last10"] = avg(a10, "gf")
    df.at[idx, "away_ga_last10"] = avg(a10, "ga")

    # Update history AFTER computing features (no leakage)
    team_history.setdefault(home, []).append({"gf": row["home_goals"], "ga": row["away_goals"]})
    team_history.setdefault(away, []).append({"gf": row["away_goals"], "ga": row["home_goals"]})

# Goal-difference features
df["home_gd_last5"] = df["home_gf_last5"] - df["home_ga_last5"]
df["away_gd_last5"] = df["away_gf_last5"] - df["away_ga_last5"]
df["gd_diff_last5"] = df["home_gd_last5"] - df["away_gd_last5"]

df["home_gd_last10"] = df["home_gf_last10"] - df["home_ga_last10"]
df["away_gd_last10"] = df["away_gf_last10"] - df["away_ga_last10"]
df["gd_diff_last10"] = df["home_gd_last10"] - df["away_gd_last10"]

df.to_csv(OUT_PATH, index=False)
print("Saved:", OUT_PATH, "rows=", len(df))
print(df[[
    "date","home_team","away_team","elo_diff_adv",
    "home_gd_last5","away_gd_last5","gd_diff_last5",
    "home_gd_last10","away_gd_last10","gd_diff_last10",
    "home_win"
]].head(10))
