import pandas as pd

IN_PATH = "data/processed/epl_cl_2526_binary.csv"
OUT_PATH = "data/processed/epl_cl_2526_features_v2_elo.csv"

START_ELO = 1000
K = 20  # learning rate; 20 is a good starting point

df = pd.read_csv(IN_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# ELO store
elo = {}

def get_elo(team: str) -> float:
    return float(elo.get(team, START_ELO))

def expected_score(r_a: float, r_b: float) -> float:
    # expected score for A vs B
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))

# Add columns
df["elo_home"] = 0.0
df["elo_away"] = 0.0
df["elo_diff"] = 0.0

for i, row in df.iterrows():
    home = row["home_team"]
    away = row["away_team"]

    r_home = get_elo(home)
    r_away = get_elo(away)

    # store pre-match elos as features (NO leakage)
    df.at[i, "elo_home"] = r_home
    df.at[i, "elo_away"] = r_away
    df.at[i, "elo_diff"] = r_home - r_away

    # actual result as a score for ELO update
    if row["home_goals"] > row["away_goals"]:
        s_home, s_away = 1.0, 0.0
    elif row["home_goals"] < row["away_goals"]:
        s_home, s_away = 0.0, 1.0
    else:
        s_home, s_away = 0.5, 0.5

    e_home = expected_score(r_home, r_away)
    e_away = expected_score(r_away, r_home)

    # update after match
    elo[home] = r_home + K * (s_home - e_home)
    elo[away] = r_away + K * (s_away - e_away)

df.to_csv(OUT_PATH, index=False)
print(f"Saved ELO feature file: {OUT_PATH}")
print(df[["date", "competition", "home_team", "away_team", "elo_home", "elo_away", "elo_diff", "home_win"]].head(10))
