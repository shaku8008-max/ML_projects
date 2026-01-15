import pandas as pd

IN_PATH = "data/processed/epl_cl_2526_binary.csv"
OUT_PATH = "data/processed/epl_cl_2526_features_v2_elo_homeadv.csv"

START_ELO = 1000
K = 20
HOME_ADV = 75  # football home advantage in Elo points (common range: 50-100)

df = pd.read_csv(IN_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

elo = {}

def get_elo(team: str) -> float:
    return float(elo.get(team, START_ELO))

def expected_score(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))

# Feature columns (pre-match)
df["elo_home"] = 0.0
df["elo_away"] = 0.0
df["elo_diff"] = 0.0

df["elo_home_adv"] = 0.0
df["elo_diff_adv"] = 0.0

for i, row in df.iterrows():
    home = row["home_team"]
    away = row["away_team"]

    r_home = get_elo(home)
    r_away = get_elo(away)

    # Save pre-match ratings as features (NO leakage)
    df.at[i, "elo_home"] = r_home
    df.at[i, "elo_away"] = r_away
    df.at[i, "elo_diff"] = r_home - r_away

    # Apply home advantage ONLY for expectation + derived features
    r_home_adv = r_home + HOME_ADV
    df.at[i, "elo_home_adv"] = r_home_adv
    df.at[i, "elo_diff_adv"] = r_home_adv - r_away

    # Actual match score for Elo updates
    if row["home_goals"] > row["away_goals"]:
        s_home, s_away = 1.0, 0.0
    elif row["home_goals"] < row["away_goals"]:
        s_home, s_away = 0.0, 1.0
    else:
        s_home, s_away = 0.5, 0.5

    # Expected scores (use home-advantaged rating for home team)
    e_home = expected_score(r_home_adv, r_away)
    e_away = 1.0 - e_home

    # Update ratings AFTER match (no leakage)
    elo[home] = r_home + K * (s_home - e_home)
    elo[away] = r_away + K * (s_away - e_away)

df.to_csv(OUT_PATH, index=False)

print(f"Saved: {OUT_PATH}")
print(df[["date", "competition", "home_team", "away_team", "elo_home", "elo_away", "elo_home_adv", "elo_diff_adv", "home_win"]].head(10))
