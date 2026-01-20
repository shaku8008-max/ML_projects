import difflib
import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = "models/btts_winner_rf.joblib"

FEATURE_FILE_CANDIDATES = [
    "data/processed/epl_2526_features_form_elo_v2_btts_winner.csv",
    "data/processed/epl_2526_features_form_elo_v2.csv",
]

FEATURES = [
    "elo_diff_adv",
    "home_gd_last5","away_gd_last5","gd_diff_last5",
    "home_gd_last10","away_gd_last10","gd_diff_last10",
    "home_gf_last5","home_ga_last5","away_gf_last5","away_ga_last5",
    "home_gf_last10","home_ga_last10","away_gf_last10","away_ga_last10",
]

def find_feature_file():
    for p in FEATURE_FILE_CANDIDATES:
        if Path(p).exists():
            return p
    raise FileNotFoundError(f"Could not find processed feature file. Looked for: {FEATURE_FILE_CANDIDATES}")

def normalize_team_name(user_input: str, valid_teams: list[str]) -> str:
    s = user_input.strip()
    lower_map = {t.lower(): t for t in valid_teams}
    if s.lower() in lower_map:
        return lower_map[s.lower()]
    matches = difflib.get_close_matches(s, valid_teams, n=1, cutoff=0.6)
    if matches:
        return matches[0]
    matches2 = difflib.get_close_matches(s.lower(), list(lower_map.keys()), n=1, cutoff=0.6)
    if matches2:
        return lower_map[matches2[0]]
    raise ValueError(f"Team '{user_input}' not found.")

def build_latest_team_state(df: pd.DataFrame) -> dict:
    df = df.sort_values("date").reset_index(drop=True)
    teams = pd.unique(pd.concat([df["home_team"], df["away_team"]], ignore_index=True))
    state = {}
    for team in teams:
        team_rows = df[(df["home_team"] == team) | (df["away_team"] == team)]
        if team_rows.empty:
            continue
        last = team_rows.iloc[-1]
        if last["home_team"] == team:
            state[team] = {
                "gf5": float(last["home_gf_last5"]),
                "ga5": float(last["home_ga_last5"]),
                "gf10": float(last["home_gf_last10"]),
                "ga10": float(last["home_ga_last10"]),
                "elo": float(last.get("elo_home", 1000.0)),
            }
        else:
            state[team] = {
                "gf5": float(last["away_gf_last5"]),
                "ga5": float(last["away_ga_last5"]),
                "gf10": float(last["away_gf_last10"]),
                "ga10": float(last["away_ga_last10"]),
                "elo": float(last.get("elo_away", 1000.0)),
            }
    return state

def make_feature_row(home: str, away: str, state: dict, home_adv: float = 75.0) -> pd.DataFrame:
    h = state[home]
    a = state[away]

    home_gf_last5, home_ga_last5 = h["gf5"], h["ga5"]
    away_gf_last5, away_ga_last5 = a["gf5"], a["ga5"]
    home_gf_last10, home_ga_last10 = h["gf10"], h["ga10"]
    away_gf_last10, away_ga_last10 = a["gf10"], a["ga10"]

    home_gd_last5 = home_gf_last5 - home_ga_last5
    away_gd_last5 = away_gf_last5 - away_ga_last5
    gd_diff_last5 = home_gd_last5 - away_gd_last5

    home_gd_last10 = home_gf_last10 - home_ga_last10
    away_gd_last10 = away_gf_last10 - away_ga_last10
    gd_diff_last10 = home_gd_last10 - away_gd_last10

    elo_diff_adv = (h["elo"] + home_adv) - a["elo"]

    row = {
        "elo_diff_adv": elo_diff_adv,
        "home_gd_last5": home_gd_last5,
        "away_gd_last5": away_gd_last5,
        "gd_diff_last5": gd_diff_last5,
        "home_gd_last10": home_gd_last10,
        "away_gd_last10": away_gd_last10,
        "gd_diff_last10": gd_diff_last10,
        "home_gf_last5": home_gf_last5,
        "home_ga_last5": home_ga_last5,
        "away_gf_last5": away_gf_last5,
        "away_ga_last5": away_ga_last5,
        "home_gf_last10": home_gf_last10,
        "home_ga_last10": home_ga_last10,
        "away_gf_last10": away_gf_last10,
        "away_ga_last10": away_ga_last10,
    }
    return pd.DataFrame([row])[FEATURES]

def main():
    feature_path = find_feature_file()
    df = pd.read_csv(feature_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    model = joblib.load(MODEL_PATH)

    state = build_latest_team_state(df)
    teams = sorted(state.keys())

    print("✅ EPL Predictor — Target: BTTS + Winner (both teams score AND not a draw)")
    print("Type team names naturally; it will auto-match.\n")

    while True:
        home_in = input("Home team (or 'q' to quit): ").strip()
        if home_in.lower() == "q":
            break
        away_in = input("Away team: ").strip()

        try:
            home = normalize_team_name(home_in, teams)
            away = normalize_team_name(away_in, teams)

            x = make_feature_row(home, away, state)
            p = float(model.predict_proba(x)[:, 1][0])

            print("\n--- Prediction ---")
            print(f"{home} vs {away}")
            print(f"P(BTTS + Winner): {p:.3f}")
            print("Meaning: probability that BOTH teams score AND the match has a winner.\n")

        except Exception as e:
            print("Error:", e, "\n")

if __name__ == "__main__":
    main()