import difflib
import subprocess
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# Config
# -----------------------------
# This is your "one button" pipeline
UPDATE_CMD = "python3 src/app/run_update.py"

# Try these in order (because filenames vary across your scripts)
FEATURE_FILE_CANDIDATES = [
    "data/processed/epl_2526_features_form_elo_v2.csv",
    "data/processed/epl_2526_features_form_elo.csv",
    "data/processed/epl_features_form_elo_v2.csv",
    "data/processed/epl_features_form_elo.csv",
]

# Model settings (same as your v2 RF baseline)
RF_PARAMS = dict(
    n_estimators=1200,
    max_depth=10,
    min_samples_leaf=6,
    random_state=42,
)

# Features you used before
FEATURES = [
    "elo_diff_adv",
    "home_gd_last5","away_gd_last5","gd_diff_last5",
    "home_gd_last10","away_gd_last10","gd_diff_last10",
    "home_gf_last5","home_ga_last5","away_gf_last5","away_ga_last5",
    "home_gf_last10","home_ga_last10","away_gf_last10","away_ga_last10",
]

# If your Elo builder outputs these, we’ll use them for better team-state tracking
OPTIONAL_ELO_COLS = ["elo_home", "elo_away"]


# -----------------------------
# Helpers
# -----------------------------
def run(cmd: str):
    print("\n>>>", cmd)
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")

def find_feature_file() -> str:
    for p in FEATURE_FILE_CANDIDATES:
        if Path(p).exists():
            return p
    # if none matched, print what's in data/processed to help debugging
    processed = Path("data/processed")
    if processed.exists():
        files = sorted([x.name for x in processed.glob("*.csv")])
        raise FileNotFoundError(
            "Could not find a feature CSV in known locations.\n"
            f"Looked for: {FEATURE_FILE_CANDIDATES}\n"
            f"Found in data/processed: {files}"
        )
    raise FileNotFoundError("data/processed folder not found. Run updater first.")

def normalize_team_name(user_input: str, valid_teams: list[str]) -> str:
    """
    Map user input -> dataset team name.
    - exact match (case-insensitive)
    - fuzzy match (difflib)
    """
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

    raise ValueError(f"Team '{user_input}' not found. Try running: python3 src/tools/list_teams.py")

def confidence_tier(prob_home: float) -> str:
    # Confidence = distance from 0.5
    conf = abs(prob_home - 0.5)
    if conf >= 0.20:
        return "A_high"
    if conf >= 0.13:
        return "B_medium"
    if conf >= 0.08:
        return "C_low"
    return "No_Pick"

def build_latest_team_state(df: pd.DataFrame) -> dict:
    """
    For each team, capture latest known rolling form and Elo.
    Uses the most recent row in which the team appears.
    """
    df = df.sort_values("date").reset_index(drop=True)
    teams = pd.unique(pd.concat([df["home_team"], df["away_team"]], ignore_index=True))

    state = {}
    has_elo_cols = all(c in df.columns for c in OPTIONAL_ELO_COLS)

    for team in teams:
        team_rows = df[(df["home_team"] == team) | (df["away_team"] == team)]
        if team_rows.empty:
            continue
        last = team_rows.iloc[-1]

        if last["home_team"] == team:
            gf5, ga5 = last["home_gf_last5"], last["home_ga_last5"]
            gf10, ga10 = last["home_gf_last10"], last["home_ga_last10"]
            elo = float(last["elo_home"]) if has_elo_cols else None
        else:
            gf5, ga5 = last["away_gf_last5"], last["away_ga_last5"]
            gf10, ga10 = last["away_gf_last10"], last["away_ga_last10"]
            elo = float(last["elo_away"]) if has_elo_cols else None

        state[team] = {
            "gf5": float(gf5),
            "ga5": float(ga5),
            "gf10": float(gf10),
            "ga10": float(ga10),
            "elo": elo,
        }

    return state

def make_feature_row(home: str, away: str, state: dict, home_adv: float = 75.0) -> pd.DataFrame:
    h = state[home]
    a = state[away]

    # If Elo is missing from file, we still can predict using form.
    # But elo_diff_adv is required by model, so default to 1000 baseline if missing.
    h_elo = h["elo"] if h["elo"] is not None else 1000.0
    a_elo = a["elo"] if a["elo"] is not None else 1000.0

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

    elo_diff_adv = (h_elo + home_adv) - a_elo

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


# -----------------------------
# Main
# -----------------------------
def main():
    # Auto-update every time you run predict_cli
    try:
        print("🔄 Updating EPL data + rebuilding features...")
        run(UPDATE_CMD)
    except Exception as e:
        print("\n⚠️ Update step failed. I will try using existing processed files.")
        print("Reason:", e)

    # Load latest features
    feature_path = find_feature_file()
    df = pd.read_csv(feature_path)

    # basic validation
    if "date" not in df.columns:
        raise ValueError(f"{feature_path} has no 'date' column. Check your pipeline output.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # Ensure required columns exist
    missing = [c for c in FEATURES + ["home_team", "away_team", "home_win"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {feature_path}: {missing}")

    # Train model (fast for your dataset size)
    X = df[FEATURES]
    y = df["home_win"].astype(int).values

    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X, y)

    # Build team state
    state = build_latest_team_state(df)
    valid_teams = sorted(state.keys())

    print("\n✅ EPL Predictor (binary: Home Win vs Away/Draw)")
    print("Type team names naturally; it will auto-match to dataset names.\n")

    while True:
        home_in = input("Home team (or 'q' to quit): ").strip()
        if home_in.lower() == "q":
            break
        away_in = input("Away team: ").strip()

        try:
            home = normalize_team_name(home_in, valid_teams)
            away = normalize_team_name(away_in, valid_teams)

            print(f"Matched: {home} vs {away}")

            x_row = make_feature_row(home, away, state, home_adv=75.0)
            prob_home = float(model.predict_proba(x_row)[:, 1][0])
            prob_away_or_draw = 1.0 - prob_home

            tier = confidence_tier(prob_home)
            pick = "Home Win" if prob_home >= 0.5 else "Away/Draw"

            print("\n--- Prediction ---")
            print(f"Match: {home} vs {away}")
            print(f"P(Home Win):     {prob_home:.3f}")
            print(f"P(Away/Draw):    {prob_away_or_draw:.3f}")
            print(f"Pick (binary):   {pick}")
            print(f"Confidence tier: {tier}\n")

        except Exception as e:
            print("\nError:", e)
            print("Tip: check exact names with the teams in your dataset.\n")

if __name__ == "__main__":
    main()
