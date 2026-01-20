import pandas as pd
from pathlib import Path

IN_PATH = "data/processed/epl_2526_features_form_elo_v2.csv"
OUT_PATH = "data/processed/epl_2526_features_form_elo_v2_btts_winner.csv"

def main():
    df = pd.read_csv(IN_PATH)

    # Must have final scores in the features file
    if not {"home_goals", "away_goals"}.issubset(df.columns):
        raise ValueError(
            f"{IN_PATH} must contain 'home_goals' and 'away_goals' columns. "
            "If your pipeline dropped them, we will adjust the feature builder output."
        )

    hg = df["home_goals"].astype(int)
    ag = df["away_goals"].astype(int)

    df["btts_and_winner"] = ((hg > 0) & (ag > 0) & (hg != ag)).astype(int)

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"✅ Saved: {OUT_PATH}")
    print("Positive rate:", df["btts_and_winner"].mean().round(3))
    print(df["btts_and_winner"].value_counts())

if __name__ == "__main__":
    main()
