import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from pathlib import Path
import joblib

DATA_PATH = "data/processed/epl_2526_features_form_elo_v2_btts_winner.csv"
MODEL_OUT = "models/btts_winner_rf.joblib"

FEATURES = [
    "elo_diff_adv",
    "home_gd_last5","away_gd_last5","gd_diff_last5",
    "home_gd_last10","away_gd_last10","gd_diff_last10",
    "home_gf_last5","home_ga_last5","away_gf_last5","away_ga_last5",
    "home_gf_last10","home_ga_last10","away_gf_last10","away_ga_last10",
]

def main():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    y = df["btts_and_winner"].astype(int)
    X = df[FEATURES]

    split_idx = int(len(df) * 0.70)
    X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
    X_test, y_test = X.iloc[split_idx:], y.iloc[split_idx:]

    model = RandomForestClassifier(
        n_estimators=1200,
        max_depth=10,
        min_samples_leaf=6,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    print("ROC-AUC:", roc_auc_score(y_test, proba))
    print(classification_report(y_test, pred, digits=3))
    print(confusion_matrix(y_test, pred))

    Path("models").mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_OUT)
    print(f"✅ Saved model: {MODEL_OUT}")

if __name__ == "__main__":
    main()
