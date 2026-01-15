import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import roc_auc_score, brier_score_loss
import matplotlib.pyplot as plt

# -----------------------
# Load data
# -----------------------
train_df = pd.read_csv("data/processed/epl_train_70_v2.csv")
test_df  = pd.read_csv("data/processed/epl_test_30_v2.csv")

features = [
    "elo_diff_adv",
    "home_gd_last5","away_gd_last5","gd_diff_last5",
    "home_gd_last10","away_gd_last10","gd_diff_last10",
    "home_gf_last5","home_ga_last5","away_gf_last5","away_ga_last5",
    "home_gf_last10","home_ga_last10","away_gf_last10","away_ga_last10",
]

X_train = train_df[features]
y_train = train_df["home_win"].values

X_test = test_df[features]
y_test = test_df["home_win"].values

# -----------------------
# Base model (uncalibrated)
# -----------------------
base_rf = RandomForestClassifier(
    n_estimators=1200,
    max_depth=10,
    min_samples_leaf=6,
    random_state=42
)
base_rf.fit(X_train, y_train)

proba_raw = base_rf.predict_proba(X_test)[:, 1]

print("Raw RF ROC-AUC:", roc_auc_score(y_test, proba_raw))
print("Raw RF Brier:", brier_score_loss(y_test, proba_raw))

# -----------------------
# Platt scaling (sigmoid)
# NOTE: use estimator= (NOT base_estimator=)
# -----------------------
platt = CalibratedClassifierCV(
    estimator=RandomForestClassifier(
        n_estimators=1200,
        max_depth=10,
        min_samples_leaf=6,
        random_state=42
    ),
    method="sigmoid",
    cv=3
)
platt.fit(X_train, y_train)
proba_platt = platt.predict_proba(X_test)[:, 1]

print("\nPlatt ROC-AUC:", roc_auc_score(y_test, proba_platt))
print("Platt Brier:", brier_score_loss(y_test, proba_platt))

# -----------------------
# Isotonic calibration
# -----------------------
iso = CalibratedClassifierCV(
    estimator=RandomForestClassifier(
        n_estimators=1200,
        max_depth=10,
        min_samples_leaf=6,
        random_state=42
    ),
    method="isotonic",
    cv=3
)
iso.fit(X_train, y_train)
proba_iso = iso.predict_proba(X_test)[:, 1]

print("\nIsotonic ROC-AUC:", roc_auc_score(y_test, proba_iso))
print("Isotonic Brier:", brier_score_loss(y_test, proba_iso))

# -----------------------
# Save calibrated probs for Step E
# -----------------------
test_out = test_df.copy()
test_out["proba_home_raw"] = proba_raw
test_out["proba_home_platt"] = proba_platt
test_out["proba_home_iso"] = proba_iso

out_path = "data/processed/epl_test_with_calibrated_probs.csv"
test_out.to_csv(out_path, index=False)
print("\nSaved calibrated probabilities to:", out_path)

# -----------------------
# Reliability plot
# -----------------------
plt.figure(figsize=(7, 7))

for probs, label in [
    (proba_raw, "Uncalibrated"),
    (proba_platt, "Platt"),
    (proba_iso, "Isotonic"),
]:
    frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=10)
    plt.plot(mean_pred, frac_pos, marker="o", label=label)

plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("Predicted probability")
plt.ylabel("Observed frequency")
plt.title("Calibration curves (EPL Home Win)")
plt.legend()
plt.grid(True)
plt.show()
