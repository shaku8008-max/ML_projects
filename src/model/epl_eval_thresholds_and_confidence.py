import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score
)

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

rf = RandomForestClassifier(
    n_estimators=1200,
    max_depth=10,
    min_samples_leaf=6,
    random_state=42
)
rf.fit(X_train, y_train)

proba = rf.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, proba)
print("ROC-AUC:", auc)

# ---- Threshold sweep
thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
print("\nThreshold sweep (positive = home win):")
print("thr | acc  prec rec  f1")
for t in thresholds:
    pred = (proba >= t).astype(int)
    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred, zero_division=0)
    rec = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)
    print(f"{t:>3.2f} | {acc:>4.2f} {prec:>4.2f} {rec:>4.2f} {f1:>4.2f}")

# ---- "Bet only when confident": choose top X% by confidence
# confidence = distance from 0.5 (more extreme probs are more confident)
confidence = np.abs(proba - 0.5)

for top_pct in [0.20, 0.30, 0.40]:
    k = int(len(proba) * top_pct)
    idx = np.argsort(confidence)[-k:]  # most confident
    proba_sel = proba[idx]
    y_sel = y_test[idx]

    pred_sel = (proba_sel >= 0.5).astype(int)
    acc_sel = accuracy_score(y_sel, pred_sel)
    prec_sel = precision_score(y_sel, pred_sel, zero_division=0)
    rec_sel = recall_score(y_sel, pred_sel, zero_division=0)
    print(f"\nTop {int(top_pct*100)}% most confident ({k} games):")
    print(f"Accuracy={acc_sel:.3f}  Precision={prec_sel:.3f}  Recall={rec_sel:.3f}")
