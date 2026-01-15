import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Inputs
TRAIN_PATH = "data/processed/epl_train_70_v2.csv"
TEST_PATH  = "data/processed/epl_test_30_v2.csv"
OUT_PATH   = "data/processed/epl_test_predictions_with_tiers.csv"

features = [
    "elo_diff_adv",
    "home_gd_last5","away_gd_last5","gd_diff_last5",
    "home_gd_last10","away_gd_last10","gd_diff_last10",
    "home_gf_last5","home_ga_last5","away_gf_last5","away_ga_last5",
    "home_gf_last10","home_ga_last10","away_gf_last10","away_ga_last10",
]

train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

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
pred = (proba >= 0.5).astype(int)

# Confidence = distance from 0.5
conf = np.abs(proba - 0.5)

# Create tiers by percentile
p80 = np.quantile(conf, 0.80)  # top 20%
p70 = np.quantile(conf, 0.70)  # top 30%
p60 = np.quantile(conf, 0.60)  # top 40%

def tier(c):
    if c >= p80: return "A_top20"
    if c >= p70: return "B_top30"
    if c >= p60: return "C_top40"
    return "No_Pick"

out = test_df.copy()
out["proba_home"] = proba
out["pred_home_win"] = pred
out["confidence"] = conf
out["tier"] = [tier(c) for c in conf]

out.to_csv(OUT_PATH, index=False)
print("Saved:", OUT_PATH)

# Quick tier accuracy summary
for t in ["A_top20","B_top30","C_top40","No_Pick"]:
    subset = out[out["tier"] == t]
    if len(subset) == 0:
        continue
    acc = (subset["pred_home_win"].values == subset["home_win"].values).mean()
    print(t, "n=", len(subset), "acc=", round(acc, 3))
