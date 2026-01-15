import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

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
y_train = train_df["home_win"]

X_test = test_df[features]
y_test = test_df["home_win"]

rf = RandomForestClassifier(
    n_estimators=1200,
    max_depth=10,
    min_samples_leaf=6,
    random_state=42
)

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))
print("\nClassification report:")
print(classification_report(y_test, y_pred))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))
