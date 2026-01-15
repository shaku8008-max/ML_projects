import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

train_df = pd.read_csv("data/processed/epl_train_70.csv")
test_df = pd.read_csv("data/processed/epl_test_30.csv")

num_features = [
    "elo_diff_adv",
    "home_gf_last5", "home_ga_last5",
    "away_gf_last5", "away_ga_last5",
]

X_train = train_df[num_features]
y_train = train_df["home_win"]

X_test = test_df[num_features]
y_test = test_df["home_win"]

rf = RandomForestClassifier(
    n_estimators=800,
    max_depth=8,
    min_samples_leaf=8,
    random_state=42
)

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:")
print(classification_report(y_test, y_pred))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))
