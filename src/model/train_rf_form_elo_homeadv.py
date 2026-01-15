import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

train_df = pd.read_csv("data/processed/train_70_v4.csv")
test_df  = pd.read_csv("data/processed/test_30_v4.csv")

# Features: emphasize DIFFERENCE features (best for trees)
num_features = [
    "elo_diff_adv",
    "home_gf_last5", "home_ga_last5",
    "away_gf_last5", "away_ga_last5",
]

cat_features = ["competition"]

X_train = train_df[num_features + cat_features]
y_train = train_df["home_win"]

X_test = test_df[num_features + cat_features]
y_test = test_df["home_win"]

preprocess = ColumnTransformer(
    transformers=[
        ("num", "passthrough", num_features),
        ("cat", OneHotEncoder(drop="first"), cat_features),
    ]
)

rf = RandomForestClassifier(
    n_estimators=800,
    max_depth=8,
    min_samples_leaf=8,
    random_state=42,
    class_weight=None
)

pipe = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", rf),
])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:")
print(classification_report(y_test, y_pred))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))
