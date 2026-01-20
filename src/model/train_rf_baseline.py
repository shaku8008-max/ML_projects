import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load data
train_df = pd.read_csv("data/processed/train_70.csv")
test_df  = pd.read_csv("data/processed/test_30.csv")

# Feature sets
num_features = [
    "home_gf_last5",
    "home_ga_last5",
    "away_gf_last5",
    "away_ga_last5",
]
cat_features = ["competition"]

X_train = train_df[num_features + cat_features]
y_train = train_df["home_win"]

X_test = test_df[num_features + cat_features]
y_test = test_df["home_win"]

# Preprocessing
preprocess = ColumnTransformer(
    transformers=[
        ("num", "passthrough", num_features),
        ("cat", OneHotEncoder(drop="first"), cat_features),
    ]
)

# Random Forest model
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=6,
    min_samples_leaf=10,
    random_state=42
)

# Pipeline
pipe = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", rf),
])

# Train
pipe.fit(X_train, y_train)

# Predict
y_pred = pipe.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:")
print(classification_report(y_test, y_pred))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))
