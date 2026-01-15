import pandas as pd

# Load feature dataset
df = pd.read_csv("data/processed/epl_cl_2526_features_v1.csv")

# Ensure date is datetime and sorted
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# Compute split index
split_idx = int(len(df) * 0.70)

train_df = df.iloc[:split_idx].copy()
test_df  = df.iloc[split_idx:].copy()

# Save outputs
train_df.to_csv(
    "data/processed/train_70.csv",
    index=False
)
test_df.to_csv(
    "data/processed/test_30.csv",
    index=False
)

print("Total matches:", len(df))
print("Train size:", len(train_df))
print("Test size:", len(test_df))
print("\nTrain date range:", train_df["date"].min(), "→", train_df["date"].max())
print("Test date range:", test_df["date"].min(), "→", test_df["date"].max())
