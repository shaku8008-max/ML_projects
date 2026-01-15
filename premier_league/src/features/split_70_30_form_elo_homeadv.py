import pandas as pd

IN_PATH = "data/processed/epl_cl_2526_features_v3_form_elo_homeadv.csv"
TRAIN_OUT = "data/processed/train_70_v4.csv"
TEST_OUT = "data/processed/test_30_v4.csv"

df = pd.read_csv(IN_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

split_idx = int(len(df) * 0.70)
train_df = df.iloc[:split_idx].copy()
test_df = df.iloc[split_idx:].copy()

train_df.to_csv(TRAIN_OUT, index=False)
test_df.to_csv(TEST_OUT, index=False)

print("Total:", len(df))
print("Train:", len(train_df), "Test:", len(test_df))
print("Train range:", train_df["date"].min(), "→", train_df["date"].max())
print("Test range:", test_df["date"].min(), "→", test_df["date"].max())
print(f"Saved: {TRAIN_OUT}")
print(f"Saved: {TEST_OUT}")
