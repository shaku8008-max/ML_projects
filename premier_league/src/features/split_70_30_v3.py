import pandas as pd

df = pd.read_csv("data/processed/epl_cl_2526_features_v3_form_elo.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

split_idx = int(len(df) * 0.70)
train_df = df.iloc[:split_idx].copy()
test_df = df.iloc[split_idx:].copy()

train_df.to_csv("data/processed/train_70_v3.csv", index=False)
test_df.to_csv("data/processed/test_30_v3.csv", index=False)

print("Total:", len(df))
print("Train:", len(train_df), "Test:", len(test_df))
print("Train range:", train_df["date"].min(), "→", train_df["date"].max())
print("Test range:", test_df["date"].min(), "→", test_df["date"].max())
