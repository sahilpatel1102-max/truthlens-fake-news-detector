import pandas as pd


# =========================
# ISOT DATASET
# =========================

true_path = "backend/data/isot/True.csv"
fake_path = "backend/data/isot/Fake.csv"

true_df = pd.read_csv(true_path)
fake_df = pd.read_csv(fake_path)

print("\n==============================")
print("ISOT DATASET")
print("==============================")

print("\nTrue.csv shape:")
print(true_df.shape)

print("\nFake.csv shape:")
print(fake_df.shape)

print("\nTrue.csv columns:")
print(true_df.columns.tolist())

print("\nFake.csv columns:")
print(fake_df.columns.tolist())

print("\nTrue.csv sample:")
print(true_df.head(3))

print("\nFake.csv sample:")
print(fake_df.head(3))


# =========================
# WELFAKE DATASET
# =========================

welfake_path = "backend/data/welfake/WELFake_Dataset.csv"

welfake_df = pd.read_csv(welfake_path)

print("\n==============================")
print("WELFAKE DATASET")
print("==============================")

print("\nShape:")
print(welfake_df.shape)

print("\nColumns:")
print(welfake_df.columns.tolist())

print("\nFirst 5 rows:")
print(welfake_df.head())

print("\nLabel distribution:")
print(welfake_df["label"].value_counts())

print("\nMissing values:")
print(welfake_df.isnull().sum())