import pandas as pd
import re


# ============================================================
# 1. LOAD ISOT
# ============================================================

true_df = pd.read_csv("backend/data/isot/True.csv")
fake_df = pd.read_csv("backend/data/isot/Fake.csv")


# Add labels
true_df["label"] = 1   # Real
fake_df["label"] = 0   # Fake


# Combine
isot_df = pd.concat(
    [true_df, fake_df],
    ignore_index=True
)


# Keep only required columns
isot_df = isot_df[["title", "text", "label"]]


print("ISOT combined:", isot_df.shape)


# ============================================================
# 2. LOAD WELFAKE
# ============================================================

welfake_df = pd.read_csv(
    "backend/data/welfake/WELFake_Dataset.csv"
)


# Remove unnecessary index column
if "Unnamed: 0" in welfake_df.columns:
    welfake_df = welfake_df.drop(columns=["Unnamed: 0"])


# Keep required columns
welfake_df = welfake_df[["title", "text", "label"]]


print("WELFake original:", welfake_df.shape)


# ============================================================
# 3. COMBINE DATASETS
# ============================================================

df = pd.concat(
    [isot_df, welfake_df],
    ignore_index=True
)

print("Combined before cleaning:", df.shape)


# ============================================================
# 4. HANDLE MISSING VALUES
# ============================================================

df["title"] = df["title"].fillna("")
df["text"] = df["text"].fillna("")


# Convert everything to string
df["title"] = df["title"].astype(str)
df["text"] = df["text"].astype(str)


# ============================================================
# 5. CLEAN WHITESPACE
# ============================================================

def clean_whitespace(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


df["title"] = df["title"].apply(clean_whitespace)
df["text"] = df["text"].apply(clean_whitespace)


# ============================================================
# 6. REMOVE ROWS WITH NO ARTICLE TEXT
# ============================================================

df = df[df["text"].str.len() > 0]


# ============================================================
# 7. REMOVE EXACT DUPLICATES
# ============================================================

before_duplicates = len(df)

df = df.drop_duplicates(
    subset=["title", "text"]
)

after_duplicates = len(df)

print(
    "Duplicates removed:",
    before_duplicates - after_duplicates
)


# ============================================================
# 8. SHUFFLE DATA
# ============================================================

df = df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# ============================================================
# 9. FINAL INFORMATION
# ============================================================

print("\n==============================")
print("FINAL DATASET")
print("==============================")

print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nLabel distribution:")
print(df["label"].value_counts())

print("\nMissing values:")
print(df.isnull().sum())


# ============================================================
# 10. SAVE CLEAN DATASET
# ============================================================

output_path = "backend/data/combined_news.csv"

df.to_csv(
    output_path,
    index=False
)

print("\nSaved successfully:")
print(output_path)