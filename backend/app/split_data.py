import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("backend/data/combined_news.csv")

print("Original dataset:", df.shape)


# ============================================================
# CREATE FULL TEXT
# ============================================================

# Combine title + article text.
# The title often contains useful information for fake-news
# classification.

df["full_text"] = (
    df["title"].fillna("") + " " +
    df["text"].fillna("")
)


# ============================================================
# FIRST SPLIT: 80% TRAIN, 20% TEMPORARY
# ============================================================

train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["label"]
)


# ============================================================
# SECOND SPLIT: 10% VALIDATION, 10% TEST
# ============================================================

validation_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["label"]
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n==============================")
print("DATA SPLIT")
print("==============================")

print("\nTrain:")
print(train_df.shape)

print("\nValidation:")
print(validation_df.shape)

print("\nTest:")
print(test_df.shape)


print("\nTrain label distribution:")
print(train_df["label"].value_counts(normalize=True))

print("\nValidation label distribution:")
print(validation_df["label"].value_counts(normalize=True))

print("\nTest label distribution:")
print(test_df["label"].value_counts(normalize=True))


# ============================================================
# SAVE
# ============================================================

train_df.to_csv(
    "backend/data/train.csv",
    index=False
)

validation_df.to_csv(
    "backend/data/validation.csv",
    index=False
)

test_df.to_csv(
    "backend/data/test.csv",
    index=False
)


print("\nFiles saved successfully.")