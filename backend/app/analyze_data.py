import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("backend/data/combined_news.csv")

print("==============================")
print("DATASET QUALITY CHECK")
print("==============================")

print("\nShape:")
print(df.shape)


# ============================================================
# LABEL ANALYSIS
# ============================================================

print("\nLabel distribution:")
print(df["label"].value_counts())

print("\nLabel percentage:")
print(
    (df["label"].value_counts(normalize=True) * 100).round(2)
)


# ============================================================
# ARTICLE LENGTH
# ============================================================

df["text_length"] = df["text"].str.len()

print("\nText length statistics:")
print(df["text_length"].describe())


# ============================================================
# WORD COUNT
# ============================================================

df["word_count"] = df["text"].str.split().str.len()

print("\nWord count statistics:")
print(df["word_count"].describe())


# ============================================================
# VERY SHORT ARTICLES
# ============================================================

print("\nArticles with fewer than 20 words:")
print((df["word_count"] < 20).sum())

print("\nArticles with fewer than 50 words:")
print((df["word_count"] < 50).sum())


# ============================================================
# DUPLICATE CHECK
# ============================================================

print("\nDuplicate titles:")
print(df["title"].duplicated().sum())

print("\nDuplicate texts:")
print(df["text"].duplicated().sum())


# ============================================================
# EMPTY VALUES
# ============================================================

print("\nEmpty titles:")
print((df["title"].str.strip() == "").sum())

print("\nEmpty texts:")
print((df["text"].str.strip() == "").sum())


# ============================================================
# LABEL + ARTICLE LENGTH
# ============================================================

print("\nAverage word count by label:")

print(
    df.groupby("label")["word_count"].mean()
)


# ============================================================
# SAVE ANALYSIS
# ============================================================

df.to_csv(
    "backend/data/combined_news.csv",
    index=False
)

print("\nAnalysis complete.")


# ============================================================
# SAVE CLEAN DATASET
# ============================================================

df = df[["title", "text", "label"]]

df.to_csv(
    "backend/data/combined_news.csv",
    index=False
)

print("\nAnalysis complete.")