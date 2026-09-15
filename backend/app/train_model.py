import pandas as pd
import joblib
from pathlib import Path 

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. LOAD DATA
# ============================================================

train_df = pd.read_csv("backend/data/train.csv")
validation_df = pd.read_csv("backend/data/validation.csv")
test_df = pd.read_csv("backend/data/test.csv")


print("==============================")
print("DATA LOADED")
print("==============================")

print("Train:", train_df.shape)
print("Validation:", validation_df.shape)
print("Test:", test_df.shape)


# ============================================================
# 2. GET TEXT AND LABELS
# ============================================================

X_train = train_df["full_text"]
y_train = train_df["label"]

X_validation = validation_df["full_text"]
y_validation = validation_df["label"]

X_test = test_df["full_text"]
y_test = test_df["label"]


# ============================================================
# 3. TF-IDF
# ============================================================

print("\n==============================")
print("TF-IDF")
print("==============================")

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_features=100000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)


print("Fitting TF-IDF on training data...")

X_train_tfidf = vectorizer.fit_transform(X_train)

print("Transforming validation data...")

X_validation_tfidf = vectorizer.transform(X_validation)

print("Transforming test data...")

X_test_tfidf = vectorizer.transform(X_test)


print("\nTF-IDF train matrix:")
print(X_train_tfidf.shape)


# ============================================================
# 4. LOGISTIC REGRESSION
# ============================================================

print("\n==============================")
print("LOGISTIC REGRESSION")
print("==============================")

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)


print("Training model...")

model.fit(
    X_train_tfidf,
    y_train
)


print("Training complete!")


# ============================================================
# 5. VALIDATION EVALUATION
# ============================================================

print("\n==============================")
print("VALIDATION RESULTS")
print("==============================")

validation_predictions = model.predict(
    X_validation_tfidf
)


validation_accuracy = accuracy_score(
    y_validation,
    validation_predictions
)

validation_precision = precision_score(
    y_validation,
    validation_predictions
)

validation_recall = recall_score(
    y_validation,
    validation_predictions
)

validation_f1 = f1_score(
    y_validation,
    validation_predictions
)


print(f"Accuracy : {validation_accuracy:.4f}")
print(f"Precision: {validation_precision:.4f}")
print(f"Recall   : {validation_recall:.4f}")
print(f"F1 Score : {validation_f1:.4f}")


# ============================================================
# 6. FINAL TEST EVALUATION
# ============================================================

print("\n==============================")
print("TEST RESULTS")
print("==============================")

test_predictions = model.predict(
    X_test_tfidf
)


test_accuracy = accuracy_score(
    y_test,
    test_predictions
)

test_precision = precision_score(
    y_test,
    test_predictions
)

test_recall = recall_score(
    y_test,
    test_predictions
)

test_f1 = f1_score(
    y_test,
    test_predictions
)


print(f"Accuracy : {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall   : {test_recall:.4f}")
print(f"F1 Score : {test_f1:.4f}")


# ============================================================
# 7. CLASSIFICATION REPORT
# ============================================================

print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================")

print(
    classification_report(
        y_test,
        test_predictions,
        target_names=["Fake", "Real"]
    )
)


# ============================================================
# 8. CONFUSION MATRIX
# ============================================================

print("\n==============================")
print("CONFUSION MATRIX")
print("==============================")

cm = confusion_matrix(
    y_test,
    test_predictions
)

print(cm)


# ============================================================
# 9. SAVE MODEL
# ============================================================

print("\n==============================")
print("SAVING MODEL")
print("==============================")

# Create models directory
model_dir = Path("backend/models")
model_dir.mkdir(parents=True, exist_ok=True)

# Save TF-IDF vectorizer
joblib.dump(
    vectorizer,
    model_dir / "tfidf_vectorizer.pkl"
)

# Save Logistic Regression model
joblib.dump(
    model,
    model_dir / "logistic_regression_model.pkl"
)

print("\nModels saved successfully!")
print("TF-IDF vectorizer: backend/models/tfidf_vectorizer.pkl")
print("Logistic Regression: backend/models/logistic_regression_model.pkl")
