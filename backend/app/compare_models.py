import time
from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Load datasets
# --------------------------------------------------

print("Loading datasets...")

train_df = pd.read_csv(DATA_DIR / "train.csv")
validation_df = pd.read_csv(DATA_DIR / "validation.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")

print(f"Training samples:   {len(train_df)}")
print(f"Validation samples: {len(validation_df)}")
print(f"Test samples:       {len(test_df)}")


# --------------------------------------------------
# Prepare text
# --------------------------------------------------

X_train_text = train_df["full_text"].fillna("")
y_train = train_df["label"]

X_val_text = validation_df["full_text"].fillna("")
y_val = validation_df["label"]

X_test_text = test_df["full_text"].fillna("")
y_test = test_df["label"]


# --------------------------------------------------
# TF-IDF
# --------------------------------------------------

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_features=100000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
)

X_train = vectorizer.fit_transform(X_train_text)
X_val = vectorizer.transform(X_val_text)
X_test = vectorizer.transform(X_test_text)

print(f"TF-IDF training matrix: {X_train.shape}")


# --------------------------------------------------
# Models
# --------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42,
    ),
    "SVM": CalibratedClassifierCV(
        estimator=LinearSVC(random_state=42),
        method="sigmoid",
        cv=3,
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    ),

    "Naive Bayes": MultinomialNB(),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
    ),
}


# --------------------------------------------------
# Train and evaluate
# --------------------------------------------------

results = []
trained_models = {}

print("\n" + "=" * 70)
print("TRAINING AND EVALUATING MODELS")
print("=" * 70)

for name, model in models.items():

    print(f"\nTraining: {name}")

    start_time = time.time()

    model.fit(X_train, y_train)

    training_time = time.time() - start_time

    # Validation predictions
    val_predictions = model.predict(X_val)

    accuracy = accuracy_score(y_val, val_predictions)
    precision = precision_score(
        y_val,
        val_predictions,
        zero_division=0,
    )
    recall = recall_score(
        y_val,
        val_predictions,
        zero_division=0,
    )
    f1 = f1_score(
        y_val,
        val_predictions,
        zero_division=0,
    )

    results.append(
        {
            "model": name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "training_time_seconds": training_time,
        }
    )

    trained_models[name] = model

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Training:  {training_time:.2f} seconds")


# --------------------------------------------------
# Results table
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="f1_score",
    ascending=False,
).reset_index(drop=True)

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    results_df[
        [
            "model",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "training_time_seconds",
        ]
    ].to_string(index=False)
)


# --------------------------------------------------
# Select best model
# --------------------------------------------------

best_model_name = results_df.iloc[0]["model"]
best_model = trained_models[best_model_name]

print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)

print(f"Best model: {best_model_name}")
print(
    f"Validation F1: "
    f"{results_df.iloc[0]['f1_score']:.4f}"
)


# --------------------------------------------------
# Test the best model
# --------------------------------------------------

print("\nTesting the best model on the test dataset...")

test_predictions = best_model.predict(X_test)

test_accuracy = accuracy_score(y_test, test_predictions)
test_precision = precision_score(
    y_test,
    test_predictions,
    zero_division=0,
)
test_recall = recall_score(
    y_test,
    test_predictions,
    zero_division=0,
)
test_f1 = f1_score(
    y_test,
    test_predictions,
    zero_division=0,
)

print("\nBest Model Test Results")
print("-" * 40)
print(f"Model:     {best_model_name}")
print(f"Accuracy:  {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall:    {test_recall:.4f}")
print(f"F1 Score:  {test_f1:.4f}")


# --------------------------------------------------
# Save best model
# --------------------------------------------------

best_model_path = MODEL_DIR / "best_model.pkl"

joblib.dump(best_model, best_model_path)

print(f"\nBest model saved to:")
print(best_model_path)


# --------------------------------------------------
# Save TF-IDF vectorizer
# --------------------------------------------------

vectorizer_path = MODEL_DIR / "best_tfidf_vectorizer.pkl"

joblib.dump(vectorizer, vectorizer_path)

print(f"TF-IDF vectorizer saved to:")
print(vectorizer_path)


# --------------------------------------------------
# Save comparison results
# --------------------------------------------------

comparison_path = MODEL_DIR / "model_comparison.csv"

results_df.to_csv(
    comparison_path,
    index=False,
)

print(f"\nModel comparison saved to:")
print(comparison_path)

print("\nDone!")