import joblib
from pathlib import Path
import math


# --------------------------------------------------
# Load model and vectorizer
# --------------------------------------------------

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

model = joblib.load(MODEL_DIR / "best_model.pkl")
vectorizer = joblib.load(MODEL_DIR / "best_tfidf_vectorizer.pkl")


# --------------------------------------------------
# Prediction function
# --------------------------------------------------

def predict_news(title: str, article_text: str):
    """
    Predict whether a news article is Fake or Real.

    Returns:
        prediction
        fake_score
        real_score
    """

    full_text = title + " " + article_text

    # Convert text into TF-IDF features
    text_tfidf = vectorizer.transform([full_text])

    # Predict class
    prediction = model.predict(text_tfidf)[0]

    # Calibrated SVM provides probabilities
    probabilities = model.predict_proba(text_tfidf)[0]

    fake_score = probabilities[0]
    real_score = probabilities[1]

    if prediction == 0:
        label = "Fake"
    else:
        label = "Real"

    return {
        "prediction": label,
        "fake_score": round(float(fake_score), 4),
        "real_score": round(float(real_score), 4),
    }


# --------------------------------------------------
# Test from command line
# --------------------------------------------------

if __name__ == "__main__":

    title = input("Enter headline: ")
    article_text = input("Enter article text: ")

    result = predict_news(title, article_text)

    print("\nPrediction Result")
    print("-------------------------")
    print(f"Prediction: {result['prediction']}")
    print(f"Fake Score: {result['fake_score'] * 100:.2f}%")
    print(f"Real Score: {result['real_score'] * 100:.2f}%")