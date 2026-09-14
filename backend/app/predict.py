import joblib
from pathlib import Path


# Model paths
MODEL_DIR = Path("backend/models")

vectorizer = joblib.load(
    MODEL_DIR / "tfidf_vectorizer.pkl"
)

model = joblib.load(
    MODEL_DIR / "logistic_regression_model.pkl"
)


def predict_news(title, article_text):
    """
    Predict whether a news article is fake or real.
    """

    # Combine headline and article
    full_text = title + " " + article_text

    # Convert text into TF-IDF features
    text_tfidf = vectorizer.transform([full_text])

    # Make prediction
    prediction = model.predict(text_tfidf)[0]

    # Get model scores
    probabilities = model.predict_proba(text_tfidf)[0]

    fake_probability = probabilities[0]
    real_probability = probabilities[1]

    if prediction == 0:
        label = "Fake"
    else:
        label = "Real"

    return label, fake_probability, real_probability


if __name__ == "__main__":

    print("=" * 50)
    print("FAKE NEWS DETECTOR")
    print("=" * 50)

    title = input("\nEnter news headline: ")

    article_text = input(
        "\nEnter article text: "
    )

    label, fake_probability, real_probability = predict_news(
        title,
        article_text
    )

    print("\n" + "=" * 50)
    print("RESULT")
    print("=" * 50)

    print(f"Prediction: {label}")
    print(f"Fake score: {fake_probability * 100:.2f}%")
    print(f"Real score: {real_probability * 100:.2f}%")

    print("=" * 50)