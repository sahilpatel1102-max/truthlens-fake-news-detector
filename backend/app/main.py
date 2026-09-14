# print("Fake News Detector backend is working!")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import json
from pathlib import Path

from .fact_check import search_fact_checks
from .news_api import search_news
from .openai_service import analyze_with_ai

import os
from dotenv import load_dotenv

load_dotenv()





# ============================================================
# CREATE FASTAPI APP
# ============================================================

app = FastAPI(
    title="Fake News Detector API",
    description="API for detecting potentially fake news",
    version="1.0.0"
)

FRONTEND_URL = os.getenv(
    "FRONTEND_URL",
    "http://localhost:5173"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        FRONTEND_URL,
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# LOAD MODEL
# ============================================================

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

vectorizer = joblib.load(
    MODEL_DIR / "tfidf_vectorizer.pkl"
)

model = joblib.load(
    MODEL_DIR / "logistic_regression_model.pkl"
)


# ============================================================
# REQUEST FORMAT
# ============================================================

class NewsRequest(BaseModel):
    title: str
    article_text: str


# ============================================================
# HOME ENDPOINT
# ============================================================

@app.get("/")
def home():
    return {
        "message": "Fake News Detector API is running"
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================


@app.post("/analyze")
def analyze_news(news: NewsRequest):

    # ========================================================
    # 1. ML MODEL
    # ========================================================

    full_text = news.title + " " + news.article_text

    text_tfidf = vectorizer.transform([full_text])

    prediction = model.predict(text_tfidf)[0]

    probabilities = model.predict_proba(text_tfidf)[0]

    fake_probability = probabilities[0]
    real_probability = probabilities[1]

    if prediction == 0:
        label = "Fake"
    else:
        label = "Real"


    # ========================================================
    # 2. GOOGLE FACT CHECK
    # ========================================================

    try:

        fact_checks = search_fact_checks(
            news.title
        )

    except Exception as e:

        print("Fact Check API error:", e)

        fact_checks = []


    # ========================================================
    # 3. NEWSAPI
    # ========================================================

    try:

        related_articles = search_news(
            news.title,
            page_size=5
        )

    except Exception as e:

        print("NewsAPI error:", e)

        related_articles = []


    # ========================================================
    # 4. OPENAI ANALYSIS
    # ========================================================

    try:

        ai_analysis = analyze_with_ai(
            title=news.title,
            article_text=news.article_text,
            ml_prediction=label,
            fake_score=float(fake_probability),
            real_score=float(real_probability),
            fact_checks=fact_checks,
            related_news=related_articles
        )

    except Exception as e:

        print("OpenAI API error:", e)

        ai_analysis = None


    # ========================================================
    # 5. CONVERT AI RESPONSE TO JSON
    # ========================================================

    if ai_analysis is not None:

        try:

            ai_analysis = json.loads(ai_analysis)

        except (json.JSONDecodeError, TypeError):

            ai_analysis = {
                "verdict": "Uncertain",
                "confidence": "Low",
                "explanation": str(ai_analysis),
                "evidence": [],
                "warning": "AI response could not be parsed into structured JSON."
            }


    # ========================================================
    # 6. FINAL RESPONSE
    # ========================================================

    return {

        "prediction": label,

        "fake_score": round(
            float(fake_probability),
            4
        ),

        "real_score": round(
            float(real_probability),
            4
        ),

        "fact_checks": fact_checks,

        "related_news": related_articles,

        "ai_analysis": ai_analysis
    }
