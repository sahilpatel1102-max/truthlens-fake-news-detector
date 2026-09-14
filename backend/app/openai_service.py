import os
import json
from openai import OpenAI
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY is not configured."
    )

client = OpenAI(
    api_key=OPENAI_API_KEY
)


def analyze_with_ai(
    title,
    article_text,
    ml_prediction,
    fake_score,
    real_score,
    fact_checks,
    related_news
):
    """
    Use OpenAI to analyze the evidence collected
    by our fake-news detection system.
    """

    prompt = f"""
You are an AI news-analysis assistant.

Analyze the news article using ONLY the evidence provided below.

IMPORTANT:
- Do not claim that something is true merely because the ML model says "Real".
- Do not claim something is false merely because the ML model says "Fake".
- Treat the ML prediction as a classification signal.
- Give higher importance to reliable fact-checking evidence.
- Use related news only as supporting evidence.
- If there is insufficient evidence, clearly say so.
- Do not invent sources, facts, URLs, or events.

NEWS HEADLINE:
{title}

NEWS ARTICLE:
{article_text}

ML CLASSIFIER:
Prediction: {ml_prediction}
Fake score: {fake_score}
Real score: {real_score}

GOOGLE FACT CHECK RESULTS:
{json.dumps(fact_checks, indent=2)}

RELATED NEWS:
{json.dumps(related_news, indent=2)}

Return a concise analysis with:

1. verdict
2. confidence
3. explanation
4. evidence
5. warning

The verdict must be one of:
- "Likely Fake"
- "Likely Real"
- "Uncertain"

The confidence must be one of:
- "Low"
- "Medium"
- "High"

Return ONLY valid JSON in this format:

{{
    "verdict": "...",
    "confidence": "...",
    "explanation": "...",
    "evidence": [
        "...",
        "..."
    ],
    "warning": "..."
}}
"""

    response = client.responses.create(
        model="gpt-5.6-luna",
        input=prompt
    )

    return response.output_text