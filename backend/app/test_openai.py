from openai_service import analyze_with_ai


result = analyze_with_ai(
    title="Delhi High Court Questions Authorities Over Satya Niketan Building Collapse",

    article_text=(
        "New Delhi, September 8, 2026: "
        "The Delhi High Court has strongly criticised "
        "civic authorities following the collapse of a "
        "multi-storey paying guest accommodation in "
        "Satya Niketan that left seven people dead "
        "and several others injured."
    ),

    ml_prediction="Real",

    fake_score=0.1894,

    real_score=0.8106,

    fact_checks=[],

    related_news=[]
)


print("\nAI RESULT:")
print(result)