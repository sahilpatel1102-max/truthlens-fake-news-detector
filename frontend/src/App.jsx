import { useState } from "react";
import "./App.css";
const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

function App() {
  const [title, setTitle] = useState("");
  const [articleText, setArticleText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const analyzeNews = async () => {
    if (!title.trim() || !articleText.trim()) {
      setError("Please enter both the headline and article text.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(`${API_URL}/analyze`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          title: title,
          article_text: articleText,
        }),
      });

      if (!response.ok) {
        throw new Error("Analysis failed.");
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(
        "Could not connect to the backend. Make sure FastAPI is running."
      );
    } finally {
      setLoading(false);
    }
  };

  const fakePercent = result
    ? (result.fake_score * 100).toFixed(1)
    : 0;

  const realPercent = result
    ? (result.real_score * 100).toFixed(1)
    : 0;

  return (
    <div className="app">

      {/* HEADER */}
      <header className="site-header">
       <div className="header-inner">
         <div className="brand">
            <div className="brand-icon">✓</div>
            <div>
             <h1>TruthLens</h1>
             <p>AI-Powered News Verification</p>
            </div>
          </div>

          <div className="header-badge">
            Fake News Detector
          </div>
        </div>
      </header>

      <main className="page-container">
        <section className="hero">
          <span className="hero-badge">AI + ML + Fact Checking</span>

          <h2>Verify the news before you believe it.</h2>

          <p>
            Analyze a news article using machine learning, fact-checking sources,
            related news, and AI-powered reasoning.
          </p>
        </section>


        {/* INPUT CARD */}
        <section className="input-card">

          <div className="section-title">
            <span className="section-icon">🔍</span>
            <div>
              <h2>Analyze a News Article</h2>
              <p>Enter a headline and the article text below.</p>
            </div>
          </div>

          <label>Headline</label>

          <input
            type="text"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            placeholder="Enter the news headline..."
          />

          <div className="character-count">
            {title.length} characters
          </div>

          <label>Article Text</label>

          <textarea
            value={articleText}
            onChange={(e) => setArticleText(e.target.value)}
            placeholder="Paste the full news article here..."
            rows="10"
          />

          <div className="character-count">
            {articleText.length} characters
          </div>

          <div className="button-row">
  <button
  type="button"
  className="analyze-button"
  onClick={analyzeNews}
  disabled={loading}
>
  {loading ? (
    <span className="loading-content">
      <span className="spinner"></span>
      Analyzing News...
    </span>
  ) : (
    "Analyze News"
  )}
</button>
{loading && (
  <div className="loading-message">
    <strong>Analyzing your article...</strong>
    <span>
      Checking the ML model, fact-checking sources, related news, and AI analysis.
    </span>
  </div>
)}

  <button
    type="button"
    className="clear-button"
    onClick={() => {
      setTitle("");
      setArticleText("");
      setResult(null);
      setError("");
    }}
  >
    Clear
  </button>
</div>

          {error && <div className="error">⚠️ {error}</div>}
        </section>

        {/* RESULTS */}
        {result && (
          <section className="results">

            <div className="results-heading">
              <h2>Analysis Results</h2>
              <span>✓ Analysis completed</span>
            </div>

            {/* MAIN VERDICT */}
<div
  className={`verdict-card ${
    result.prediction === "Fake"
      ? "verdict-fake"
      : "verdict-real"
  }`}
>
  <div className="verdict-icon">
    {result.prediction === "Fake" ? "⚠️" : "✓"}
  </div>

  <div>
    <p className="verdict-label">ML PREDICTION</p>

    <div
      className={`verdict-badge ${
        result.prediction === "Fake"
          ? "badge-fake"
          : "badge-real"
      }`}
    >
      {result.prediction === "Fake"
        ? "⚠ Likely Fake News"
        : "✓ Likely Real News"}
    </div>

    <p>
      Based on the trained machine learning model.
    </p>
  </div>
</div>
            {/* SCORE CARDS */}
            <div className="score-grid">

              <div className="score-card">
                <div className="score-header">
                  <span>Fake Probability</span>
                  <strong>{fakePercent}%</strong>
                </div>

                <div className="progress-bar">
                  <div
                    className="progress fake-progress"
                    style={{
                      width: `${fakePercent}%`,
                    }}
                  ></div>
                </div>
              </div>

              <div className="score-card">
                <div className="score-header">
                  <span>Real Probability</span>
                  <strong>{realPercent}%</strong>
                </div>

                <div className="progress-bar">
                  <div
                    className="progress real-progress"
                    style={{
                      width: `${realPercent}%`,
                    }}
                  ></div>
                </div>
              </div>

            </div>

            {/* AI ANALYSIS */}
            {result.ai_analysis && (
              <div className="result-card">

                <div className="card-heading">
                  <span>🤖</span>
                  <h3>AI Analysis</h3>
                </div>

                <div className="ai-grid">

                  <div>
                    <span className="small-label">VERDICT</span>
                    <p className="large-value">
                      {result.ai_analysis.verdict}
                    </p>
                  </div>

                  <div>
                    <span className="small-label">CONFIDENCE</span>
                    <p className="large-value">
                      {result.ai_analysis.confidence}
                    </p>
                  </div>

                </div>

                <div className="explanation">
                  <span className="small-label">EXPLANATION</span>
                  <p>
                    {result.ai_analysis.explanation}
                  </p>
                </div>

                {result.ai_analysis.evidence?.length > 0 && (
                  <div className="evidence">
                    <span className="small-label">
                      EVIDENCE
                    </span>

                    <ul>
                      {result.ai_analysis.evidence.map(
                        (item, index) => (
                          <li key={index}>{item}</li>
                        )
                      )}
                    </ul>
                  </div>
                )}

                {result.ai_analysis.warning && (
                  <div className="warning">
                    ⚠️ {result.ai_analysis.warning}
                  </div>
                )}

              </div>
            )}

            {/* FACT CHECKS */}
<div className="result-card">
  <div className="card-heading">
    <span>🔎</span>
    <div>
      <h3>Fact Checks</h3>
      <p className="card-subtitle">
        Claims found by Google Fact Check
      </p>
    </div>
  </div>

  {result.fact_checks?.length > 0 ? (
    <div className="source-list">
      {result.fact_checks.map((check, index) => {
        const review = check.claimReview?.[0];

        return (
          <div className="source-card" key={index}>
            <div className="source-top">
              <span className="source-badge">FACT CHECK</span>

              {review?.publisher?.name && (
                <span className="publisher">
                  {review.publisher.name}
                </span>
              )}
            </div>

            <h4>
              {review?.title ||
                check.text ||
                "Fact check available"}
            </h4>

            {review?.textualRating && (
              <p className="rating">
                Rating: <strong>{review.textualRating}</strong>
              </p>
            )}

            {review?.url && (
              <a
                href={review.url}
                target="_blank"
                rel="noopener noreferrer"
                className="source-link"
              >
                Read Fact Check →
              </a>
            )}
          </div>
        );
      })}
    </div>
  ) : (
    <div className="empty-state">
      <span>✓</span>
      <p>No matching fact checks were found.</p>
    </div>
  )}
</div>


{/* RELATED NEWS */}
<div className="result-card">
  <div className="card-heading">
    <span>📰</span>
    <div>
      <h3>Related News</h3>
      <p className="card-subtitle">
        Other articles related to this story
      </p>
    </div>
  </div>

  {result.related_news?.length > 0 ? (
    <div className="source-list">
      {result.related_news.map((article, index) => (
        <div className="source-card" key={index}>

          <div className="source-top">
            <span className="source-badge news-badge">
              NEWS
            </span>

            {article.source?.name && (
              <span className="publisher">
                {article.source.name}
              </span>
            )}
          </div>

          <h4>{article.title}</h4>

          {article.description && (
            <p className="article-description">
              {article.description}
            </p>
          )}

          {article.publishedAt && (
            <p className="published">
              Published:{" "}
              {new Date(article.publishedAt).toLocaleDateString()}
            </p>
          )}

          {article.url && (
            <a
              href={article.url}
              target="_blank"
              rel="noopener noreferrer"
              className="source-link"
            >
              Read Article →
            </a>
          )}
        </div>
      ))}
    </div>
  ) : (
    <div className="empty-state">
      <span>📰</span>
      <p>No related news articles were found.</p>
    </div>
  )}
</div>

            {/* DISCLAIMER */}
            <div className="disclaimer">
              <strong>⚠️ Important:</strong> This tool provides
              an AI/ML-based credibility assessment. It does not
              guarantee that a news article is absolutely true or
              false. Always verify important information using
              reliable sources.
            </div>

          </section>
        )}
      </main>
    </div>
 );

 <footer className="site-footer">
  <div className="footer-inner">
    <div>
      <strong>TruthLens</strong>
      <p>AI-Powered News Verification</p>
    </div>

    <p className="footer-note">
      Built with Machine Learning, FastAPI, React, and AI.
    </p>
  </div>

  <div className="footer-disclaimer">
    <strong>Important:</strong> TruthLens provides an automated assessment,
    not a definitive determination of truth. Always verify important claims
    using reliable primary sources and trusted fact-checking organizations.
  </div>

  <p className="copyright">
    © 2026 TruthLens. For educational and informational purposes.
  </p>
</footer>
}

export default App;