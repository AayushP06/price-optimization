import React, { useState } from 'react';

const Results = () => {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const runOptimization = async () => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await fetch('/api/optimize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          cost_price: 1000,
          fixed_costs: 50,
          competitor_prices: [],
          num_competitors: 100,
          min_margin: 15,
          max_margin: 35
        })
      });

      const json = await res.json();
      if (!json.success) throw new Error(json.error || 'Unknown error');
      setResult(json);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const fmt = (price) => (parseFloat(price) || 0).toLocaleString('en-IN');

  return (
    <section id="results">
      <h2>Results & Insights</h2>
      <p className="section-subtitle">See a live recommendation from our optimizer</p>

      <div className="card" style={{ textAlign: 'center' }}>
        <button className="btn" onClick={runOptimization} disabled={loading}>
          {loading ? 'Analyzing...' : '⚡ Get Sample Recommendation'}
        </button>

        <div className="result-display">
          {loading && (
            <div className="loading" style={{ marginTop: '1.5rem' }}>
              <div className="spinner" />
              <div>Analyzing market data...</div>
            </div>
          )}

          {error && (
            <div className="error" style={{ marginTop: '1rem' }}>
              <strong>❌ Error:</strong> {error}
              <div style={{ marginTop: '0.4rem', fontSize: '0.85rem' }}>
                Make sure the backend is running on http://localhost:5000
              </div>
            </div>
          )}

          {result && (
            <div className="result-content" style={{ marginTop: '1.5rem', textAlign: 'left' }}>
              <div className="optimal-price-card">
                <div className="optimal-price-value">₹{fmt(result.optimal_price?.price)}</div>
                <div className="optimal-price-label">Recommended Optimal Price</div>
                <div className="optimal-price-stats">
                  <div>
                    <div className="stat-value">{parseFloat(result.optimal_price?.margin || 0).toFixed(2)}%</div>
                    <div className="stat-label">Profit Margin</div>
                  </div>
                  <div>
                    <div className="stat-value">₹{fmt(result.optimal_price?.expected_profit)}</div>
                    <div className="stat-label">Expected Profit</div>
                  </div>
                </div>
              </div>

              {result.competitor_stats && (
                <div className="market-summary">
                  Analyzed <strong style={{ color: 'var(--accent)' }}>{result.competitor_stats.count}</strong> competitors
                  &nbsp;·&nbsp; Range: ₹{fmt(result.competitor_stats.min)} – ₹{fmt(result.competitor_stats.max)}
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      <SuccessMetrics />
    </section>
  );
};

const SuccessMetrics = () => (
  <div className="card">
    <div className="metrics-grid">
      <div className="metric"><h4>&lt;1s</h4><p>Response Time</p></div>
      <div className="metric"><h4>95%</h4><p>Accuracy Rate</p></div>
      <div className="metric"><h4>18%</h4><p>Avg Profit Increase</p></div>
      <div className="metric"><h4>10K+</h4><p>Products Analyzed</p></div>
    </div>
  </div>
);

export default Results;
