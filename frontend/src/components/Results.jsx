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
      const payload = {
        cost_price: 1000,
        fixed_costs: 50,
        competitor_prices: [],
        num_competitors: 100,
        min_margin: 15,
        max_margin: 35
      };

      const res = await fetch('/api/optimize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      const json = await res.json();
      
      if (!json.success) {
        throw new Error(json.error || 'Unknown error');
      }

      setResult(json.data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const formatPrice = (price) => {
    const num = parseFloat(price) || 0;
    return num.toLocaleString('en-IN');
  };

  return (
    <section id="results">
      <h2>Results & Insights</h2>
      <p className="section-subtitle">
        See how our platform helps sellers maximize profits
      </p>

      <div className="card" style={{ textAlign: 'center' }}>
        <h3 style={{ color: 'var(--accent-primary)' }}>Try It Now</h3>
        <p style={{ color: 'var(--text-secondary)', margin: '0.5rem 0 1rem' }}>
          See how our system works with a sample product. We'll analyze competitor prices and recommend the optimal price for maximum profit.
        </p>

        <div className="sample-details">
          <h4>Sample Product Details:</h4>
          <div className="sample-grid">
            <div><strong>Cost Price:</strong> ₹1,000</div>
            <div><strong>Fixed Costs:</strong> ₹50 per unit</div>
            <div><strong>Competitors:</strong> 100 (simulated market)</div>
            <div><strong>Target Margin:</strong> 15% - 35%</div>
          </div>
          <p className="sample-note">
            Note: This demo uses consistent sample data to show how the system works. 
            In real use, you would provide actual competitor prices from your market research.
          </p>
        </div>

        <button className="btn" onClick={runOptimization} disabled={loading}>
          {loading ? 'Analyzing...' : 'Get Sample Recommendation'}
        </button>

        <div className="result-display">
          {loading && (
            <div className="loading">
              <div className="spinner"></div>
              <div>⏳ Analyzing market data...</div>
              <div style={{ fontSize: '0.9rem', marginTop: '0.5rem' }}>
                Processing competitor prices and calculating optimal price point
              </div>
            </div>
          )}

          {error && (
            <div className="error">
              <strong>❌ Error:</strong> {error}
              <div style={{ marginTop: '0.5rem', fontSize: '0.9rem' }}>
                Make sure the backend server is running on http://localhost:5000
              </div>
            </div>
          )}

          {result && (
            <div className="result-content">
              <h4>✨ Optimization Complete!</h4>

              <div className="optimal-price-card">
                <div className="optimal-price-value">
                  ₹{formatPrice(result.optimal_price?.price || 0)}
                </div>
                <div className="optimal-price-label">Recommended Optimal Price</div>
                <div className="optimal-price-stats">
                  <div>
                    <div className="stat-value">{parseFloat(result.optimal_price?.margin || 0).toFixed(2)}%</div>
                    <div className="stat-label">Profit Margin</div>
                  </div>
                  <div>
                    <div className="stat-value">₹{formatPrice(result.optimal_price?.expected_profit || 0)}</div>
                    <div className="stat-label">Expected Profit</div>
                  </div>
                </div>
              </div>

              {result.competitor_stats && (
                <div className="market-summary">
                  <strong>Market Summary:</strong>
                  <div>Analyzed {result.competitor_stats.count} competitor prices</div>
                </div>
              )}

              {result.competitor_prices && result.competitor_prices.length > 0 && (
                <CompetitorPricesGrid 
                  prices={result.competitor_prices} 
                  optimalPrice={result.optimal_price?.price}
                  stats={result.competitor_stats}
                />
              )}
            </div>
          )}
        </div>
      </div>

      <div className="card">
        <h3>Why Sellers Love Our Platform</h3>
        <ul className="benefits-list">
          <li>95% accuracy in identifying optimal price points compared to manual analysis</li>
          <li>Instant results - analyze thousands of competitor prices in seconds</li>
          <li>Average profit improvement of 18% when sellers adopt recommended pricing</li>
          <li>Works across all product categories and price ranges</li>
          <li>Easy to use - no technical knowledge required</li>
        </ul>
      </div>

      <MarketAnalysisTable />
      <SuccessMetrics />
    </section>
  );
};

const CompetitorPricesGrid = ({ prices, optimalPrice, stats }) => {
  const [showAll, setShowAll] = useState(false);
  const pricesToShow = showAll ? prices : prices.slice(0, 100);
  const hasMore = prices.length > 100;

  return (
    <div className="competitor-prices-container">
      <h4>Competitor Prices Analyzed ({prices.length} total)</h4>
      <div className="prices-grid">
        {pricesToShow.map((price, idx) => {
          const priceNum = parseFloat(price) || 0;
          const optPrice = parseFloat(optimalPrice) || 0;
          const isNearOptimal = optPrice > 0 && Math.abs(priceNum - optPrice) < (optPrice * 0.1);

          return (
            <div
              key={idx}
              className={`price-tag ${isNearOptimal ? 'near-optimal' : ''}`}
            >
              <div>₹{priceNum.toLocaleString('en-IN')}</div>
              {isNearOptimal && <div className="price-label">Near optimal</div>}
            </div>
          );
        })}
      </div>
      {hasMore && !showAll && (
        <button className="show-more-btn" onClick={() => setShowAll(true)}>
          Show All {prices.length} Prices
        </button>
      )}
      {stats && (
        <div className="price-stats">
          <div><strong>Lowest:</strong> ₹{stats.min?.toLocaleString('en-IN')}</div>
          <div><strong>Highest:</strong> ₹{stats.max?.toLocaleString('en-IN')}</div>
          <div><strong>Average:</strong> ₹{stats.mean?.toLocaleString('en-IN')}</div>
          <div><strong>Median:</strong> ₹{stats.median?.toLocaleString('en-IN')}</div>
        </div>
      )}
    </div>
  );
};

const MarketAnalysisTable = () => {
  return (
    <>
      <h3>Market Analysis Example</h3>
      <table className="results-table">
        <thead>
          <tr>
            <th>Competitor</th>
            <th>Price (₹)</th>
            <th>Estimated Profit Margin</th>
            <th>Market Position</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Competitor A</td>
            <td>₹1,299</td>
            <td>12%</td>
            <td>Low Price Leader</td>
          </tr>
          <tr>
            <td>Competitor B</td>
            <td>₹1,549</td>
            <td>22%</td>
            <td>Mid-Range</td>
          </tr>
          <tr>
            <td>Competitor C</td>
            <td>₹1,799</td>
            <td>28%</td>
            <td>Premium</td>
          </tr>
          <tr>
            <td>Competitor D</td>
            <td>₹1,649</td>
            <td>24%</td>
            <td>Mid-Range</td>
          </tr>
          <tr className="optimal-row">
            <td><strong>System Recommendation</strong></td>
            <td className="optimal-price">₹1,599</td>
            <td className="optimal-price">26%</td>
            <td className="optimal-price">Optimal Sweet Spot</td>
          </tr>
          <tr>
            <td>Competitor E</td>
            <td>₹1,899</td>
            <td>30%</td>
            <td>High Premium</td>
          </tr>
        </tbody>
      </table>
    </>
  );
};

const SuccessMetrics = () => {
  return (
    <div className="card">
      <h3>Success Metrics</h3>
      <div className="metrics-grid">
        <div className="metric">
          <h4>&lt;1s</h4>
          <p>Average Response Time</p>
        </div>
        <div className="metric">
          <h4>95%</h4>
          <p>Accuracy Rate</p>
        </div>
        <div className="metric">
          <h4>18%</h4>
          <p>Avg Profit Increase</p>
        </div>
        <div className="metric">
          <h4>10K+</h4>
          <p>Products Analyzed</p>
        </div>
      </div>
    </div>
  );
};

export default Results;


