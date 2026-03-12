import React, { useState, useEffect, useMemo } from 'react';

const PriceOptimizer = ({ onOptimize }) => {
  const [formData, setFormData] = useState({
    productName: '',
    quality: 'standard',
    costPrice: 1000,
    fixedCosts: 50,
    availableUnits: '',
    numCompetitors: 100,
    competitorPrices: ''
  });

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [showPreview, setShowPreview] = useState(false);
  const [animatedValue, setAnimatedValue] = useState(0);

  // Real-time margin calculation preview
  const previewCalculation = useMemo(() => {
    const cost = parseFloat(formData.costPrice) || 0;
    const fixed = parseFloat(formData.fixedCosts) || 0;
    if (cost <= 0) return null;

    // Calculate sample margins
    const margins = [15, 20, 25, 30, 35];
    return margins.map(margin => {
      const price = cost * (1 + margin / 100);
      const profitPerUnit = price - cost - fixed;
      const marginPercent = (profitPerUnit / price) * 100;
      return {
        margin,
        price: price.toFixed(2),
        profitPerUnit: profitPerUnit.toFixed(2),
        marginPercent: marginPercent.toFixed(2)
      };
    });
  }, [formData.costPrice, formData.fixedCosts]);

  // Animate numbers when result appears
  useEffect(() => {
    if (result?.optimal_price?.expected_profit) {
      const target = result.optimal_price.expected_profit;
      const duration = 1500;
      const steps = 60;
      const increment = target / steps;
      let current = 0;
      const timer = setInterval(() => {
        current += increment;
        if (current >= target) {
          setAnimatedValue(target);
          clearInterval(timer);
        } else {
          setAnimatedValue(current);
        }
      }, duration / steps);
      return () => clearInterval(timer);
    }
  }, [result]);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
    setError(null);
    setResult(null);
    // Show preview when cost price changes
    if (name === 'costPrice' || name === 'fixedCosts') {
      setShowPreview(true);
    }
  };

  const parsePrices = (text) => {
    if (!text) return [];
    return text.split(/[,\n\s]+/)
      .map(s => s.trim())
      .filter(Boolean)
      .map(Number)
      .filter(n => !Number.isNaN(n));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const competitorPrices = parsePrices(formData.competitorPrices);
      const availableUnits = formData.availableUnits ? parseInt(formData.availableUnits) : null;
      const numCompetitors = formData.numCompetitors ? parseInt(formData.numCompetitors) : 100;

      const body = {
        product_name: formData.productName,
        quality: formData.quality,
        cost_price: parseFloat(formData.costPrice),
        fixed_costs: parseFloat(formData.fixedCosts),
        available_units: availableUnits,
        competitor_prices: competitorPrices.length > 0 ? competitorPrices : [],
        num_competitors: competitorPrices.length > 0 ? undefined : numCompetitors
      };

      const res = await fetch('/api/optimize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });

      const json = await res.json();
      
      if (!json.success) {
        throw new Error(json.error || 'Unknown error');
      }

      setResult(json.data);
      if (onOptimize) {
        onOptimize(json.data);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <section id="optimize">
      <h2>Optimize Your Product Price</h2>
      <p className="section-subtitle">
        Enter your product details and get the optimal price recommendation instantly
      </p>

      <div className="card">
        <form id="optForm" className="price-form" onSubmit={handleSubmit}>
          <div>
            <label htmlFor="productName">Product Name</label>
            <input
              id="productName"
              name="productName"
              type="text"
              placeholder="e.g., Wireless Earbuds"
              value={formData.productName}
              onChange={handleChange}
            />
          </div>
          
          <div>
            <label htmlFor="quality">Quality</label>
            <select
              id="quality"
              name="quality"
              value={formData.quality}
              onChange={handleChange}
            >
              <option value="budget">Budget</option>
              <option value="standard">Standard</option>
              <option value="premium">Premium</option>
            </select>
          </div>
          
          <div className="input-group">
            <label htmlFor="costPrice">
              Cost Price (₹)
              {formData.costPrice > 0 && (
                <span className="input-hint">✓ Valid</span>
              )}
            </label>
            <input
              id="costPrice"
              name="costPrice"
              type="number"
              min="0"
              step="0.01"
              value={formData.costPrice}
              onChange={handleChange}
              required
              className={formData.costPrice > 0 ? 'input-valid' : ''}
            />
            {showPreview && previewCalculation && (
              <div className="preview-tooltip">
                <div className="preview-header">💡 Quick Preview</div>
                <div className="preview-grid">
                  {previewCalculation.slice(0, 3).map((calc, idx) => (
                    <div key={idx} className="preview-item">
                      <div className="preview-margin">{calc.margin}% margin</div>
                      <div className="preview-price">₹{parseFloat(calc.price).toLocaleString('en-IN')}</div>
                      <div className="preview-profit">+₹{parseFloat(calc.profitPerUnit).toLocaleString('en-IN')}/unit</div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
          
          <div className="input-group">
            <label htmlFor="fixedCosts">
              Fixed Costs per Unit (₹)
              {formData.fixedCosts >= 0 && (
                <span className="input-hint">✓ Set</span>
              )}
            </label>
            <input
              id="fixedCosts"
              name="fixedCosts"
              type="number"
              min="0"
              step="0.01"
              value={formData.fixedCosts}
              onChange={handleChange}
              className={formData.fixedCosts >= 0 ? 'input-valid' : ''}
            />
          </div>
          
          <div>
            <label htmlFor="availableUnits">Available Units (Stock)</label>
            <input
              id="availableUnits"
              name="availableUnits"
              type="number"
              min="0"
              step="1"
              placeholder="Leave blank for unlimited"
              value={formData.availableUnits}
              onChange={handleChange}
            />
            <small>Optional: Limits demand based on available stock</small>
          </div>
          
          <div>
            <label htmlFor="numCompetitors">Number of Competitors (Auto-generate)</label>
            <input
              id="numCompetitors"
              name="numCompetitors"
              type="number"
              min="1"
              step="1"
              value={formData.numCompetitors}
              onChange={handleChange}
              placeholder="Enter any number"
            />
            <small>Enter any number of competitors (minimum 1). Leave blank if entering manual prices below</small>
          </div>
          
          <div style={{ gridColumn: '1 / -1' }}>
            <label htmlFor="competitorPrices">OR Enter Competitor Prices Manually (comma-separated)</label>
            <textarea
              id="competitorPrices"
              name="competitorPrices"
              rows="3"
              placeholder="e.g., 1299, 1549, 1799, 1649, 1899"
              value={formData.competitorPrices}
              onChange={handleChange}
            />
            <small>If you enter prices here, they will be used instead of auto-generated ones</small>
          </div>
          
          <div>
            <button 
              type="submit" 
              className={`btn btn-primary ${loading ? 'btn-loading' : ''}`} 
              disabled={loading}
            >
              {loading ? (
                <>
                  <span className="btn-spinner"></span>
                  Calculating...
                </>
              ) : (
                <>
                  <span className="btn-icon">🚀</span>
                  Generate Optimal Price
                </>
              )}
            </button>
          </div>
        </form>

        <div id="optOutput" className="result-container">
          {loading && (
            <div className="loading">
              <div className="spinner"></div>
              <p>Calculating optimal price...</p>
            </div>
          )}
          
          {error && (
            <div className="error">
              <strong>❌ Error:</strong> {error}
            </div>
          )}
          
          {result && result.optimal_price && (
            <div className="result-success animate-in">
              <div className="result-header">
                <div className="result-icon">✨</div>
                <div>
                  <strong>{formData.productName || 'Product'}</strong> ({formData.quality})
                </div>
              </div>
              <div className="result-main-card">
                <div className="result-price-large">
                  ₹{result.optimal_price.price?.toLocaleString('en-IN')}
                  <span className="result-label">Optimal Price</span>
                </div>
              </div>
              <div className="result-details">
                <div className="result-stat-card">
                  <div className="stat-icon">📊</div>
                  <div className="stat-value">{result.optimal_price.margin}%</div>
                  <div className="stat-label">Profit Margin</div>
                </div>
                <div className="result-stat-card">
                  <div className="stat-icon">💰</div>
                  <div className="stat-value animate-number">
                    ₹{Math.floor(animatedValue || 0).toLocaleString('en-IN')}
                  </div>
                  <div className="stat-label">Expected Profit</div>
                </div>
                {result.optimal_price.expected_units_sold && (
                  <div className="result-stat-card">
                    <div className="stat-icon">📦</div>
                    <div className="stat-value">{result.optimal_price.expected_units_sold}</div>
                    <div className="stat-label">Units Expected</div>
                  </div>
                )}
              </div>
              {result.competitor_stats && (
                <div className="market-insight">
                  <div className="insight-header">📈 Market Insight</div>
                  <div className="insight-stats">
                    <span>Analyzed {result.competitor_stats.count} competitors</span>
                    <span>•</span>
                    <span>Range: ₹{result.competitor_stats.min?.toLocaleString('en-IN')} - ₹{result.competitor_stats.max?.toLocaleString('en-IN')}</span>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      <div className="features-grid">
        <div className="card feature-card">
          <h3>⚡ Fast & Accurate</h3>
          <p>Get pricing recommendations in seconds with 95% accuracy compared to manual analysis.</p>
        </div>
        <div className="card feature-card">
          <h3>📈 Boost Profits</h3>
          <p>Average profit improvement of 18% when sellers adopt our recommended pricing.</p>
        </div>
        <div className="card feature-card">
          <h3>🎯 Stay Competitive</h3>
          <p>Balance profitability with market competitiveness automatically.</p>
        </div>
      </div>
    </section>
  );
};

export default PriceOptimizer;

