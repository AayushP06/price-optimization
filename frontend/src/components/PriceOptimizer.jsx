import React, { useState, useEffect } from 'react';

const PriceOptimizer = ({ onOptimize }) => {
  const [formData, setFormData] = useState({
    productName: '',
    quality: 'standard',
    costPrice: 1000,
    fixedCosts: 0,
    availableUnits: '',
    numCompetitors: 100,
    competitorPrices: '',
    category: 'Electronics',
    month: 6,
    rating: 4.2,
    adSpend: 1000,
    minMarkup: 5,
    maxMarkup: 80
  });

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [animatedValue, setAnimatedValue] = useState(0);
  const [stale, setStale] = useState(false);

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
    if (result) setStale(true);
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
        num_competitors: competitorPrices.length > 0 ? undefined : numCompetitors,
        category: formData.category,
        month: parseInt(formData.month),
        rating: parseFloat(formData.rating),
        ad_spend: parseFloat(formData.adSpend),
        min_markup_pct: parseFloat(formData.minMarkup),
        max_markup_pct: parseFloat(formData.maxMarkup)
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

      setResult(json);
      setStale(false);
      if (onOptimize) {
        onOptimize(json);
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
        <form id="optForm" className="price-form" onSubmit={handleSubmit} aria-busy={loading}>
          <div className="form-section form-wide">
            <div className="form-section-heading"><span>01</span><div><h3>Product & costs</h3><p>Start with the item and what it costs you.</p></div></div>
            <div className="field-grid">
              <div><label htmlFor="productName">Product name <small>(optional)</small></label><input id="productName" name="productName" placeholder="e.g. Wireless earbuds" value={formData.productName} onChange={handleChange} /></div>
              <div><label htmlFor="category">Category</label><select id="category" name="category" value={formData.category} onChange={handleChange}><option>Electronics</option><option>Clothing</option><option>Home</option><option>Beauty</option><option>Sports</option></select></div>
              <div><label htmlFor="costPrice">Unit cost (₹)</label><input id="costPrice" name="costPrice" type="number" min="0.01" step="0.01" value={formData.costPrice} onChange={handleChange} required /><small>Required</small></div>
              <div><label htmlFor="fixedCosts">Other costs per unit (₹)</label><input id="fixedCosts" name="fixedCosts" type="number" min="0" step="0.01" value={formData.fixedCosts} onChange={handleChange} /><small>Optional · defaults to 0</small></div>
            </div>
          </div>

          <div className="form-section form-wide">
            <div className="form-section-heading"><span>02</span><div><h3>Market inputs</h3><p>Use a simulation or add prices you collected.</p></div></div>
            <div className="field-grid">
              <div><label htmlFor="numCompetitors">Simulated competitors</label><input id="numCompetitors" name="numCompetitors" type="number" min="1" max="5000" value={formData.numCompetitors} onChange={handleChange} /><small>Used when manual prices are blank</small></div>
              <div><label htmlFor="competitorPrices">Competitor prices <small>(optional)</small></label><textarea id="competitorPrices" name="competitorPrices" rows="2" placeholder="1299, 1549, 1799" value={formData.competitorPrices} onChange={handleChange} /><small>Separate prices with commas or spaces</small></div>
            </div>
          </div>

          <details className="form-section form-wide advanced-settings">
            <summary><span className="advanced-label">More details</span><span className="advanced-hint">Quality, stock, timing, and markup range</span></summary>
            <div className="field-grid advanced-grid">
              <div><label htmlFor="quality">Quality tier</label><select id="quality" name="quality" value={formData.quality} onChange={handleChange}><option value="budget">Budget</option><option value="standard">Standard</option><option value="premium">Premium</option></select><small>Saved as product metadata</small></div>
              <div><label htmlFor="month">Launch month</label><select id="month" name="month" value={formData.month} onChange={handleChange}>{['January','February','March','April','May','June','July','August','September','October','November','December'].map((month, i) => <option value={i + 1} key={month}>{month}</option>)}</select></div>
              <div><label htmlFor="rating">Expected rating</label><input id="rating" name="rating" type="number" min="1" max="5" step="0.1" value={formData.rating} onChange={handleChange} /></div>
              <div><label htmlFor="adSpend">Ad spend (₹)</label><input id="adSpend" name="adSpend" type="number" min="0" step="100" value={formData.adSpend} onChange={handleChange} /></div>
              <div><label htmlFor="availableUnits">Available stock</label><input id="availableUnits" name="availableUnits" type="number" min="0" step="1" placeholder="No limit" value={formData.availableUnits} onChange={handleChange} /></div>
              <div><label>Allowed markup range</label><div className="range-fields"><input aria-label="Minimum markup percent" name="minMarkup" type="number" min="0" value={formData.minMarkup} onChange={handleChange} /><span>to</span><input aria-label="Maximum markup percent" name="maxMarkup" type="number" min="0" value={formData.maxMarkup} onChange={handleChange} /><span>%</span></div></div>
            </div>
          </details>

          {error && <div className="form-error form-wide" role="alert">{error}</div>}
          <button
            type="submit"
            className={`btn btn-primary form-wide${loading ? ' is-loading' : ''}`}
            disabled={loading}
          >
            <span>{loading ? 'Calculating…' : 'Get my price recommendation'}</span>
          </button>
        </form>

        <div id="optOutput" className="result-container" aria-live="polite" aria-atomic="true">
          {loading && (
            <div className="result-empty result-loading" role="status"><span className="eyebrow">Working on it</span><strong>Comparing price points…</strong><p>Estimating expected profit from your cost and market inputs.</p></div>
          )}
          
          {error && !result && (
            <div className="result-empty result-error">
              <span className="eyebrow">ANALYSIS PAUSED</span>
              <strong>We couldn&apos;t complete this run.</strong>
              <p>Review the message beside the form, then try again.</p>
            </div>
          )}
          
          {result && result.optimal_price && (
            <div className={`result-success animate-in${stale ? ' result-stale' : ''}`}>
              <div className="result-header">
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
              <div className="result-badges"><span>Method: {result.method || 'Price optimizer'}</span>{result.simulated_competitors && <span>Simulated market prices</span>}{stale && <span>Inputs changed · recalculate to refresh</span>}</div>
              <div className="result-details">
                <div className="result-stat-card">
                  <div className="stat-value">{result.optimal_price.margin_pct ?? result.optimal_price.margin}%</div>
                  <div className="stat-label">Net margin</div>
                </div>
                <div className="result-stat-card">
                  <div className="stat-value animate-number">
                    ₹{Math.floor(animatedValue || 0).toLocaleString('en-IN')}
                  </div>
                  <div className="stat-label">Expected Profit</div>
                </div>
                {result.optimal_price.expected_units_sold && (
                  <div className="result-stat-card">
                    <div className="stat-value">{result.optimal_price.expected_units_sold}</div>
                    <div className="stat-label">Units Expected</div>
                  </div>
                )}
              </div>
              {result.competitor_stats && (
                <div className="market-insight">
                  <div className="insight-header">Market Insight</div>
                  <div className="insight-stats">
                    <span>Analyzed {result.competitor_stats.count} competitors</span>
                    <span>•</span>
                    <span>Range: ₹{result.competitor_stats.min?.toLocaleString('en-IN')} - ₹{result.competitor_stats.max?.toLocaleString('en-IN')}</span>
                  </div>
                </div>
              )}
            </div>
          )}
          {!loading && !error && !result && <div className="result-empty"><span className="eyebrow">Your recommendation</span><strong>Ready when you are.</strong><p>Enter a unit cost and choose your competitor data, then get a price recommendation.</p><div className="empty-result-note"><span />Results appear here</div></div>}
        </div>
      </div>

    </section>
  );
};

export default PriceOptimizer;

