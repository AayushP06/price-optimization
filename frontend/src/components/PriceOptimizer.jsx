import React, { useState, useCallback } from 'react';

// ── Product-name validator ───────────────────────────────────────────────────
// Returns a warning string when the input looks like a non-product, else null.
const KEYBOARD_SPAM = /^([a-z])\1{3,}$/i;           // aaaa, zzzz
const GIBBERISH     = /^[^aeiou\s]{5,}$/i;           // bcdfghjkl (no vowels)
const ONLY_NUMBERS  = /^[\d\s.,/\\-]+$/;             // 12345, 99.99
const ONLY_SYMBOLS  = /^[^a-z0-9]+$/i;               // !@#$%
const COMMON_WORDS  = /\b(test|asdf|qwerty|hello|world|foo|bar|baz|abc|xyz|lol|haha|temp|dummy|sample|blah|null|undefined|random|idk|wtf|omg|ugh|hey|hi|yo)\b/i;

function validateProductName(value) {
  const v = value.trim();
  if (!v) return null;
  if (v.length < 2)          return '⚠ Too short — please enter a recognisable product name.';
  if (ONLY_NUMBERS.test(v))  return '⚠ This looks like a number, not a product name.';
  if (ONLY_SYMBOLS.test(v))  return '⚠ Symbols alone don\'t describe a product. Try a real name.';
  if (KEYBOARD_SPAM.test(v)) return '⚠ That looks like repeated characters. Enter an actual product name.';
  if (GIBBERISH.test(v) && v.length > 3)
                             return '⚠ Unrecognised pattern — does your product have a proper name?';
  if (COMMON_WORDS.test(v))  return '⚠ "' + v + '" doesn\'t look like a product name. Try something specific.';
  // Too many digits relative to total chars (e.g. "a1b2c3d4")
  const digitRatio = (v.match(/\d/g) || []).length / v.length;
  if (digitRatio > 0.6 && v.length > 4)
                             return '⚠ Mostly digits detected — is this really a product name?';
  return null;
}

const currency = (value) => `₹${Math.round(Number(value || 0)).toLocaleString('en-IN')}`;

const readPrices = (value) => String(value || '')
  .split(/[\s,\n]+/)
  .map((price) => Number(price.trim()))
  .filter((price) => Number.isFinite(price) && price > 0)
  .sort((a, b) => a - b);

export const createLocalAnalysis = (form) => {
  const cost = Math.max(1, Number(form.costPrice) || 0);
  const fixed = Math.max(0, Number(form.fixedCosts) || 0);
  const suppliedPrices = readPrices(form.competitorPrices);
  const competitors = suppliedPrices.length ? suppliedPrices : [1.22, 1.28, 1.34, 1.39, 1.43, 1.48, 1.53, 1.59].map((multiplier) => Math.round(cost * multiplier));
  const midpoint = competitors[Math.floor(competitors.length / 2)];
  const candidates = Array.from({ length: 11 }, (_, index) => Math.round((cost * 1.17) + ((midpoint * 1.08 - cost * 1.17) * index) / 10));
  const allPrices = candidates.map((price) => {
    const position = competitors.filter((competitor) => competitor <= price).length / competitors.length;
    const demand = Math.max(18, Math.round(118 - position * 76 - Math.max(0, (price - midpoint) / midpoint) * 25));
    const profitPerUnit = Math.max(0, price - cost - fixed);
    return {
      price,
      margin: Number(((profitPerUnit / price) * 100).toFixed(1)),
      demand,
      expected_units_sold: demand,
      expected_profit: Math.round(profitPerUnit * demand),
    };
  });
  const optimal = allPrices.reduce((best, candidate) => candidate.expected_profit > best.expected_profit ? candidate : best, allPrices[0]);
  const total = competitors.reduce((sum, price) => sum + price, 0);
  return {
    success: true,
    optimal_price: optimal,
    all_prices: allPrices,
    competitor_prices: competitors,
    competitor_stats: {
      count: competitors.length,
      min: competitors[0],
      max: competitors[competitors.length - 1],
      median: midpoint,
      mean: Math.round(total / competitors.length),
    },
  };
};

const PriceOptimizer = ({ initialForm, onComplete }) => {
  const [form, setForm] = useState(initialForm);
  const [loading, setLoading] = useState(false);
  const [note, setNote] = useState('');
  const [nameWarning, setNameWarning] = useState(null);

  const updateField = useCallback((event) => {
    const { name, value } = event.target;
    setForm((previous) => ({ ...previous, [name]: value }));
    setNote('');
    if (name === 'productName') {
      setNameWarning(validateProductName(value));
    }
  }, []);

  const handleSubmit = async (event) => {
    event.preventDefault();
    const cost = Number(form.costPrice);
    if (!Number.isFinite(cost) || cost <= 0) {
      setNote('Add a valid unit cost to generate a recommendation.');
      return;
    }

    setLoading(true);
    setNote('');
    const fallback = createLocalAnalysis(form);
    const payload = {
      product_name: form.productName,
      cost_price: cost,
      fixed_costs: Number(form.fixedCosts) || 0,
      competitor_prices: readPrices(form.competitorPrices),
      num_competitors: 50,
      category: form.category,
      month: new Date().getMonth() + 1,
      rating: Number(form.rating) || 4.2,
      ad_spend: Number(form.adSpend) || 0,
    };

    try {
      const response = await fetch('/api/optimize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const result = await response.json();
      if (!response.ok || !result.success || !result.optimal_price) throw new Error(result.error || 'Recommendation unavailable');
      onComplete(result, 'live');
    } catch {
      onComplete(fallback, 'preview');
      setNote('Showing a local scenario while your API is offline. Start the backend to use the trained model.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <form className="optimizer-card" onSubmit={handleSubmit}>
      <div className="card-intro">
        <div><span className="panel-label">1 · Product inputs</span><h3>Tell us about your offer.</h3></div>
        <span className="form-state">Required fields marked <b>•</b></span>
      </div>

      <div className="form-grid">
        <label className={`form-field field-wide${nameWarning ? ' field-warning' : ''}`}>
          <span>Product name <b>•</b></span>
          <input
            name="productName"
            value={form.productName}
            onChange={updateField}
            placeholder="e.g. Wireless headphones"
            autoComplete="off"
            required
          />
          {nameWarning && (
            <span className="name-warning">
              {nameWarning}
              <span className="name-warning-tip">Suggestion: use the actual name of your product, e.g. <em>Noise-Cancelling Earbuds</em> or <em>Stainless Steel Water Bottle</em>.</span>
            </span>
          )}
        </label>
        <label className="form-field"><span>Category</span><select name="category" value={form.category} onChange={updateField}><option>Electronics</option><option>Home & Living</option><option>Beauty</option><option>Fashion</option><option>Sports</option></select></label>
        <label className="form-field"><span>Product rating</span><div className="input-with-suffix"><input name="rating" type="number" min="1" max="5" step="0.1" value={form.rating} onChange={updateField} /><span>★</span></div></label>
        <label className="form-field"><span>Unit cost <b>•</b></span><div className="input-with-prefix"><span>₹</span><input name="costPrice" type="number" min="1" step="1" value={form.costPrice} onChange={updateField} required /></div></label>
        <label className="form-field"><span>Fixed cost / unit</span><div className="input-with-prefix"><span>₹</span><input name="fixedCosts" type="number" min="0" step="1" value={form.fixedCosts} onChange={updateField} /></div></label>
        <label className="form-field field-wide"><span>Competitor prices <small>Optional — comma-separated</small></span><textarea name="competitorPrices" value={form.competitorPrices} onChange={updateField} rows="3" placeholder="1499, 1599, 1699" /></label>
        <label className="form-field"><span>Monthly ad spend</span><div className="input-with-prefix"><span>₹</span><input name="adSpend" type="number" min="0" step="100" value={form.adSpend} onChange={updateField} /></div></label>
        <div className="market-hint"><span>⌁</span><p>We balance demand against your margin to locate the profitable sweet spot.</p></div>
      </div>

      {note && <p className="form-note">{note}</p>}
      <div className="form-footer"><span><i></i> Your information stays in this session</span><button className="button button-primary" disabled={loading}>{loading ? 'Calculating…' : 'Generate recommendation'} <b>→</b></button></div>
    </form>
  );
};

export default PriceOptimizer;
