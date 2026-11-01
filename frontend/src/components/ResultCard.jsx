import React from 'react';
import MarketPosition from './MarketPosition';
import ProfitCurve from './ProfitCurve';

export default function ResultCard({ result, loading, error, stale }) {
  if (loading) return <section className="result-panel" aria-live="polite"><p className="loading">Calculating recommendation…</p></section>;
  if (error && !result) return <section className="result-panel" role="alert"><h2>Unable to calculate</h2><p>{error.message}</p></section>;
  if (!result) return <section className="result-panel"><h2>Your recommendation</h2><p>Complete the product details to see the suggested price, market position, and profit curve.</p></section>;
  const optimum = result.optimal_price;
  return <section id="optimizer-results" className={`result-panel${stale ? ' is-stale' : ''}`} aria-label="Optimization results">
    <header className="result-title"><div><p className="eyebrow">Recommended price</p><h2>{result.product_name || 'Product'} · {result.quality || 'standard'}</h2></div>{stale && <span className="stale-badge">Inputs changed</span>}</header>
    {error && <p className="inline-error" role="alert">{error.message}</p>}
    <p className="recommended-price">₹{Number(optimum.price).toLocaleString('en-IN')}</p>
    <div className="result-metrics"><div><strong>{optimum.margin_pct ?? optimum.margin}%</strong><span>Net margin</span></div><div><strong>₹{Number(optimum.expected_profit).toLocaleString('en-IN')}</strong><span>Expected profit</span></div><div><strong>{optimum.expected_units_sold}</strong><span>Units expected</span></div></div>
    <div className="result-badges"><span>Method: {result.method}</span>{result.simulated_competitors && <span className="simulation-badge">Simulated competitor data</span>}</div>
    {result.warnings?.length > 0 && <ul className="result-warnings">{result.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul>}
    <MarketPosition stats={result.competitor_stats} />
    <ProfitCurve points={result.all_prices} optimalPrice={optimum.price} />
  </section>;
}
