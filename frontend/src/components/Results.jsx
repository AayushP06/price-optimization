import React from 'react';

const money = (value) => `₹${Math.round(Number(value || 0)).toLocaleString('en-IN')}`;

const Results = ({ data }) => {
  const result = data?.optimal_price || {};
  const market = data?.competitor_stats || {};
  const products = data?.competitor_prices || [];
  const recommended = Number(result.price || 0);
  const median = Number(market.median || 0);
  const delta = median ? ((recommended - median) / median) * 100 : 0;

  return (
    <aside className="recommendation-card">
      <div className="recommendation-topline"><span className="panel-label">2 · Recommendation</span><span className="confidence-badge"><i></i> High confidence</span></div>
      <p className="recommendation-eyebrow">YOUR RECOMMENDED PRICE</p>
      <div className="recommended-price">{money(recommended)}<span>/ unit</span></div>
      <p className="recommended-description">Positioned {Math.abs(delta).toFixed(1)}% {delta >= 0 ? 'above' : 'below'} the market midpoint while protecting a healthy return.</p>
      <div className="recommendation-metrics">
        <div><span>Expected profit</span><strong>{money(result.expected_profit)}</strong><small>per sales cycle</small></div>
        <div><span>Profit margin</span><strong>{Number(result.margin || 0).toFixed(1)}%</strong><small>at this price</small></div>
        <div><span>Expected demand</span><strong>{Math.round(result.expected_units_sold || result.demand || 0)}</strong><small>units estimated</small></div>
      </div>

      <div className="market-summary-card">
        <div className="market-summary-title"><span>Market position</span><strong>{market.count || 0} offers tracked</strong></div>
        <div className="range-bar"><i style={{ left: `${Math.max(4, Math.min(94, median ? ((recommended - Number(market.min || 0)) / Math.max(1, Number(market.max || 0) - Number(market.min || 0))) * 100 : 50))}%` }}></i></div>
        <div className="range-values"><span>Low {money(market.min)}</span><span>Median {money(median)}</span><span>High {money(market.max)}</span></div>
      </div>

      <div className="competitive-list"><div><span>Closest market prices</span><button type="button" onClick={() => document.querySelector('#insights')?.scrollIntoView({ behavior: 'smooth' })}>View curve ↗</button></div><div className="price-chips">{products.slice(0, 6).map((price, index) => <span key={`${price}-${index}`} className={Math.abs(price - recommended) < Math.max(80, recommended * 0.05) ? 'price-chip price-chip-active' : 'price-chip'}>{money(price)}</span>)}</div></div>
    </aside>
  );
};

export default Results;
