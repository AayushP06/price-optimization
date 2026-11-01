import React from 'react';
import MarketPosition from './MarketPosition';
import ProfitCurve from './ProfitCurve';

const rupees = (value) => `₹${Number(value ?? 0).toLocaleString('en-IN', { maximumFractionDigits: 2 })}`;

export function resultFlags(result, form = {}) {
  if (!result?.optimal_price) return [];
  const optimum = Number(result.optimal_price.price);
  const stats = result.competitor_stats || {};
  const prices = result.competitor_prices || [];
  const lower = Number(result.min_markup_pct ?? form.min_markup_pct ?? 5);
  const upper = Number(result.max_markup_pct ?? form.max_markup_pct ?? 80);
  const cost = Number(result.cost_price ?? form.cost_price ?? 0);
  const lowerPrice = cost * (1 + lower / 100);
  const upperPrice = cost * (1 + upper / 100);
  const flags = [];
  const epsilon = Math.max(0.01, optimum * 0.0001);
  if (Math.abs(optimum - lowerPrice) <= epsilon || Math.abs(optimum - upperPrice) <= epsilon) flags.push({ severity: 'warning', text: 'At markup range limit' });
  if (result.simulated_competitors) flags.push({ severity: 'warning', text: 'Simulated competitor data' });
  if (result.method === 'heuristic-fallback') flags.push({ severity: 'warning', text: 'Heuristic fallback used' });
  if (Number.isFinite(Number(stats.median)) && Number(stats.median) > 0) {
    const difference = (optimum - Number(stats.median)) / Number(stats.median) * 100;
    if (Math.abs(difference) > 0.01) flags.push({ severity: 'info', text: `${Math.abs(difference).toFixed(1)}% ${difference > 0 ? 'above' : 'below'} competitor median` });
  }
  const available = result.available_units ?? form.available_units;
  if (available !== '' && available != null && Number(available) >= 0 && Number(result.optimal_price.expected_units_sold) >= Number(available)) flags.push({ severity: 'info', text: 'Units capped by available stock' });
  if (prices.length >= 4) {
    const sorted = prices.map(Number).filter(Number.isFinite).sort((a, b) => a - b);
    const q1 = sorted[Math.floor((sorted.length - 1) * 0.25)];
    const q3 = sorted[Math.ceil((sorted.length - 1) * 0.75)];
    if (optimum >= q1 && optimum <= q3) flags.push({ severity: 'success', text: 'Within middle 50% of competitor prices' });
  }
  return flags;
}

export default function ResultCard({ result, form, loading, error, stale }) {
  const panelClass = `result-panel${stale ? ' is-stale' : ''}`;
  if (!result) return <section className="result-container" aria-live="polite" aria-label="Optimization result">
    {loading ? <div className="result-skeleton" role="status" aria-label="Loading recommendation"><span /><span /><span /><span /></div> : <div className="result-empty"><span className="eyebrow">Example · illustrative only</span><strong>Review price and profit together.</strong><p>Run an optimization to see your recommendation, market position, and profit curve here.</p><div className="example-result"><span>Suggested price</span><strong>₹1,249</strong><span>Expected profit · sample</span><strong>₹8,420</strong></div><small>Example values are not a model result.</small></div>}
    {error && <p className="inline-error" role="alert">{error.message}</p>}
  </section>;

  const optimum = result.optimal_price;
  const flags = resultFlags(result, form);
  const cost = Number(result.cost_price ?? form?.cost_price ?? 0);
  const expectedProfit = Number(optimum.expected_profit ?? 0);
  const expectedUnits = Number(optimum.expected_units_sold ?? 0);
  const profitPerUnit = Number(optimum.profit_per_unit ?? (Number(optimum.price) - cost - Number(result.fixed_costs ?? form?.fixed_costs ?? 0)));
  const markup = Number(optimum.markup_pct ?? (cost ? (Number(optimum.price) - cost) / cost * 100 : 0));
  const margin = Number(optimum.margin_pct ?? optimum.margin ?? 0);
  const simulated = result.simulated_competitors ? 'Simulated' : 'User supplied';
  const assumptions = [
    ['Method', result.method || 'Not specified'],
    ['Competitor prices', `${result.competitor_stats?.count ?? 0} · ${simulated.toLowerCase()}`],
    ['Markup range', `${result.min_markup_pct ?? form?.min_markup_pct ?? '—'}%–${result.max_markup_pct ?? form?.max_markup_pct ?? '—'}%`],
    ['Category', result.category ?? form?.category ?? '—'],
    ['Month', result.month ?? form?.month ?? '—'],
    ['Rating', result.rating ?? form?.rating ?? '—'],
    ['Ad spend', rupees(result.ad_spend ?? form?.ad_spend)],
    ['Quality tier', `${result.quality ?? form?.quality ?? 'standard'} · metadata only`],
  ];

  return <section id="optimizer-results" className={`result-container ${panelClass}`} aria-label="Optimization results" aria-live="polite" aria-busy={loading}>
    <header className="result-title"><div><p className="eyebrow">Recommended price · {result.quality || 'standard'} tier</p><h2>{result.product_name || form?.product_name || 'Product'}</h2></div>{stale && <span className="stale-badge">Inputs changed · result is stale</span>}</header>
    {error && <p className="inline-error" role="alert">{error.message}</p>}
    {loading && <div className="refreshing-note" role="status">Updating recommendation…</div>}
    <p className="recommended-price">{rupees(optimum.price)}</p>
    <p className="metric-caption">Suggested selling price</p>
    <div className="result-metrics">
      <div><strong>{rupees(expectedProfit)}</strong><span>Expected profit</span></div>
      <div><strong>{rupees(profitPerUnit)}</strong><span>Profit per unit</span></div>
      <div><strong>{markup.toFixed(1)}%</strong><span title="(price − cost) ÷ cost">Markup <button type="button" className="help-tip" aria-label="Markup means price minus cost, divided by cost.">i</button></span></div>
      <div><strong>{margin.toFixed(1)}%</strong><span title="(price − cost − fixed costs) ÷ price">Net margin <button type="button" className="help-tip" aria-label="Margin means price minus cost and fixed costs, divided by price.">i</button></span></div>
      <div><strong>{expectedUnits.toLocaleString('en-IN')}</strong><span>Expected units</span></div>
    </div>
    {flags.length > 0 && <ul className="result-flags" aria-label="Recommendation notes">{flags.map((flag, index) => <li className={`flag flag-${flag.severity}`} key={`${flag.severity}-${index}`}><span aria-hidden="true">{flag.severity === 'success' ? '✓' : flag.severity === 'warning' ? '!' : 'i'}</span>{flag.text}</li>)}</ul>}
    <MarketPosition stats={result.competitor_stats} prices={result.competitor_prices} optimalPrice={optimum.price} />
    <ProfitCurve points={result.all_prices} optimalPrice={optimum.price} minPrice={cost * (1 + Number(result.min_markup_pct ?? form?.min_markup_pct ?? 5) / 100)} maxPrice={cost * (1 + Number(result.max_markup_pct ?? form?.max_markup_pct ?? 80) / 100)} />
    <details className="assumptions-panel"><summary>Method & assumptions</summary><dl>{assumptions.map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}</dl><div className="formula-notes"><p><strong>Markup:</strong> (price − cost) ÷ cost.</p><p><strong>Margin:</strong> (price − cost − fixed costs) ÷ price.</p><p>Quality is saved as metadata only and is not used by the model.</p></div></details>
  </section>;
}
