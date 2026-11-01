import React, { lazy, Suspense } from 'react';

const ProfitCurve = lazy(() => import('./ProfitCurve'));

const rupees = (value) => `₹${Number(value ?? 0).toLocaleString('en-IN', { maximumFractionDigits: 2 })}`;
const units = (value) => Number(value ?? 0).toLocaleString('en-IN');
const markup = (record) => record.cost_price ? (Number(record.recommended_price) - Number(record.cost_price)) / Number(record.cost_price) * 100 : 0;

export default function HistoryCompare({ records }) {
  if (records.length !== 2) return null;
  const [left, right] = records;
  const entries = [
    ['Recommended price', left.recommended_price, right.recommended_price, rupees],
    ['Markup', markup(left), markup(right), (value) => `${Number(value).toFixed(1)}%`],
    ['Expected profit', left.expected_profit, right.expected_profit, rupees],
    ['Expected units', left.expected_units_sold, right.expected_units_sold, units],
  ];
  return <section className="history-compare" aria-label="Compare optimization history">
    <div className="subsection-title"><h3>Compare runs</h3><span>Second minus first</span></div>
    <div className="compare-headings"><strong>{left.product_name || 'Product'} · {new Date(left.timestamp).toLocaleDateString()}</strong><strong>{right.product_name || 'Product'} · {new Date(right.timestamp).toLocaleDateString()}</strong></div>
    <div className="compare-grid">{entries.map(([label, a, b, format]) => <div className="compare-row" key={label}><span>{label}</span><strong>{format(a)}</strong><strong>{format(b)}</strong><em>{format(Number(b || 0) - Number(a || 0))}</em></div>)}</div>
    <Suspense fallback={<div className="profit-chart-skeleton" role="status" aria-label="Loading comparison curves"><span /><span /><span /></div>}><ProfitCurve points={left.all_prices || []} comparisonPoints={right.all_prices || []} costPrice={Number(left.cost_price)} comparisonCostPrice={Number(right.cost_price)} optimalPrice={left.recommended_price} minPrice={left.all_prices?.[0]?.price} maxPrice={left.all_prices?.at(-1)?.price} comparisonLabel={`${right.product_name || 'Second run'} expected profit`} /></Suspense>
  </section>;
}
