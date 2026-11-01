import React from 'react';
import { CartesianGrid, Line, LineChart, ReferenceArea, ReferenceDot, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';

const rupees = (value) => `₹${Number(value).toLocaleString('en-IN', { maximumFractionDigits: 2 })}`;

export default function ProfitCurve({ points = [], optimalPrice, minPrice, maxPrice, comparisonPoints, comparisonLabel = 'Comparison run', loading = false, costPrice, comparisonCostPrice }) {
  if (!points.length) return null;
  const optimal = points.reduce((best, point) => Math.abs(point.price - optimalPrice) < Math.abs(best.price - optimalPrice) ? point : best, points[0]);
  const peak = points.reduce((best, point) => Number(point.expected_profit) > Number(best.expected_profit) ? point : best, points[0]);
  const summary = `Profit peaks at ${rupees(peak.price)} with expected profit ${rupees(peak.expected_profit)}.`;
  const merged = new Map();
  [...points, ...(comparisonPoints || [])].forEach((point) => {
    const key = Number(point.price);
    merged.set(key, { ...merged.get(key), price: key });
  });
  points.forEach((point) => { merged.get(Number(point.price)).expected_profit = point.expected_profit; merged.get(Number(point.price)).markup_pct = point.markup_pct ?? (costPrice ? (Number(point.price) - costPrice) / costPrice * 100 : null); });
  (comparisonPoints || []).forEach((point) => { merged.get(Number(point.price)).compare_profit = point.expected_profit; if (merged.get(Number(point.price)).markup_pct == null) merged.get(Number(point.price)).markup_pct = point.markup_pct ?? (comparisonCostPrice ? (Number(point.price) - comparisonCostPrice) / comparisonCostPrice * 100 : null); });
  const data = [...merged.values()].sort((a, b) => a.price - b.price);
  return <section className="profit-curve" aria-label="Profit curve">
    <div className="subsection-title"><h3>Expected profit by price</h3><span>{points.length} price points evaluated</span></div>
    <p className="chart-summary">{summary}</p>
    {loading ? <div className="profit-chart-skeleton" role="status" aria-label="Loading profit curve"><span /><span /><span /></div> : <div className="profit-chart"><ResponsiveContainer width="100%" height="100%"><LineChart data={data} margin={{ top: 12, right: 16, bottom: 12, left: 8 }}>
      <CartesianGrid stroke="var(--border)" strokeDasharray="4 4" />
      <XAxis dataKey="price" type="number" domain={['dataMin', 'dataMax']} tickFormatter={(value) => `₹${Number(value).toLocaleString('en-IN')}`} />
      <YAxis tickFormatter={(value) => `₹${Number(value).toLocaleString('en-IN')}`} />
      {(minPrice != null && maxPrice != null) && <ReferenceArea x1={minPrice} x2={maxPrice} fill="var(--accent)" fillOpacity={0.08} strokeOpacity={0} />}
      <Tooltip content={({ active, payload, label }) => active && payload?.length ? <div className="chart-tooltip"><strong>Price {rupees(label)}</strong><span>Markup {payload.find((item) => item.payload?.markup_pct != null)?.payload.markup_pct?.toFixed?.(1) ?? '—'}%</span>{payload.map((item) => <span key={item.dataKey}>{item.name === comparisonLabel ? comparisonLabel : 'Expected profit'}: {rupees(item.value)}</span>)}</div> : null} />
      <Line type="monotone" dataKey="expected_profit" name="Expected profit" stroke="var(--accent)" strokeWidth={2} dot={false} isAnimationActive={false} />
      {comparisonPoints?.length > 0 && <Line type="monotone" dataKey="compare_profit" name={comparisonLabel} stroke="var(--coral)" strokeWidth={2} strokeDasharray="6 4" dot={false} isAnimationActive={false} />}
      <ReferenceDot x={optimal.price} y={optimal.expected_profit} r={5} fill="var(--accent)" stroke="var(--text-primary)" />
    </LineChart></ResponsiveContainer></div>}
    <p className="sr-only">{summary}</p>
  </section>;
}
