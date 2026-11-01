import React from 'react';
import { CartesianGrid, Line, LineChart, ReferenceArea, ReferenceDot, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';

const rupees = (value) => `₹${Number(value).toLocaleString('en-IN', { maximumFractionDigits: 2 })}`;

export default function ProfitCurve({ points = [], optimalPrice, minPrice, maxPrice, comparisonPoints, comparisonLabel = 'Comparison run' }) {
  if (!points.length) return null;
  const optimal = points.reduce((best, point) => Math.abs(point.price - optimalPrice) < Math.abs(best.price - optimalPrice) ? point : best, points[0]);
  const peak = points.reduce((best, point) => Number(point.expected_profit) > Number(best.expected_profit) ? point : best, points[0]);
  const summary = `Profit peaks at ${rupees(peak.price)} with expected profit ${rupees(peak.expected_profit)}.`;
  const data = points.map((point, index) => ({ ...point, compare_profit: comparisonPoints?.[index]?.expected_profit }));
  return <section className="profit-curve" aria-label="Profit curve">
    <div className="subsection-title"><h3>Expected profit by price</h3><span>{points.length} price points evaluated</span></div>
    <p className="chart-summary">{summary}</p>
    <div className="profit-chart"><ResponsiveContainer width="100%" height="100%"><LineChart data={data} margin={{ top: 12, right: 16, bottom: 12, left: 8 }}>
      <CartesianGrid stroke="var(--border)" strokeDasharray="4 4" />
      <XAxis dataKey="price" type="number" domain={['dataMin', 'dataMax']} tickFormatter={(value) => `₹${Number(value).toLocaleString('en-IN')}`} />
      <YAxis tickFormatter={(value) => `₹${Number(value).toLocaleString('en-IN')}`} />
      {(minPrice != null && maxPrice != null) && <ReferenceArea x1={minPrice} x2={maxPrice} fill="var(--accent)" fillOpacity={0.08} strokeOpacity={0} />}
      <Tooltip formatter={(value, name) => [rupees(value), name === 'compare_profit' ? comparisonLabel : 'Expected profit']} labelFormatter={(value) => `Price ${rupees(value)}`} />
      <Line type="monotone" dataKey="expected_profit" name="Expected profit" stroke="var(--accent)" strokeWidth={2} dot={false} isAnimationActive={false} />
      {comparisonPoints?.length > 0 && <Line type="monotone" dataKey="compare_profit" name={comparisonLabel} stroke="var(--coral)" strokeWidth={2} strokeDasharray="6 4" dot={false} isAnimationActive={false} />}
      <ReferenceDot x={optimal.price} y={optimal.expected_profit} r={5} fill="var(--accent)" stroke="var(--text-primary)" />
    </LineChart></ResponsiveContainer></div>
    <p className="sr-only">{summary}</p>
  </section>;
}
