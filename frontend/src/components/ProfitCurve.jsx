import React from 'react';
import { CartesianGrid, Line, LineChart, ReferenceDot, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';

export default function ProfitCurve({ points = [], optimalPrice }) {
  if (!points.length) return null;
  const optimal = points.reduce((best, point) => Math.abs(point.price - optimalPrice) < Math.abs(best.price - optimalPrice) ? point : best, points[0]);
  return <section className="profit-curve" aria-label="Profit curve">
    <h3>Expected profit by price</h3>
    <div className="profit-chart"><ResponsiveContainer width="100%" height="100%"><LineChart data={points} margin={{ top: 12, right: 16, bottom: 12, left: 8 }}>
      <CartesianGrid stroke="var(--border)" strokeDasharray="4 4" />
      <XAxis dataKey="price" type="number" domain={['dataMin', 'dataMax']} tickFormatter={(value) => `₹${value}`} />
      <YAxis tickFormatter={(value) => `₹${value}`} />
      <Tooltip formatter={(value) => `₹${Number(value).toLocaleString('en-IN')}`} labelFormatter={(value) => `Price ₹${Number(value).toLocaleString('en-IN')}`} />
      <Line type="monotone" dataKey="expected_profit" name="Expected profit" stroke="var(--accent)" strokeWidth={2} dot={false} />
      <ReferenceDot x={optimal.price} y={optimal.expected_profit} r={6} fill="var(--green)" stroke="var(--text-primary)" label="Optimal" />
    </LineChart></ResponsiveContainer></div>
  </section>;
}
