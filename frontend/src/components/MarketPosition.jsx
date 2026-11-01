import React from 'react';

export default function MarketPosition({ stats }) {
  if (!stats) return null;
  return <section className="market-position" aria-label="Competitor market position">
    <h3>Market position</h3>
    <p>{stats.count} competitors · ₹{Number(stats.min).toLocaleString('en-IN')}–₹{Number(stats.max).toLocaleString('en-IN')}</p>
    <p>Median ₹{Number(stats.median).toLocaleString('en-IN')} · Average ₹{Number(stats.mean).toLocaleString('en-IN')}</p>
  </section>;
}
