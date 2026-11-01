import React from 'react';

export default function MarketPosition({ stats, prices = [], optimalPrice }) {
  if (!stats || !Number.isFinite(Number(stats.min)) || !Number.isFinite(Number(stats.max))) return null;
  const min = Number(stats.min);
  const max = Math.max(min + 0.01, Number(stats.max));
  const left = (value) => `${Math.min(100, Math.max(0, (Number(value) - min) / (max - min) * 100))}%`;
  const ticks = prices.slice(0, 150);
  return <section className="market-position" aria-label={`Competitor prices range from ₹${min.toLocaleString('en-IN')} to ₹${max.toLocaleString('en-IN')}; median ₹${Number(stats.median).toLocaleString('en-IN')}; recommendation ₹${Number(optimalPrice).toLocaleString('en-IN')}`}>
    <div className="subsection-title"><h3>Market position</h3><span>{stats.count} prices</span></div>
    <div className="market-track" role="img" aria-label="Competitor price strip plot with minimum, median, maximum, and recommended price markers">
      {ticks.map((price, index) => <span className="competitor-dot" key={`${price}-${index}`} style={{ left: left(price) }} />)}
      <span className="market-marker market-min" style={{ left: left(min) }}><i />Min</span>
      <span className="market-marker market-median" style={{ left: left(stats.median) }}><i />Median</span>
      <span className="market-marker market-max" style={{ left: left(max) }}><i />Max</span>
      <span className="market-marker market-optimal" style={{ left: left(optimalPrice) }}><i />Your price</span>
    </div>
    <div className="market-range"><span>₹{min.toLocaleString('en-IN')}</span><span>Median ₹{Number(stats.median).toLocaleString('en-IN')}</span><span>₹{max.toLocaleString('en-IN')}</span></div>
  </section>;
}
