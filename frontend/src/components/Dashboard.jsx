import React, { useMemo, useState } from 'react';
import PriceOptimizer, { createLocalAnalysis } from './PriceOptimizer';
import ProfitChart from './ProfitChart';
import Results from './Results';

const initialForm = {
  productName: 'Nova Wireless Headphones',
  category: 'Electronics',
  costPrice: 1120,
  fixedCosts: 85,
  competitorPrices: '1499, 1549, 1599, 1649, 1699, 1749, 1799, 1849',
  rating: 4.5,
  adSpend: 2200,
};

const Dashboard = () => {
  const [analysis, setAnalysis] = useState(() => createLocalAnalysis(initialForm));
  const [source, setSource] = useState('preview');
  const summary = useMemo(() => {
    const optimal = analysis?.optimal_price || {};
    const stats = analysis?.competitor_stats || {};
    const price = Number(optimal.price || 0);
    const marketMedian = Number(stats.median || price);
    return { optimal, stats, price, marketMedian, priceDelta: marketMedian ? ((price - marketMedian) / marketMedian) * 100 : 0 };
  }, [analysis]);

  return (
    <>
      <section id="overview" className="overview-section">
        <div className="overview-copy">
          <div className="eyebrow"><span className="spark">✦</span> Revenue intelligence, simplified</div>
          <h1>Price with <em>clarity.</em><br />Grow with confidence.</h1>
          <p>Turn your costs, competitors, and demand signals into a pricing recommendation your team can act on today.</p>
        </div>
        <div className="overview-pulse" aria-hidden="true">
          <div className="pulse-grid"></div>
          <div className="pulse-card pulse-card-top"><span>Market confidence</span><strong>94<span>%</span></strong><small>+3.2% this week</small></div>
          <div className="pulse-card pulse-card-bottom"><i></i><span>Recommendation ready</span><b>↗</b></div>
          <svg viewBox="0 0 450 150" preserveAspectRatio="none"><path d="M0 125 C55 120 70 90 110 100 S165 60 205 80 S275 15 312 48 S380 28 450 5" /></svg>
        </div>
      </section>

      <section className="metric-row" aria-label="Price optimization summary">
        <div className="metric-card"><span>Recommended price</span><strong>₹{Math.round(summary.price).toLocaleString('en-IN')}</strong><small className="positive">↑ {Math.abs(summary.priceDelta).toFixed(1)}% vs. market median</small></div>
        <div className="metric-card"><span>Expected profit</span><strong>₹{Math.round(summary.optimal.expected_profit || 0).toLocaleString('en-IN')}</strong><small>Per recommended sales cycle</small></div>
        <div className="metric-card"><span>Profit margin</span><strong>{Number(summary.optimal.margin || 0).toFixed(1)}<b>%</b></strong><small className="positive">Healthy margin target</small></div>
        <div className="metric-card metric-card-accent"><span>Competitors tracked</span><strong>{summary.stats.count || 0}</strong><small>Live market sample</small></div>
      </section>

      <section id="workspace" className="workspace-section">
        <div className="section-heading">
          <div><span className="section-kicker">Pricing workspace</span><h2>Build your next recommendation.</h2></div>
          <div className={source === 'live' ? 'data-source source-live' : 'data-source'}><i></i>{source === 'live' ? 'Calculated with your API' : 'Interactive preview data'}</div>
        </div>
        <div className="workspace-grid">
          <PriceOptimizer initialForm={initialForm} onComplete={(data, resultSource) => { setAnalysis(data); setSource(resultSource); }} />
          <div className="analysis-column"><Results data={analysis} /></div>
        </div>
      </section>

      <section id="insights" className="insights-section">
        <div className="section-heading chart-heading">
          <div><span className="section-kicker">Decision map</span><h2>Where profit peaks.</h2><p>Compare the price points before you launch.</p></div>
          <div className="chart-legend"><span><i className="legend-purple"></i> Profit margin</span><span><i className="legend-green"></i> Best price</span></div>
        </div>
        <div className="chart-card"><ProfitChart data={analysis} optimalPrice={summary.price} /></div>
      </section>

      <section className="trust-row">
        <div><span className="trust-icon">◌</span><h3>Clear inputs</h3><p>Costs and market prices are visible at every step.</p></div>
        <div><span className="trust-icon">⌁</span><h3>Instant scenarios</h3><p>Explore recommendation changes as your inputs shift.</p></div>
        <div><span className="trust-icon">↗</span><h3>Actionable output</h3><p>Bring a decision-ready price straight to your store team.</p></div>
      </section>
    </>
  );
};

export default Dashboard;
