import React from 'react';

export default function Hero() {
  return (
    <section id="home" className="hero">
      <div className="hero-method">
        <span className="eyebrow">PRICE OPTIMIZER METHOD</span>
        <strong>PRICE<br /><i>→ PROFIT</i></strong>
        <p>Read the market.<br />Set the right price.</p>
      </div>
      <div className="hero-title-block">
        <span className="eyebrow">COST-AWARE PRICE DISCOVERY</span>
        <h1>Find the price.<br /><em>Build the margin.</em></h1>
      </div>
      <p className="hero-copy">Use your costs, stock, and competitor prices to find a price point that protects profit and fits the market.</p>
    </section>
  );
}
