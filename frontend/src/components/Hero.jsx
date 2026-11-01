import React from 'react';

export default function Hero() {
  return <section id="home" className="hero hero-compact">
    <div className="hero-copy-block"><p className="eyebrow">Price optimization · cost and market inputs</p><h1>Find a price that works.</h1><p>Compare expected profit across candidate prices using your costs and market assumptions.</p></div>
    <aside className="hero-example" aria-label="Example result preview">
      <span className="eyebrow">Example result · preview</span>
      <div><span>Suggested price</span><strong>₹—</strong></div>
      <div><span>Expected profit</span><strong>—</strong></div>
      <small>Values appear after an optimization.</small>
    </aside>
  </section>;
}
