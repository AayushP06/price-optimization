import React from 'react';

export default function Footer() {
  return <footer><div className="footer-content"><strong>PriceOptimizer</strong><span>Price recommendations from your costs and market.</span><span>© {new Date().getFullYear()}</span></div></footer>;
}
