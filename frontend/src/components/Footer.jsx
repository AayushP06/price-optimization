import React from 'react';

const Footer = () => {
  return (
    <footer>
      <div className="footer-content" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', padding: '1rem 0' }}>
        <div className="footer-brand" style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div style={{ fontWeight: 'bold', fontSize: '1.2rem', color: 'var(--accent)' }}>PriceOptimizer Pro</div>
          <span style={{ fontSize: '0.8rem', color: 'var(--light-text)' }}>by Aayush Prasad</span>
        </div>

        <div className="footer-links" style={{ display: 'flex', gap: '20px', fontSize: '0.9rem' }}>
          <a href="#" style={{ color: 'var(--light-text)', textDecoration: 'none' }}>Solutions</a>
          <a href="#" style={{ color: 'var(--light-text)', textDecoration: 'none' }}>API</a>
          <a href="#" style={{ color: 'var(--light-text)', textDecoration: 'none' }}>Pricing</a>
          <a href="#" style={{ color: 'var(--light-text)', textDecoration: 'none' }}>Contact Sales</a>
        </div>

        <div className="footer-copyright" style={{ fontSize: '0.8rem', color: 'var(--light-text)' }}>
          © {new Date().getFullYear()} PriceOptimizer Pro SaaS. All rights reserved.
        </div>
      </div>
    </footer>
  );
};

export default Footer;


