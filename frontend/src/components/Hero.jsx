import React from 'react';

const Hero = ({ onGetStarted }) => {
  return (
    <section id="home" className="hero" style={{ padding: '6rem 2rem', textAlign: 'center' }}>
      <div className="hero-content animate-in" style={{ maxWidth: '800px', margin: '0 auto' }}>
        <div style={{ display: 'inline-block', padding: '0.5rem 1rem', background: 'rgba(16, 185, 129, 0.1)', color: 'var(--accent-primary)', borderRadius: '50px', fontSize: '0.875rem', fontWeight: 'bold', marginBottom: '1.5rem', border: '1px solid rgba(16, 185, 129, 0.3)' }}>
          Version 2.0 is Live
        </div>
        
        <h1 style={{ fontSize: '3.5rem', lineHeight: '1.2', marginBottom: '1.5rem', color: 'var(--text-main)' }}>
          Find Your Perfect <span style={{ color: 'var(--accent-primary)' }}>Price</span> in Seconds
        </h1>
        
        <p style={{ fontSize: '1.25rem', color: 'var(--text-muted)', marginBottom: '2.5rem', lineHeight: '1.6' }}>
          Stop guessing. Use data-driven analytics to maximize your profit margins while remaining highly competitive in your market.
        </p>
        
        <button 
          onClick={onGetStarted}
          className="btn"
          style={{
            padding: '1rem 2rem',
            fontSize: '1.1rem',
          }}
        >
          Start Optimizing Now →
        </button>
      </div>
    </section>
  );
};

export default Hero;