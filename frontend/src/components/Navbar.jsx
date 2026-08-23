import React, { useState, useEffect } from 'react';

const Navbar = () => {
  const [menuOpen, setMenuOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 40);
    window.addEventListener('scroll', onScroll);
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  const close = () => setMenuOpen(false);

  return (
    <nav style={{ boxShadow: scrolled ? '0 4px 24px rgba(0,0,0,0.4)' : 'none' }}>
      <div className="container">
        <div className="logo">⚡ PriceOptimizer</div>

        <div
          className={`menu-toggle ${menuOpen ? 'active' : ''}`}
          onClick={() => setMenuOpen(v => !v)}
          aria-label="Toggle menu"
        >
          <span />
          <span />
          <span />
        </div>

        <ul className={menuOpen ? 'active' : ''} id="navMenu">
          <li><a href="#home"     onClick={close}>Home</a></li>
          <li><a href="#optimize" onClick={close}>Optimize</a></li>
          <li><a href="#insights" onClick={close}>Insights</a></li>
          <li><a href="#results"  onClick={close}>Results</a></li>
          <li>
            <a
              href="#optimize"
              onClick={close}
              style={{
                background: 'linear-gradient(135deg, #f59e0b, #d97706)',
                color: '#fff',
                padding: '0.4rem 1rem',
                borderRadius: '9999px',
                fontWeight: 600,
                fontSize: '0.85rem',
              }}
            >
              Try Free →
            </a>
          </li>
        </ul>
      </div>
    </nav>
  );
};

export default Navbar;
