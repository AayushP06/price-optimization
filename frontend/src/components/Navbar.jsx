import React, { useState } from 'react';

const Navbar = () => {
  const [menuOpen, setMenuOpen] = useState(false);
  const close = () => setMenuOpen(false);

  return (
    <nav className="app-nav">
      <div className="container">
        <div className="logo">PriceOptimizer</div>

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
          <li><a href="#optimize" onClick={close}>Optimize</a></li>
          <li><a href="#optimizer-results" onClick={close}>Results</a></li>
        </ul>
      </div>
    </nav>
  );
};

export default Navbar;
