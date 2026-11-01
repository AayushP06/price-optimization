import React, { useState } from 'react';

const Navbar = () => {
  const [menuOpen, setMenuOpen] = useState(false);
  const close = () => setMenuOpen(false);

  return (
    <nav>
      <div className="container">
        <a className="logo" href="#home"><span className="logo-mark">PO</span>PriceOptimizer</a>

        <button
          className={`menu-toggle ${menuOpen ? 'active' : ''}`}
          type="button"
          onClick={() => setMenuOpen((current) => !current)}
          aria-label={menuOpen ? 'Close navigation menu' : 'Open navigation menu'}
          aria-expanded={menuOpen}
          aria-controls="navMenu"
        >
          <span />
          <span />
          <span />
        </button>

        <ul className={menuOpen ? 'active' : ''} id="navMenu">
          <li><a href="#home"     onClick={close}>Home</a></li>
          <li><a href="#optimize" onClick={close}>Optimize</a></li>
          <li><a href="#insights" onClick={close}>Insights</a></li>
          <li><a href="#results"  onClick={close}>Results</a></li>
          <li>
            <a href="#optimize" onClick={close}>Try Free →</a>
          </li>
        </ul>
      </div>
    </nav>
  );
};

export default Navbar;
