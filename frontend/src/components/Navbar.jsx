import React, { useState } from 'react';

const Navbar = () => {
  const [menuOpen, setMenuOpen] = useState(false);
  const close = () => setMenuOpen(false);

  return (
    <nav>
      <div className="container">
        <a className="logo" href="#home"><span className="logo-mark">PO</span>PriceOptimizer</a>

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
            <a href="#optimize" onClick={close}>Try Free →</a>
          </li>
        </ul>
      </div>
    </nav>
  );
};

export default Navbar;
