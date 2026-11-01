import React, { useState } from 'react';

const Navbar = ({ onOpenHistory }) => {
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
          <li><a href="#optimize" onClick={close}>Optimize</a></li>
          <li><a href="#how-it-works" onClick={close}>How it works</a></li>
          <li><button type="button" className="nav-link" onClick={() => { close(); onOpenHistory?.(); }}>History</button></li>
          <li><a href="#faq" onClick={close}>FAQ</a></li>
        </ul>
      </div>
    </nav>
  );
};

export default Navbar;
