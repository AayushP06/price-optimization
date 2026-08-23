import React, { useState } from 'react';

const Navbar = ({ onRunAnalysis }) => {
  const [open, setOpen] = useState(false);
  const navigate = (id) => {
    document.querySelector(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    setOpen(false);
  };

  return (
    <header className="topbar">
      <div className="topbar-inner">
        <button className="brand" onClick={() => navigate('#overview')} aria-label="Back to overview">
          <span className="brand-mark">P</span> pricely<span className="brand-dot">.</span>
        </button>
        <button className="mobile-menu" onClick={() => setOpen((value) => !value)} aria-label="Toggle navigation" aria-expanded={open}>
          <span></span><span></span>
        </button>
        <nav className={open ? 'nav-links nav-open' : 'nav-links'} aria-label="Main navigation">
          <button onClick={() => navigate('#overview')}>Overview</button>
          <button onClick={() => navigate('#workspace')}>Optimizer</button>
          <button onClick={() => navigate('#insights')}>Insights</button>
        </nav>
        <div className="topbar-actions">
          <span className="live-status"><i></i> Live analysis</span>
          <button className="button button-small" onClick={onRunAnalysis}>New analysis <span>↗</span></button>
        </div>
      </div>
    </header>
  );
};

export default Navbar;
