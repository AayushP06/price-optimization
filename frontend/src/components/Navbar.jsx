import React, { useState } from 'react';

const Navbar = () => {
  const [menuOpen, setMenuOpen] = useState(false);

  const toggleMenu = () => {
    setMenuOpen(!menuOpen);
  };

  const closeMenu = () => {
    setMenuOpen(false);
  };

  return (
    <nav>
      <div className="container">
        <div className="logo">Price Optimizer</div>
        <div className={`menu-toggle ${menuOpen ? 'active' : ''}`} onClick={toggleMenu}>
          <span></span>
          <span></span>
          <span></span>
        </div>
        <ul className={menuOpen ? 'active' : ''} id="navMenu">
          <li><a href="#home" onClick={closeMenu}>Home</a></li>
          <li><a href="#optimize" onClick={closeMenu}>Optimize Price</a></li>
          <li><a href="#insights" onClick={closeMenu}>Insights</a></li>
          <li><a href="#about" onClick={closeMenu}>About</a></li>
        </ul>
      </div>
    </nav>
  );
};

export default Navbar;


