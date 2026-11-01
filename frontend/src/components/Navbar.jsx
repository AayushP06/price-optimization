import React, { useState } from 'react';
import { Menu, Moon, Sun, X } from 'lucide-react';

export default function Navbar({ onOpenHistory, theme = 'dark', onThemeToggle }) {
  const [menuOpen, setMenuOpen] = useState(false);
  const close = () => setMenuOpen(false);
  const openHistory = () => { close(); onOpenHistory?.(); };

  return <nav className="site-nav" aria-label="Main navigation">
    <div className="container">
      <a className="logo" href="#home" onClick={close}><span className="logo-mark">PO</span>PriceOptimizer</a>
      <button type="button" className="menu-toggle" onClick={() => setMenuOpen((open) => !open)} aria-label={menuOpen ? 'Close navigation menu' : 'Open navigation menu'} aria-expanded={menuOpen} aria-controls="navMenu">{menuOpen ? <X size={20} /> : <Menu size={20} />}</button>
      <ul className={menuOpen ? 'active' : ''} id="navMenu">
        <li><a href="#optimize" onClick={close}>Optimize</a></li>
        <li><a href="#how-it-works" onClick={close}>How it works</a></li>
        <li><button type="button" className="nav-link" onClick={openHistory}>History</button></li>
        <li><a href="#faq" onClick={close}>FAQ</a></li>
        <li><button type="button" className="theme-toggle" onClick={onThemeToggle} aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`} title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`}>{theme === 'dark' ? <Sun size={17} /> : <Moon size={17} />}<span>{theme === 'dark' ? 'Light' : 'Dark'}</span></button></li>
      </ul>
    </div>
  </nav>;
}
