import React, { useEffect, useState } from 'react';
import Navbar from './components/Navbar';
import Hero from './components/Hero';
import PriceOptimizer from './components/PriceOptimizer';
import Results from './components/Results';
import Faq from './components/Faq';
import HistoryDrawer from './components/HistoryDrawer';
import Footer from './components/Footer';
import { useOptimizer } from './hooks/useOptimizer';

function preferredTheme() {
  try {
    const saved = window.localStorage.getItem('priceoptimizer.theme');
    if (saved === 'light' || saved === 'dark') return saved;
  } catch { /* storage may be disabled */ }
  return window.matchMedia?.('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
}

function App() {
  const optimizer = useOptimizer();
  const [historyOpen, setHistoryOpen] = useState(false);
  const [theme, setTheme] = useState(preferredTheme);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try { window.localStorage.setItem('priceoptimizer.theme', theme); } catch { /* theme still works for this visit */ }
  }, [theme]);

  return <div className="App" data-theme={theme}>
    <Navbar onOpenHistory={() => setHistoryOpen(true)} theme={theme} onThemeToggle={() => setTheme((current) => current === 'dark' ? 'light' : 'dark')} />
    <main>
      <Hero />
      <PriceOptimizer optimizer={optimizer} />
      <Results />
      <Faq />
    </main>
    <Footer />
    {historyOpen && <HistoryDrawer onClose={() => setHistoryOpen(false)} onLoad={(form, result) => { optimizer.setForm(form); optimizer.setResult(result); optimizer.setStale(false); optimizer.setError(null); }} />}
  </div>;
}

export default App;
