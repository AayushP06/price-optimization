import React, { useState } from 'react';
import Navbar from './components/Navbar';
import Hero from './components/Hero';
import PriceOptimizer from './components/PriceOptimizer';
import Results from './components/Results';
import Faq from './components/Faq';
import HistoryDrawer from './components/HistoryDrawer';
import { useOptimizer } from './hooks/useOptimizer';
import Footer from './components/Footer';

function App() {
  const optimizer = useOptimizer();
  const [historyOpen, setHistoryOpen] = useState(false);
  // Smooth scroll handler
  React.useEffect(() => {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
      anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
          target.scrollIntoView({
            behavior: 'smooth',
            block: 'start'
          });
        }
      });
    });

    // Navbar scroll effect
    let lastScroll = 0;
    const handleScroll = () => {
      const nav = document.querySelector('nav');
      const currentScroll = window.pageYOffset;

      if (currentScroll > 100) {
        nav.classList.add('scrolled');
      } else {
        nav.classList.remove('scrolled');
      }

      lastScroll = currentScroll;
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <div className="App">
      <Navbar onOpenHistory={() => setHistoryOpen(true)} />
      <Hero />
      <PriceOptimizer optimizer={optimizer} />
      <Results />
      <Faq />
      {historyOpen && <HistoryDrawer onClose={() => setHistoryOpen(false)} onLoad={(form, result) => { optimizer.setForm(form); optimizer.setResult(result); optimizer.setStale(false); optimizer.setError(null); }} />}
      <Footer />
    </div>
  );
}

export default App;


