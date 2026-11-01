import React from 'react';
import Navbar from './components/Navbar';
import Hero from './components/Hero';
import PriceOptimizer from './components/PriceOptimizer';
import Results from './components/Results';
import Faq from './components/Faq';
import Footer from './components/Footer';

function App() {
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
      <Navbar />
      <Hero />
      <PriceOptimizer />
      <Results />
      <Faq />
      <Footer />
    </div>
  );
}

export default App;


