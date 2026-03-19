import React, { useState } from 'react';
import Navbar from './components/Navbar';
import Hero from './components/Hero';
import PriceOptimizer from './components/PriceOptimizer';
import ProfitChart from './components/ProfitChart';
import Results from './components/Results';
import Footer from './components/Footer';

function App() {
  const [chartData, setChartData] = useState(null);
  const [optimalPrice, setOptimalPrice] = useState(null);

  const handleOptimize = (data) => {
    setChartData(data);
    setOptimalPrice(data?.optimal_price?.price);
  };

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
        nav.style.boxShadow = '0 5px 20px rgba(0,0,0,0.3)';
      } else {
        nav.style.boxShadow = 'none';
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
      <PriceOptimizer onOptimize={handleOptimize} />
      <ProfitChart data={chartData} optimalPrice={optimalPrice} />
      <Results />
      <Footer />
    </div>
  );
}

export default App;


