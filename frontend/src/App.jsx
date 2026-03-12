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
      
      <section id="future">
        <h2>Coming Soon</h2>
        <p className="section-subtitle">Upcoming features to help you grow your business</p>
        <ComingSoonFeatures />
      </section>

      <Footer />
    </div>
  );
}

const ComingSoonFeatures = () => {
  const features = [
    {
      title: 'AI-Powered Predictions',
      description: 'Predict future price trends and seasonal demand patterns to help you plan ahead and maximize profits during peak seasons.'
    },
    {
      title: 'Automatic Competitor Monitoring',
      description: 'Continuously monitor competitor prices across multiple platforms in real-time, so you\'re always aware of market changes.'
    },
    {
      title: 'Interactive Dashboard',
      description: 'Visualize pricing trends, track performance, and adjust your strategy with an easy-to-use dashboard designed for sellers.'
    },
    {
      title: 'Auto-Pricing',
      description: 'Automatically adjust your prices based on market changes, keeping you competitive without constant manual updates.'
    },
    {
      title: 'Bulk Product Management',
      description: 'Optimize prices for your entire product catalog at once, considering relationships between products and inventory levels.'
    },
    {
      title: 'Customer Insights',
      description: 'Understand customer behavior and conversion patterns to refine your pricing strategy and boost sales.'
    },
    {
      title: 'E-Commerce Integrations',
      description: 'Connect directly with your favorite platforms like Shopify, WooCommerce, and Amazon Seller Central for seamless pricing updates.'
    },
    {
      title: 'Mobile App',
      description: 'Access pricing recommendations and market insights on the go with our mobile app for iOS and Android.'
    },
    {
      title: 'Price Testing Tools',
      description: 'Test different pricing strategies and measure their impact on your sales and profitability with built-in A/B testing.'
    }
  ];

  return (
    <div className="scope-list">
      {features.map((feature, idx) => (
        <div key={idx} className="scope-item">
          <h4>{feature.title}</h4>
          <p>{feature.description}</p>
        </div>
      ))}
    </div>
  );
};

export default App;


