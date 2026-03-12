import React, { useState, useEffect } from 'react';

const Hero = () => {
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    setIsVisible(true);
  }, []);

  useEffect(() => {
    const handleMouseMove = (e) => {
      setMousePosition({
        x: (e.clientX / window.innerWidth) * 100,
        y: (e.clientY / window.innerHeight) * 100
      });
    };

    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  const scrollToOptimize = () => {
    const element = document.getElementById('optimize');
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  };

  return (
    <section 
      id="home" 
      className="hero"
      style={{
        background: `radial-gradient(circle at ${mousePosition.x}% ${mousePosition.y}%, rgba(243, 156, 18, 0.1) 0%, transparent 50%)`
      }}
    >
      <div className={`hero-content ${isVisible ? 'fade-in' : ''}`}>
        <h1 className="hero-title">
          <span className="title-word">Find</span>{' '}
          <span className="title-word">Your</span>{' '}
          <span className="title-word">Perfect</span>{' '}
          <span className="title-word highlight">Price</span>{' '}
          <span className="title-word">in</span>{' '}
          <span className="title-word">Seconds</span>
        </h1>
        <p className="tagline">
          Maximize profits while staying competitive. Get instant pricing recommendations powered by smart analytics.
        </p>
        <button className="btn btn-hero" onClick={scrollToOptimize}>
          <span>Get Started Now</span>
          <span className="btn-arrow">→</span>
        </button>
        <div className="hero-stats">
          <div className="hero-stat">
            <div className="stat-number" data-target="95">0</div>
            <div className="stat-text">% Accuracy</div>
          </div>
          <div className="hero-stat">
            <div className="stat-number" data-target="18">0</div>
            <div className="stat-text">% Profit Boost</div>
          </div>
          <div className="hero-stat">
            <div className="stat-number" data-target="10000">0</div>
            <div className="stat-text">Products Analyzed</div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Hero;


