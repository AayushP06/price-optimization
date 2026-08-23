import React, { useState, useEffect, useRef } from 'react';

const useCountUp = (target, duration = 1800, start = false) => {
  const [value, setValue] = useState(0);
  useEffect(() => {
    if (!start || !target) return;
    let startTime = null;
    const step = (timestamp) => {
      if (!startTime) startTime = timestamp;
      const progress = Math.min((timestamp - startTime) / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      setValue(Math.floor(eased * target));
      if (progress < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }, [target, duration, start]);
  return value;
};

const Hero = () => {
  const [isVisible, setIsVisible] = useState(false);
  const [mousePos, setMousePos] = useState({ x: 50, y: 50 });
  const heroRef = useRef(null);

  const accuracy  = useCountUp(95,   1600, isVisible);
  const profit    = useCountUp(18,   1400, isVisible);
  const products  = useCountUp(10000, 2000, isVisible);

  useEffect(() => {
    const timer = setTimeout(() => setIsVisible(true), 200);
    return () => clearTimeout(timer);
  }, []);

  useEffect(() => {
    const handleMouseMove = (e) => {
      setMousePos({
        x: (e.clientX / window.innerWidth) * 100,
        y: (e.clientY / window.innerHeight) * 100,
      });
    };
    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  const scrollToOptimize = () => {
    document.getElementById('optimize')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };

  return (
    <section id="home" className="hero" ref={heroRef}
      style={{
        background: `radial-gradient(ellipse 60% 50% at ${mousePos.x}% ${mousePos.y}%, rgba(245,158,11,0.08) 0%, transparent 60%),
                     linear-gradient(180deg, #060918 0%, #0d1229 100%)`
      }}
    >
      <div className="hero-orb hero-orb-1" />
      <div className="hero-orb hero-orb-2" />

      <div className={`hero-content ${isVisible ? 'fade-in' : ''}`}>
        <div className="hero-badge">AI-Powered Price Optimization</div>

        <h1 className="hero-title">
          Find Your Perfect{' '}
          <span className="highlight">Price</span>
          {' '}in Seconds
        </h1>

        <p className="tagline">
          Maximize profits while staying competitive. Get instant pricing
          recommendations powered by smart market analytics and Q-Learning AI.
        </p>

        <div className="hero-actions">
          <button className="btn btn-hero" onClick={scrollToOptimize}>
            <span className="btn-icon">🚀</span>
            Get Started Now
            <span className="btn-arrow">→</span>
          </button>
          <a href="#results" className="btn-outline">
            See How It Works
          </a>
        </div>

        <div className="hero-stats">
          <div className="hero-stat">
            <span className="stat-number">{accuracy}%</span>
            <div className="stat-text">Accuracy Rate</div>
          </div>
          <div className="hero-stat">
            <span className="stat-number">+{profit}%</span>
            <div className="stat-text">Avg Profit Boost</div>
          </div>
          <div className="hero-stat">
            <span className="stat-number">{products >= 10000 ? '10K+' : products}</span>
            <div className="stat-text">Products Analyzed</div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Hero;
