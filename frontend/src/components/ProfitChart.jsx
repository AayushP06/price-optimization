import React, { useEffect, useRef, useState } from 'react';
import Chart from 'chart.js/auto';

const ProfitChart = ({ data, optimalPrice }) => {
  const sectionRef = useRef(null);
  const chartRef = useRef(null);
  const chartInstance = useRef(null);
  const [isVisible, setIsVisible] = useState(false);
  const priceRows = data?.all_prices ?? [];
  const hasData = priceRows.length > 0;

  useEffect(() => {
    if (!hasData || !chartRef.current) return;

    const ctx = chartRef.current.getContext('2d');
    if (!ctx) return;

    // Destroy existing chart if it exists
    if (chartInstance.current) {
      chartInstance.current.destroy();
    }

    // Use provided data or default sample data
    const pricePoints = data.all_prices.map(p => p.price);
    const profitMargins = data.all_prices.map(p => p.margin);
    const expectedProfits = data?.all_prices?.map(p => p.expected_profit) || [];

    // Find optimal index
    const hasOptimalPrice =
      optimalPrice !== null &&
      optimalPrice !== undefined &&
      Number.isFinite(Number(optimalPrice));
    const targetPrice = Number(optimalPrice);
    const optimalIndex = hasOptimalPrice
      ? pricePoints.reduce((bestIndex, price, index) => (
          Math.abs(Number(price) - targetPrice) <
          Math.abs(Number(pricePoints[bestIndex]) - targetPrice)
            ? index
            : bestIndex
        ), 0)
      : profitMargins.indexOf(Math.max(...profitMargins));

    chartInstance.current = new Chart(ctx, {
      type: 'line',
      data: {
        labels: pricePoints.map(p => `₹${p.toLocaleString('en-IN')}`),
        datasets: [
          {
            label: 'Profit Margin (%)',
            data: profitMargins,
            borderColor: '#d5ff55',
            backgroundColor: 'rgba(213, 255, 85, 0.12)',
            borderWidth: 3,
            pointRadius: pricePoints.map((_, i) => i === optimalIndex ? 10 : 6),
            pointHoverRadius: 8,
            pointBackgroundColor: pricePoints.map((_, i) => i === optimalIndex ? '#fa8273' : '#d5ff55'),
            pointBorderColor: '#10151d',
            pointBorderWidth: 2,
            tension: 0.4,
            fill: true,
            yAxisID: 'y'
          },
          ...(expectedProfits.length > 0 ? [{
            label: 'Expected Profit (₹)',
            data: expectedProfits,
            borderColor: '#fa8273',
            backgroundColor: 'rgba(250, 130, 115, 0.12)',
            borderWidth: 2,
            pointRadius: 4,
            tension: 0.4,
            fill: false,
            yAxisID: 'y1'
          }] : [])
        ]
      },
      options: {
        animation: {
          duration: window.matchMedia?.('(prefers-reduced-motion: reduce)')?.matches ? 0 : 850,
          easing: 'easeOutQuart',
        },
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
          mode: 'index',
          intersect: false,
        },
        plugins: {
          legend: {
            display: true,
            labels: {
              color: '#eef0e9',
              font: { size: 11 }
            }
          },
          tooltip: {
            backgroundColor: 'rgba(16, 21, 29, 0.96)',
            titleColor: '#d5ff55',
            bodyColor: '#eef0e9',
            borderColor: '#d5ff55',
            borderWidth: 1,
            padding: 12,
            callbacks: {
              label: function(context) {
                if (context.datasetIndex === 0) {
                  return 'Margin: ' + context.parsed.y + '%';
                } else {
                  return 'Profit: ₹' + context.parsed.y.toLocaleString('en-IN');
                }
              },
              title: function(context) {
                return 'Price: ' + context[0].label;
              }
            }
          }
        },
        scales: {
          x: {
            title: {
              display: true,
              text: 'Price Point (₹)',
              color: '#d5ff55',
              font: { size: 11, weight: 'bold' }
            },
            ticks: { color: '#eef0e9' },
            grid: { color: 'rgba(48, 55, 65, 0.55)' }
          },
          y: {
            type: 'linear',
            display: true,
            position: 'left',
            title: {
              display: true,
              text: 'Profit Margin (%)',
              color: '#d5ff55',
              font: { size: 11, weight: 'bold' }
            },
            ticks: {
              color: '#eef0e9',
              callback: function(value) {
                return value + '%';
              }
            },
            grid: { color: 'rgba(48, 55, 65, 0.55)' }
          },
          ...(expectedProfits.length > 0 ? {
            y1: {
              type: 'linear',
              display: true,
              position: 'right',
              title: {
                display: true,
                text: 'Expected Profit (₹)',
                color: '#fa8273',
                font: { size: 11, weight: 'bold' }
              },
              ticks: {
                color: '#eef0e9',
                callback: function(value) {
                  return '₹' + value.toLocaleString('en-IN');
                }
              },
              grid: {
                drawOnChartArea: false,
              },
            }
          } : {})
        }
      }
    });

    return () => {
      if (chartInstance.current) {
        chartInstance.current.destroy();
        chartInstance.current = null;
      }
    };
  }, [data, hasData, optimalPrice]);

  useEffect(() => {
    const section = sectionRef.current;
    if (!section) return;

    if (typeof IntersectionObserver === 'undefined') {
      setIsVisible(true);
      return;
    }

    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((entry) => entry.isIntersecting)) {
          setIsVisible(true);
          observer.disconnect();
        }
      },
      { threshold: 0.12 }
    );

    observer.observe(section);
    return () => observer.disconnect();
  }, []);

  return (
    <section
      id="insights"
      ref={sectionRef}
      className={`insights-section${isVisible ? ' section-visible' : ''}`}
      aria-labelledby="insights-title"
    >
      <div className="insights-intro">
        <div>
          <span className="eyebrow">02 / PROFIT SIGNAL</span>
          <h2 id="insights-title">Price is the lever.<br /><em>Profit is the signal.</em></h2>
        </div>
        <p>Model the trade-off between a competitive price and the profit you keep, using your own costs and market inputs.</p>
      </div>

      {hasData ? (
        <div className={`chart-card${isVisible ? ' chart-visible' : ''}`}>
          <div className="chart-card-heading">
            <div>
              <span className="chart-kicker">PRICE / PROFIT CURVE</span>
              <h3>Expected profit by price</h3>
            </div>
            <span className="chart-point-count">{priceRows.length} modeled price points</span>
          </div>
          <div className="chart-container">
            <canvas
              ref={chartRef}
              id="profitChart"
              role="img"
              aria-label="Price and expected profit analysis chart"
              aria-describedby="profit-chart-description"
            />
          </div>
          <p className="chart-description" id="profit-chart-description">
            The highlighted point marks the recommended price. Hover over the curve to compare projected profit across price points.
          </p>
        </div>
      ) : (
        <div className={`insights-empty${isVisible ? ' is-visible' : ''}`}>
          <span className="insights-empty-mark" aria-hidden="true">01</span>
          <div>
            <span className="eyebrow">AWAITING YOUR FIRST RUN</span>
            <h3>Turn costs and competition into a clear signal.</h3>
            <p>Run a recommendation to see how every candidate price could affect your expected profit.</p>
          </div>
          <a href="#optimize">Go to price inputs <span aria-hidden="true">↗</span></a>
        </div>
      )}
    </section>
  );
};

export default ProfitChart;


