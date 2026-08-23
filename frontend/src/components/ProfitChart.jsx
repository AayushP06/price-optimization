import React, { useEffect, useRef, useState } from 'react';
import Chart from 'chart.js/auto';

const ProfitChart = ({ data, optimalPrice }) => {
  const chartRef = useRef(null);
  const chartInstance = useRef(null);
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    if (!chartRef.current) return;

    const ctx = chartRef.current.getContext('2d');

    // Destroy existing chart if it exists
    if (chartInstance.current) {
      chartInstance.current.destroy();
    }

    // Use provided data or default sample data
    const pricePoints = data?.all_prices?.map(p => p.price) || 
      [1200, 1300, 1400, 1450, 1500, 1550, 1599, 1650, 1700, 1750, 1800, 1850, 1900];
    const profitMargins = data?.all_prices?.map(p => p.margin) || 
      [10, 14, 18, 20, 23, 25, 26, 25.5, 24, 22, 20, 17, 14];
    const expectedProfits = data?.all_prices?.map(p => p.expected_profit) || [];

    // Find optimal index
    const optimalIndex = optimalPrice 
      ? pricePoints.findIndex(p => Math.abs(p - optimalPrice) < 1)
      : profitMargins.indexOf(Math.max(...profitMargins));

    chartInstance.current = new Chart(ctx, {
      type: 'line',
      data: {
        labels: pricePoints.map(p => `₹${p.toLocaleString('en-IN')}`),
        datasets: [
          {
            label: 'Profit Margin (%)',
            data: profitMargins,
            borderColor: '#f39c12',
            backgroundColor: 'rgba(243, 156, 18, 0.1)',
            borderWidth: 3,
            pointRadius: pricePoints.map((_, i) => i === optimalIndex ? 10 : 6),
            pointHoverRadius: 8,
            pointBackgroundColor: pricePoints.map((_, i) => i === optimalIndex ? '#e67e22' : '#f39c12'),
            pointBorderColor: '#fff',
            pointBorderWidth: 2,
            tension: 0.4,
            fill: true,
            yAxisID: 'y'
          },
          ...(expectedProfits.length > 0 ? [{
            label: 'Expected Profit (₹)',
            data: expectedProfits,
            borderColor: '#3498db',
            backgroundColor: 'rgba(52, 152, 219, 0.1)',
            borderWidth: 2,
            pointRadius: 4,
            tension: 0.4,
            fill: false,
            yAxisID: 'y1'
          }] : [])
        ]
      },
      options: {
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
              color: '#e8e8e8',
              font: { size: 14 }
            }
          },
          tooltip: {
            backgroundColor: 'rgba(20, 27, 58, 0.9)',
            titleColor: '#f39c12',
            bodyColor: '#e8e8e8',
            borderColor: '#f39c12',
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
              color: '#f39c12',
              font: { size: 14, weight: 'bold' }
            },
            ticks: { color: '#e8e8e8' },
            grid: { color: 'rgba(42, 53, 85, 0.5)' }
          },
          y: {
            type: 'linear',
            display: true,
            position: 'left',
            title: {
              display: true,
              text: 'Profit Margin (%)',
              color: '#f39c12',
              font: { size: 14, weight: 'bold' }
            },
            ticks: {
              color: '#e8e8e8',
              callback: function(value) {
                return value + '%';
              }
            },
            grid: { color: 'rgba(42, 53, 85, 0.5)' }
          },
          ...(expectedProfits.length > 0 ? {
            y1: {
              type: 'linear',
              display: true,
              position: 'right',
              title: {
                display: true,
                text: 'Expected Profit (₹)',
                color: '#3498db',
                font: { size: 14, weight: 'bold' }
              },
              ticks: {
                color: '#e8e8e8',
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
      }
    };
  }, [data, optimalPrice]);

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            setIsVisible(true);
          }
        });
      },
      { threshold: 0.1 }
    );

    if (chartRef.current) {
      observer.observe(chartRef.current);
    }

    return () => {
      if (chartRef.current) {
        observer.unobserve(chartRef.current);
      }
    };
  }, []);

  return (
    <section id="insights" className={isVisible ? 'section-visible' : ''}>
      <h2>Price vs. Profit Analysis</h2>
      <p className="section-subtitle">
        See how different price points affect your profit margin
      </p>

      <div className={`chart-container ${isVisible ? 'chart-visible' : ''}`}>
        <div style={{position: 'absolute', top: 10, right: 10, background: 'rgba(0,0,0,0.8)', color: 'white', padding: '5px', zIndex: 10, fontSize: '12px'}}>
          [Debug] Points received: {data?.all_prices?.length || 'None (Fallback)'}
        </div>
        <canvas ref={chartRef} id="profitChart"></canvas>
        {data && (
          <div className="chart-overlay">
            <div className="chart-tip">
              💡 Hover over data points to see detailed information
            </div>
          </div>
        )}
      </div>
      
      <p className="chart-description">
        <strong>Understanding the Chart:</strong> This graph shows how different price points affect your profit margins. 
        The optimal price (highlighted in orange) is the sweet spot where you maximize profit while staying competitive. 
        Prices that are too low reduce your margins, while prices that are too high may reduce sales volume.
      </p>
    </section>
  );
};

export default ProfitChart;


