import React, { useEffect, useRef } from 'react';
import Chart from 'chart.js/auto';

const ProfitChart = ({ data, optimalPrice }) => {
  const chartRef = useRef(null);
  const chartInstance = useRef(null);

  useEffect(() => {
    if (!chartRef.current) return;
    const ctx = chartRef.current.getContext('2d');

    if (chartInstance.current) {
      chartInstance.current.destroy();
    }

    // Default placeholder data if none provided
    const pricePoints = data?.all_prices?.map(p => p.price) || [1200, 1300, 1400, 1450, 1500, 1550, 1599, 1650, 1700];
    const profitMargins = data?.all_prices?.map(p => p.margin) || [10, 14, 18, 20, 23, 25, 26, 25.5, 24];
    const expectedProfits = data?.all_prices?.map(p => p.expected_profit) || [];

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
            borderColor: '#8b5cf6', // Violet
            backgroundColor: 'rgba(139, 92, 246, 0.1)',
            borderWidth: 3,
            pointRadius: pricePoints.map((_, i) => i === optimalIndex ? 8 : 4),
            pointBackgroundColor: pricePoints.map((_, i) => i === optimalIndex ? '#10b981' : '#8b5cf6'),
            pointBorderColor: '#fff',
            pointBorderWidth: 2,
            tension: 0.4,
            fill: true,
            yAxisID: 'y'
          },
          ...(expectedProfits.length > 0 ? [{
            label: 'Expected Profit (₹)',
            data: expectedProfits,
            borderColor: '#6366f1', // Indigo
            borderWidth: 2,
            pointRadius: 0, // Hide dots to keep it clean
            tension: 0.4,
            fill: false,
            yAxisID: 'y1'
          }] : [])
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: { labels: { color: '#94a3b8' } },
          tooltip: {
            backgroundColor: '#0f172a',
            titleColor: '#f8fafc',
            bodyColor: '#e2e8f0',
            borderColor: '#334155',
            borderWidth: 1,
            padding: 12
          }
        },
        scales: {
          x: {
            ticks: { color: '#64748b' },
            grid: { color: 'rgba(51, 65, 85, 0.3)' }
          },
          y: {
            type: 'linear',
            position: 'left',
            ticks: { color: '#64748b', callback: value => value + '%' },
            grid: { color: 'rgba(51, 65, 85, 0.3)' }
          },
          ...(expectedProfits.length > 0 ? {
            y1: {
              type: 'linear',
              position: 'right',
              ticks: { color: '#64748b' },
              grid: { drawOnChartArea: false }
            }
          } : {})
        }
      }
    });

    return () => chartInstance.current?.destroy();
  }, [data, optimalPrice]);

  return (
    <div style={{ position: 'relative', height: '300px', width: '100%' }}>
      <canvas ref={chartRef}></canvas>
    </div>
  );
};

export default ProfitChart;