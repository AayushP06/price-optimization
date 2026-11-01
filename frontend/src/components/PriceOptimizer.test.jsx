import React from 'react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import PriceOptimizer from './PriceOptimizer';
import { resultFlags } from './ResultCard';

vi.mock('./ProfitCurve', () => ({ default: () => <div>Profit curve preview</div> }));

const sampleResult = {
  success: true, id: 1, product_name: 'Test item', quality: 'standard', method: 'heuristic-fallback',
  simulated_competitors: true, warnings: [], cost_price: 100, fixed_costs: 5, min_markup_pct: 5, max_markup_pct: 80,
  category: 'Electronics', month: 6, rating: 4.2, ad_spend: 1000, available_units: null,
  optimal_price: { price: 125, profit_per_unit: 20, margin: 16, margin_pct: 16, markup_pct: 25, expected_profit: 100, expected_units_sold: 10 },
  competitor_stats: { count: 5, min: 110, max: 150, median: 120, mean: 126 },
  competitor_prices: [110, 115, 120, 130, 150],
  all_prices: [{ price: 105, expected_profit: 50 }, { price: 125, expected_profit: 100 }, { price: 180, expected_profit: 0 }],
};

describe('PriceOptimizer', () => {
  beforeEach(() => vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => sampleResult })));
  afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

  it('submits the form and renders recommendation, metrics, and condition flags', async () => {
    render(<PriceOptimizer />);
    fireEvent.change(screen.getByLabelText('Product name'), { target: { value: 'Test item' } });
    fireEvent.click(screen.getByRole('button', { name: 'Get my price recommendation' }));
    await waitFor(() => expect(screen.getByText('₹125')).toBeInTheDocument());
    expect(screen.getByText('Test item')).toBeInTheDocument();
    expect(screen.getByText('Expected profit')).toBeInTheDocument();
    expect(screen.getByText('Markup')).toBeInTheDocument();
    expect(screen.getByText('Net margin')).toBeInTheDocument();
    expect(screen.getByText('Simulated competitor data')).toBeInTheDocument();
    expect(screen.getByText('Heuristic fallback used')).toBeInTheDocument();
    expect(fetch).toHaveBeenCalledWith('/api/optimize', expect.objectContaining({ method: 'POST' }));
  });

  it('marks the previous recommendation stale after input changes', async () => {
    render(<PriceOptimizer />);
    fireEvent.click(screen.getByRole('button', { name: 'Get my price recommendation' }));
    await screen.findByText('₹125');
    fireEvent.change(screen.getByLabelText('Unit cost (₹)'), { target: { value: '1100' } });
    expect(screen.getByText('Inputs changed · result is stale')).toBeInTheDocument();
    expect(screen.getByText('₹125')).toBeInTheDocument();
  });

  it('fills the selected sample and runs the optimizer', async () => {
    render(<PriceOptimizer />);
    fireEvent.click(screen.getByRole('button', { name: /Wireless Earbuds/ }));
    await waitFor(() => expect(screen.getByText('₹125')).toBeInTheDocument());
    expect(screen.getByLabelText('Product name')).toHaveValue('Wireless Earbuds');
    expect(fetch).toHaveBeenCalledTimes(1);
    const submitted = JSON.parse(fetch.mock.calls[0][1].body);
    expect(submitted.category).toBe('Electronics');
    expect(submitted.cost_price).toBe(1000);
  });
});

describe('result flags', () => {
  it('reports markup bounds, median distance, stock caps, and the central competitor range', () => {
    const flags = resultFlags({ ...sampleResult, competitor_prices: [90, 100, 110, 120, 130, 150, 160], optimal_price: { ...sampleResult.optimal_price, price: 105, expected_units_sold: 3 }, available_units: 3 }, {});
    expect(flags.map(({ text }) => text)).toEqual(expect.arrayContaining([
      'At markup range limit', 'Simulated competitor data', 'Heuristic fallback used',
      '12.5% below competitor median', 'Units capped by available stock', 'Within middle 50% of competitor prices',
    ]));
  });

  it('shows a median comparison without claiming a central range match', () => {
    const flags = resultFlags({ ...sampleResult, optimal_price: { ...sampleResult.optimal_price, price: 145 }, competitor_prices: [110, 115, 120] }, {});
    expect(flags.some(({ text }) => text.includes('above competitor median'))).toBe(true);
    expect(flags.some(({ text }) => text.includes('middle 50%'))).toBe(false);
  });
});
