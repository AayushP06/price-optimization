import React from 'react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import PriceOptimizer from './PriceOptimizer';

describe('PriceOptimizer', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => ({
      success: true, product_name: 'Test item', quality: 'standard', method: 'heuristic-fallback',
      simulated_competitors: true, warnings: [], optimal_price: { price: 125, margin: 16, margin_pct: 16, expected_profit: 100, expected_units_sold: 10 },
      competitor_stats: { count: 2, min: 110, max: 130, median: 120, mean: 120 },
      all_prices: [{ price: 105, expected_profit: 50 }, { price: 125, expected_profit: 100 }],
    }) }));
  });
  afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

  it('submits the form and renders the recommendation', async () => {
    render(<PriceOptimizer />);
    fireEvent.change(screen.getByLabelText('Product name'), { target: { value: 'Test item' } });
    fireEvent.click(screen.getByRole('button', { name: 'Generate recommendation' }));
    await waitFor(() => expect(screen.getByText('₹125')).toBeInTheDocument());
    expect(screen.getByText('Test item · standard')).toBeInTheDocument();
    expect(screen.getByText('Simulated competitor data')).toBeInTheDocument();
    expect(fetch).toHaveBeenCalledWith('/api/optimize', expect.objectContaining({ method: 'POST' }));
  });
});
