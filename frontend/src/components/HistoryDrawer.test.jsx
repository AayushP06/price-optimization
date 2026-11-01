import React from 'react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import HistoryDrawer from './HistoryDrawer';

vi.mock('./ProfitCurve', () => ({ default: () => <div>Compared profit curves</div> }));

const row = (id, product, price) => ({
  id, product_name: product, category: 'Home', quality: 'standard', method: 'heuristic-fallback',
  timestamp: '2026-10-31T10:00:00Z', cost_price: 100, fixed_costs: 5, available_units: null,
  competitor_stats: { count: 4, min: 105, max: 160, median: 125, mean: 130, simulated: false },
  competitor_prices: [110, 120, 140, 150], simulated_competitors: false,
  recommended_price: price, expected_profit: 100, expected_units_sold: 10,
  min_markup_pct: 5, max_markup_pct: 80, month: 6, rating: 4.2, ad_spend: 1000,
  all_prices: [{ price: 105, expected_profit: 50 }, { price, expected_profit: 100 }],
});

describe('HistoryDrawer', () => {
  afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

  it('loads history, compares two runs, and restores a selected run', async () => {
    const first = row(1, 'Wireless Earbuds', 125);
    const second = row(2, 'Yoga Mat', 135);
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => ({ data: [first, second] }) }));
    const onLoad = vi.fn();
    const onClose = vi.fn();
    render(<HistoryDrawer onLoad={onLoad} onClose={onClose} />);
    await screen.findByText('Wireless Earbuds');
    fireEvent.click(screen.getByRole('checkbox', { name: /Compare Wireless Earbuds/ }));
    fireEvent.click(screen.getByRole('checkbox', { name: /Compare Yoga Mat/ }));
    expect(await screen.findByText('Compared profit curves')).toBeInTheDocument();
    expect(screen.getByText('Compare runs')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Yoga Mat' }));
    await waitFor(() => expect(onLoad).toHaveBeenCalled());
    expect(onLoad.mock.calls[0][0]).toMatchObject({ product_name: 'Yoga Mat', competitor_prices: '110, 120, 140, 150' });
    expect(onLoad.mock.calls[0][1].optimal_price.price).toBe(135);
    expect(onClose).toHaveBeenCalled();
  });
});
