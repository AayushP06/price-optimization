import React from 'react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import App from './App';

vi.mock('./components/PriceOptimizer', () => ({ default: () => <div>Optimizer</div> }));

describe('theme preference', () => {
  afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

  it('loads and persists the selected theme', async () => {
    const values = new Map([['priceoptimizer.theme', 'dark']]);
    Object.defineProperty(window, 'localStorage', { configurable: true, value: {
      getItem: (key) => values.get(key) ?? null,
      setItem: (key, value) => values.set(key, String(value)),
    } });
    render(<App />);
    await waitFor(() => expect(document.documentElement.dataset.theme).toBe('dark'));
    fireEvent.click(screen.getByRole('button', { name: 'Switch to light theme' }));
    await waitFor(() => expect(document.documentElement.dataset.theme).toBe('light'));
    expect(values.get('priceoptimizer.theme')).toBe('light');
  });
});
