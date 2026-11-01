import { describe, expect, it } from 'vitest';
import { makeResultCsv } from './reportCsv';

describe('result CSV report', () => {
  it('includes recommendation, assumptions, and safely quoted values', () => {
    const csv = makeResultCsv({ product_name: 'Widget, blue', category: 'Home', optimal_price: { price: 150, expected_profit: 25, profit_per_unit: 5, markup_pct: 50, margin_pct: 16.6, expected_units_sold: 5 }, warnings: ['Needs, review'] }, { cost_price: 100, quality: 'standard' });
    expect(csv).toContain('"Product","Widget, blue"');
    expect(csv).toContain('"Recommended price (INR)","150"');
    expect(csv).toContain('"Quality tier","standard"');
    expect(csv).toContain('"Warnings","Needs, review"');
  });
});
