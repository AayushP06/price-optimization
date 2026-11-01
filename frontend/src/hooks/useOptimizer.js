import { useState } from 'react';
import { optimizePrice } from '../api/client';

const initialForm = {
  product_name: '', quality: 'standard', cost_price: '1000', fixed_costs: '50',
  available_units: '', num_competitors: '100', competitor_prices: '',
  category: 'Electronics', month: '6', rating: '4.2', ad_spend: '1000',
  min_markup_pct: '5', max_markup_pct: '80',
};

export function useOptimizer() {
  const [form, setForm] = useState(initialForm);
  const [result, setResult] = useState(null);
  const [stale, setStale] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  function updateField(event) {
    const { name, value } = event.target;
    setForm((current) => ({ ...current, [name]: value }));
    setStale(Boolean(result));
    setError(null);
  }

  async function submit(event) {
    event.preventDefault();
    setLoading(true);
    setError(null);
    const prices = form.competitor_prices.split(/[\s,]+/).filter(Boolean).map(Number);
    const payload = {
      ...form,
      cost_price: Number(form.cost_price), fixed_costs: Number(form.fixed_costs || 0),
      available_units: form.available_units === '' ? null : Number(form.available_units),
      num_competitors: prices.length ? undefined : Number(form.num_competitors),
      competitor_prices: prices, month: Number(form.month), rating: Number(form.rating),
      ad_spend: Number(form.ad_spend), min_markup_pct: Number(form.min_markup_pct),
      max_markup_pct: Number(form.max_markup_pct),
    };
    try {
      setResult(await optimizePrice(payload));
      setStale(false);
    } catch (submissionError) {
      setError(submissionError);
    } finally {
      setLoading(false);
    }
  }

  return { form, result, stale, loading, error, updateField, submit };
}
