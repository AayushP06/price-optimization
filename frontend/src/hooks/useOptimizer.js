import { useRef, useState } from 'react';
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
  const [fieldErrors, setFieldErrors] = useState({});
  const revision = useRef(0);

  function updateField(event) {
    revision.current += 1;
    const { name, value } = event.target;
    setForm((current) => ({ ...current, [name]: value }));
    setStale(Boolean(result));
    setError(null);
    setFieldErrors({});
  }

  async function run(candidateForm = form) {
    const submittedRevision = revision.current;
    setLoading(true);
    setStale(Boolean(result));
    setError(null);
    setFieldErrors({});
    const prices = candidateForm.competitor_prices.split(/[\s,]+/).filter(Boolean).map(Number);
    const payload = {
      ...candidateForm,
      cost_price: Number(candidateForm.cost_price), fixed_costs: Number(candidateForm.fixed_costs || 0),
      available_units: candidateForm.available_units === '' ? null : Number(candidateForm.available_units),
      num_competitors: prices.length ? undefined : Number(candidateForm.num_competitors),
      competitor_prices: prices, month: Number(candidateForm.month), rating: Number(candidateForm.rating),
      ad_spend: Number(candidateForm.ad_spend), min_markup_pct: Number(candidateForm.min_markup_pct),
      max_markup_pct: Number(candidateForm.max_markup_pct),
    };
    try {
      setResult(await optimizePrice(payload));
      setStale(revision.current !== submittedRevision);
    } catch (submissionError) {
      setError(submissionError);
      setFieldErrors(Object.fromEntries(Object.entries(submissionError.fields || {}).map(([key, messages]) => {
        const field = key.startsWith('competitor_prices') ? 'competitor_prices' : ['min_markup_pct', 'max_markup_pct', 'cost_price', 'fixed_costs', 'category', 'month', 'rating', 'ad_spend', 'available_units', 'num_competitors', 'quality', 'product_name'].includes(key) ? key : 'form';
        return [field, Array.isArray(messages) ? messages.join(', ') : String(messages)];
      })));
    } finally {
      setLoading(false);
    }
  }

  function submit(event) {
    event.preventDefault();
    return run(form);
  }

  return { form, setForm, result, setResult, stale, setStale, loading, error, setError, fieldErrors, updateField, submit, run };
}
