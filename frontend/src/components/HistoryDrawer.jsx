import React, { useCallback, useEffect, useState } from 'react';
import { X } from 'lucide-react';
import { getHistory } from '../api/client';
import HistoryCompare from './HistoryCompare';

const rupees = (value) => `₹${Number(value ?? 0).toLocaleString('en-IN', { maximumFractionDigits: 2 })}`;

export default function HistoryDrawer({ onClose, onLoad }) {
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [reloadCount, setReloadCount] = useState(0);
  const [selected, setSelected] = useState([]);

  const load = useCallback(async () => {
    setLoading(true);
    setError('');
    try { setRows(await getHistory(100)); }
    catch (loadError) { setError(loadError.message || 'Could not load history.'); }
    finally { setLoading(false); }
  }, [reloadCount]);

  useEffect(() => { load(); }, [load]);
  useEffect(() => {
    const previous = document.activeElement;
    document.body.classList.add('drawer-open');
    document.querySelector('.history-drawer .icon-button')?.focus();
    return () => { document.body.classList.remove('drawer-open'); previous?.focus?.(); };
  }, []);

  const trapDialogKeys = (event) => {
    if (event.key === 'Escape') { onClose(); return; }
    if (event.key !== 'Tab') return;
    const focusable = [...event.currentTarget.querySelectorAll('button:not(:disabled), input:not(:disabled), a[href], [tabindex]:not([tabindex="-1"])')];
    if (!focusable.length) return;
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
    else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
  };

  const toggleCompare = (record) => setSelected((current) => current.some((item) => item.id === record.id)
    ? current.filter((item) => item.id !== record.id)
    : current.length < 2 ? [...current, record] : [current[1], record]);

  const loadRecord = (record) => {
    const price = Number(record.recommended_price || 0);
    const cost = Number(record.cost_price || 0);
    const fixed = Number(record.fixed_costs || 0);
    const profitPerUnit = Math.max(0, price - cost - fixed);
    const form = {
      product_name: record.product_name || '', quality: record.quality || 'standard',
      cost_price: String(record.cost_price ?? ''), fixed_costs: String(record.fixed_costs ?? 0),
      available_units: record.available_units == null ? '' : String(record.available_units),
      num_competitors: String(record.competitor_stats?.count || 100),
      competitor_prices: record.simulated_competitors ? '' : (record.competitor_prices || []).join(', '),
      category: record.category || 'Electronics', month: String(record.month || 6),
      rating: String(record.rating ?? 4.2), ad_spend: String(record.ad_spend ?? 1000),
      min_markup_pct: String(record.min_markup_pct ?? 5), max_markup_pct: String(record.max_markup_pct ?? 80),
    };
    const result = {
      success: true, id: record.id, product_name: record.product_name, quality: record.quality,
      method: record.method, cost_price: cost, fixed_costs: fixed, category: record.category,
      month: record.month, rating: record.rating, ad_spend: record.ad_spend, available_units: record.available_units,
      min_markup_pct: record.min_markup_pct, max_markup_pct: record.max_markup_pct,
      simulated_competitors: record.simulated_competitors, competitor_stats: record.competitor_stats,
      competitor_prices: record.competitor_prices || [], all_prices: record.all_prices || [],
      optimal_price: {
        price, profit_per_unit: profitPerUnit, expected_profit: Number(record.expected_profit || 0),
        expected_units_sold: Number(record.expected_units_sold || 0),
        markup_pct: cost ? (price - cost) / cost * 100 : 0,
        margin_pct: price ? profitPerUnit / price * 100 : 0,
      },
    };
    onLoad(form, result);
    onClose();
  };

  return <div className="drawer-backdrop" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}>
    <aside className="history-drawer" role="dialog" aria-modal="true" aria-labelledby="history-title" onKeyDown={trapDialogKeys}>
      <header><div><p className="eyebrow">Saved recommendations</p><h2 id="history-title">Optimization history</h2></div><button type="button" className="icon-button" onClick={onClose} aria-label="Close history"><X size={20} /></button></header>
      <p className="drawer-hint">Select up to two runs to compare. Choose a row to load its inputs and result.</p>
      {loading && <div className="history-loading" role="status">Loading optimization history…</div>}
      {error && <div className="history-error" role="alert"><p>{error}</p><button type="button" onClick={() => setReloadCount((count) => count + 1)}>Try again</button></div>}
      {!loading && !error && rows.length === 0 && <div className="history-empty"><strong>No saved runs yet</strong><p>Run an optimization and it will appear here.</p></div>}
      {!loading && !error && rows.length > 0 && <div className="history-table-wrap"><table className="history-table"><thead><tr><th scope="col">Compare</th><th scope="col">Product</th><th scope="col">Category</th><th scope="col">Price</th><th scope="col">Expected profit</th><th scope="col">Method</th><th scope="col">Timestamp</th></tr></thead><tbody>{rows.map((row) => <tr key={row.id}>
        <td><input type="checkbox" checked={selected.some((item) => item.id === row.id)} onChange={() => toggleCompare(row)} aria-label={`Compare ${row.product_name || 'product'} run from ${row.timestamp}`} /></td>
        <td><button type="button" className="history-load" onClick={() => loadRecord(row)}>{row.product_name || 'Product'}</button></td>
        <td>{row.category || '—'}</td><td>{rupees(row.recommended_price)}</td><td>{rupees(row.expected_profit)}</td><td>{row.method || '—'}</td><td>{row.timestamp ? new Date(row.timestamp).toLocaleString('en-IN') : '—'}</td>
      </tr>)}</tbody></table></div>}
      {!loading && selected.length === 2 && <HistoryCompare records={selected} />}
    </aside>
  </div>;
}
