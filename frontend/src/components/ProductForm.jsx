import React from 'react';

const categories = ['Electronics', 'Clothing', 'Home', 'Beauty', 'Sports'];
const months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];

export default function ProductForm({ form, onChange, onSubmit, loading, fieldErrors = {} }) {
  const errorFor = (name) => fieldErrors[name];
  const field = (name, label, props = {}, helper = null) => (
    <div className="form-field" key={name}>
      <label htmlFor={name}>{label}</label>
      <input id={name} name={name} value={form[name] ?? ''} onChange={onChange} aria-invalid={Boolean(errorFor(name))} aria-describedby={errorFor(name) ? `${name}-error` : undefined} {...props} />
      {helper && <small>{helper}</small>}
      {errorFor(name) && <small className="field-error" id={`${name}-error`} role="alert">{errorFor(name)}</small>}
    </div>
  );
  return (
    <form className="price-form" onSubmit={onSubmit} noValidate>
      <div className="form-section form-wide">
        <div className="form-section-heading"><span>01</span><div><h3>Product & costs</h3><p>Start with the item and what it costs you.</p></div></div>
        <div className="field-grid">
          {field('product_name', 'Product name', { placeholder: 'e.g. Wireless earbuds', maxLength: 200 }, 'Optional')}
          <div className="form-field"><label htmlFor="category">Category</label><select id="category" name="category" value={form.category} onChange={onChange}>{categories.map((category) => <option key={category}>{category}</option>)}</select></div>
          {field('cost_price', 'Unit cost (₹)', { type: 'number', min: '0.01', step: '0.01', required: true }, 'Required')}
          {field('fixed_costs', 'Other costs per unit (₹)', { type: 'number', min: '0', step: '0.01' }, 'Optional · defaults to ₹0')}
        </div>
      </div>
      <div className="form-section form-wide">
        <div className="form-section-heading"><span>02</span><div><h3>Market inputs</h3><p>Use simulated prices or add prices you collected.</p></div></div>
        <div className="field-grid">
          {field('num_competitors', 'Simulated competitors', { type: 'number', min: '1', max: '5000', step: '1' }, 'Used when manual prices are blank')}
          <div className="form-field"><label htmlFor="competitor_prices">Competitor prices <small>(optional)</small></label><textarea id="competitor_prices" name="competitor_prices" rows="2" placeholder="1299, 1549, 1799" value={form.competitor_prices} onChange={onChange} aria-invalid={Boolean(errorFor('competitor_prices'))} aria-describedby={errorFor('competitor_prices') ? 'competitor_prices-error' : 'competitor-prices-help'} /><small id="competitor-prices-help">Separate prices with commas or spaces.</small>{errorFor('competitor_prices') && <small className="field-error" id="competitor_prices-error" role="alert">{errorFor('competitor_prices')}</small>}</div>
        </div>
      </div>
      <details className="form-section form-wide advanced-settings">
        <summary><span className="advanced-label">More details</span><span className="advanced-hint">Quality, stock, timing, and markup range</span></summary>
        <div className="field-grid advanced-grid">
          <div className="form-field"><label htmlFor="quality">Quality tier</label><select id="quality" name="quality" value={form.quality} onChange={onChange}><option value="budget">Budget</option><option value="standard">Standard</option><option value="premium">Premium</option></select><small>Metadata only; it does not change the price.</small></div>
          <div className="form-field"><label htmlFor="month">Launch month</label><select id="month" name="month" value={form.month} onChange={onChange}>{months.map((month, i) => <option value={i + 1} key={month}>{month}</option>)}</select></div>
          {field('rating', 'Expected rating (1–5)', { type: 'number', min: '1', max: '5', step: '0.1' })}
          {field('ad_spend', 'Ad spend (₹)', { type: 'number', min: '0', step: '100' })}
          {field('available_units', 'Available stock', { type: 'number', min: '0', step: '1', placeholder: 'No limit' })}
          <div className="form-field"><label>Allowed markup range</label><div className="range-fields"><input aria-label="Minimum markup percent" name="min_markup_pct" type="number" min="0" step="0.1" value={form.min_markup_pct} onChange={onChange} /><span>to</span><input aria-label="Maximum markup percent" name="max_markup_pct" type="number" min="0" step="0.1" value={form.max_markup_pct} onChange={onChange} /><span>%</span></div>{(errorFor('min_markup_pct') || errorFor('max_markup_pct')) && <small className="field-error" role="alert">{errorFor('min_markup_pct') || errorFor('max_markup_pct')}</small>}</div>
        </div>
      </details>
      {fieldErrors.form && <div className="form-error form-wide" role="alert">{fieldErrors.form}</div>}
      <button className="btn btn-primary form-wide" type="submit" disabled={loading}>{loading ? 'Calculating…' : 'Get my price recommendation'}</button>
    </form>
  );
}
