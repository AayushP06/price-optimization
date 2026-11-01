import React from 'react';

const categories = ['Electronics', 'Clothing', 'Home', 'Beauty', 'Sports'];
const months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];

export default function ProductForm({ form, onChange, onSubmit, loading, fieldErrors = {} }) {
  const errorFor = (name) => fieldErrors[name]?.join(', ');
  const field = (name, label, props = {}) => (
    <div className="form-field" key={name}>
      <label htmlFor={name}>{label}</label>
      <input id={name} name={name} value={form[name]} onChange={onChange} aria-invalid={Boolean(errorFor(name))} {...props} />
      {errorFor(name) && <small className="field-error" role="alert">{errorFor(name)}</small>}
    </div>
  );
  return (
    <form className="optimizer-form" onSubmit={onSubmit}>
      {field('product_name', 'Product name', { placeholder: 'Wireless earbuds' })}
      <div className="form-field"><label htmlFor="quality">Quality tier</label><select id="quality" name="quality" value={form.quality} onChange={onChange}><option value="budget">Budget</option><option value="standard">Standard</option><option value="premium">Premium</option></select></div>
      <div className="form-field"><label htmlFor="category">Category</label><select id="category" name="category" value={form.category} onChange={onChange}>{categories.map((category) => <option key={category}>{category}</option>)}</select></div>
      <div className="form-field"><label htmlFor="month">Launch month</label><select id="month" name="month" value={form.month} onChange={onChange}>{months.map((month, i) => <option value={i + 1} key={month}>{month}</option>)}</select></div>
      {field('cost_price', 'Cost price (₹)', { type: 'number', min: '0.01', step: '0.01', required: true })}
      {field('fixed_costs', 'Fixed costs per unit (₹)', { type: 'number', min: '0', step: '0.01' })}
      {field('min_markup_pct', 'Minimum markup (%)', { type: 'number', step: '0.1' })}
      {field('max_markup_pct', 'Maximum markup (%)', { type: 'number', step: '0.1' })}
      {field('rating', 'Expected rating (1–5)', { type: 'number', min: '1', max: '5', step: '0.1' })}
      {field('ad_spend', 'Ad spend (₹)', { type: 'number', min: '0', step: '1' })}
      {field('available_units', 'Available units', { type: 'number', min: '0', step: '1', placeholder: 'Unlimited' })}
      {field('num_competitors', 'Simulated competitor count', { type: 'number', min: '1', max: '5000', step: '1' })}
      <div className="form-field form-field-wide">
        <label htmlFor="competitor_prices">Competitor prices (optional)</label>
        <textarea id="competitor_prices" name="competitor_prices" value={form.competitor_prices} onChange={onChange} rows="3" placeholder="1299, 1549, 1799" />
        {errorFor('competitor_prices') && <small className="field-error" role="alert">{errorFor('competitor_prices')}</small>}
        <small>Enter prices separated by commas or spaces. Leave blank to simulate market prices.</small>
      </div>
      <button className="btn btn-primary form-field-wide" type="submit" disabled={loading}>{loading ? 'Calculating…' : 'Generate recommendation'}</button>
    </form>
  );
}
