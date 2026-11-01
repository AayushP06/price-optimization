import React from 'react';
import { ArrowUpRight } from 'lucide-react';
import { sampleProducts } from '../data/presets';

export default function SampleProducts({ onSelect, loading, presets = [], onSave, onDelete, storageError, notice }) {
  const [name, setName] = React.useState('');
  const [formError, setFormError] = React.useState('');
  function handleSave() {
    const saved = onSave?.(name);
    if (saved) { setName(''); setFormError(''); }
    else if (!name.trim()) setFormError('Enter a preset name.');
  }
  return <section className="sample-products" aria-labelledby="sample-heading">
    <div className="subsection-title"><div><p className="eyebrow">Start with a demo</p><h3 id="sample-heading">Try a sample product</h3></div><span>Fills the form and runs it</span></div>
    <div className="sample-chips">{sampleProducts.map((sample) => <button key={sample.id} type="button" onClick={() => onSelect(sample)} disabled={loading}>{sample.label}<ArrowUpRight size={14} aria-hidden="true" /></button>)}</div>
    <div className="saved-presets"><div className="saved-presets-heading"><span>Saved on this device</span><span>Presets</span></div>
      {presets.length > 0 && <div className="sample-chips saved-preset-list">{presets.map((preset) => <span className="saved-preset" key={preset.id}><button type="button" onClick={() => onSelect(preset.form)} disabled={loading}>{preset.name}</button><button type="button" className="delete-preset" onClick={() => onDelete(preset.id)} aria-label={`Delete ${preset.name}`} title={`Delete ${preset.name}`}>×</button></span>)}</div>}
      <div className="save-preset-row"><label className="sr-only" htmlFor="preset-name">Name this preset</label><input id="preset-name" value={name} onChange={(event) => { setName(event.target.value); setFormError(''); }} placeholder="Name current form" maxLength={60} /><button type="button" onClick={handleSave}>Save form</button></div>
      {formError && <small className="field-error" role="alert">{formError}</small>}{storageError && <small className="storage-message" role="status">{storageError}</small>}{notice && <small className="storage-message" role="status">{notice}</small>}
    </div>
  </section>;
}
