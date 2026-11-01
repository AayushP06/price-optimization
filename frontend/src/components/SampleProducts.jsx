import React from 'react';
import { ArrowUpRight } from 'lucide-react';
import { sampleProducts } from '../data/presets';

export default function SampleProducts({ onSelect, loading }) {
  return <section className="sample-products" aria-labelledby="sample-heading">
    <div className="subsection-title"><div><p className="eyebrow">Start with a demo</p><h3 id="sample-heading">Try a sample product</h3></div><span>Fills the form and runs it</span></div>
    <div className="sample-chips">{sampleProducts.map((sample) => <button key={sample.id} type="button" onClick={() => onSelect(sample)} disabled={loading}>{sample.label}<ArrowUpRight size={14} aria-hidden="true" /></button>)}</div>
  </section>;
}
