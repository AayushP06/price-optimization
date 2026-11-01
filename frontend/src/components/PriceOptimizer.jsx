import React from 'react';
import { useOptimizer } from '../hooks/useOptimizer';
import ProductForm from './ProductForm';
import ResultCard from './ResultCard';
import SampleProducts from './SampleProducts';
import { useSavedPresets } from '../hooks/useSavedPresets';

export default function PriceOptimizer({ optimizer: sharedOptimizer } = {}) {
  const localOptimizer = useOptimizer();
  const optimizer = sharedOptimizer || localOptimizer;
  const savedPresets = useSavedPresets();
  return <section id="optimize" className="optimizer-section">
    <header className="section-heading"><p className="eyebrow">Price recommendation</p><h2>Set your market inputs</h2><p className="section-subtitle">Use your costs and competitor data to compare price points.</p></header>
    <div className="optimizer-layout">
      <div className="optimizer-form-column">
        <SampleProducts loading={optimizer.loading} presets={savedPresets.presets} storageError={savedPresets.error} notice={savedPresets.notice} onSave={(name) => savedPresets.save(name, optimizer.form)} onDelete={savedPresets.remove} onSelect={(sample) => { optimizer.setForm(sample); optimizer.run(sample); }} />
        <ProductForm form={optimizer.form} onChange={optimizer.updateField} onSubmit={optimizer.submit} loading={optimizer.loading} fieldErrors={optimizer.fieldErrors} />
      </div>
      <ResultCard result={optimizer.result} form={optimizer.form} loading={optimizer.loading} error={optimizer.error} stale={optimizer.stale} />
    </div>
  </section>;
}
