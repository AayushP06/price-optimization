import React from 'react';
import { useOptimizer } from '../hooks/useOptimizer';
import ProductForm from './ProductForm';
import ResultCard from './ResultCard';
import SampleProducts from './SampleProducts';

export default function PriceOptimizer() {
  const optimizer = useOptimizer();
  return <section id="optimize" className="optimizer-section">
    <header className="section-heading"><p className="eyebrow">Price recommendation</p><h2>Set your market inputs</h2><p className="section-subtitle">Use your costs and competitor data to compare price points.</p></header>
    <div className="optimizer-layout">
      <div className="optimizer-form-column">
        <SampleProducts loading={optimizer.loading} onSelect={(sample) => { optimizer.setForm(sample); optimizer.run(sample); }} />
        <ProductForm form={optimizer.form} onChange={optimizer.updateField} onSubmit={optimizer.submit} loading={optimizer.loading} fieldErrors={optimizer.fieldErrors} />
      </div>
      <ResultCard result={optimizer.result} form={optimizer.form} loading={optimizer.loading} error={optimizer.error} stale={optimizer.stale} />
    </div>
  </section>;
}
