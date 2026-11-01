import React from 'react';
import ProductForm from './ProductForm';
import ResultCard from './ResultCard';
import { useOptimizer } from '../hooks/useOptimizer';

export default function PriceOptimizer() {
  const optimizer = useOptimizer();
  return <main id="optimize" className="optimizer-layout">
    <section className="form-panel">
      <header><p className="eyebrow">Price optimization</p><h1>Find your best price</h1><p>Set your costs and market assumptions to estimate a profitable selling price.</p></header>
      <ProductForm form={optimizer.form} onChange={optimizer.updateField} onSubmit={optimizer.submit} loading={optimizer.loading} fieldErrors={optimizer.error?.fields} />
    </section>
    <ResultCard result={optimizer.result} loading={optimizer.loading} error={optimizer.error} stale={optimizer.stale} />
  </main>;
}
