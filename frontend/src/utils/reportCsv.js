function csvCell(value) {
  let text = String(value ?? '');
  if (/^\s*[=+\-@]/.test(text)) text = `'${text}`;
  return `"${text.replace(/"/g, '""')}"`;
}

export function makeResultCsv(result, form = {}) {
  const optimum = result?.optimal_price || {};
  const fields = [
    ['Product', result?.product_name || form.product_name || 'Product'],
    ['Category', result?.category ?? form.category], ['Quality tier', result?.quality ?? form.quality],
    ['Recommended price (INR)', optimum.price], ['Expected profit (INR)', optimum.expected_profit],
    ['Profit per unit (INR)', optimum.profit_per_unit], ['Markup (%)', optimum.markup_pct],
    ['Net margin (%)', optimum.margin_pct ?? optimum.margin], ['Expected units', optimum.expected_units_sold],
    ['Method', result?.method], ['Competitors simulated', result?.simulated_competitors],
    ['Competitor count', result?.competitor_stats?.count], ['Competitor minimum (INR)', result?.competitor_stats?.min],
    ['Competitor median (INR)', result?.competitor_stats?.median], ['Competitor maximum (INR)', result?.competitor_stats?.max],
    ['Unit cost (INR)', result?.cost_price ?? form.cost_price], ['Other unit costs (INR)', result?.fixed_costs ?? form.fixed_costs],
    ['Minimum markup (%)', result?.min_markup_pct ?? form.min_markup_pct], ['Maximum markup (%)', result?.max_markup_pct ?? form.max_markup_pct],
    ['Month', result?.month ?? form.month], ['Rating', result?.rating ?? form.rating],
    ['Ad spend (INR)', result?.ad_spend ?? form.ad_spend], ['Available units', result?.available_units ?? form.available_units],
    ['Competitor prices (INR)', (result?.competitor_prices || []).join('; ')],
    ['Price curve (price INR | markup % | margin % | expected profit INR)', (result?.all_prices || []).map((point) => `${point.price} | ${point.markup_pct ?? ''} | ${point.margin_pct ?? ''} | ${point.expected_profit}`).join('; ')],
    ['Warnings', (result?.warnings || []).join('; ')],
  ];
  return [['Field', 'Value'], ...fields].map((row) => row.map(csvCell).join(',')).join('\r\n');
}

export function downloadResultCsv(result, form = {}) {
  const csv = `\uFEFF${makeResultCsv(result, form)}`;
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  const slug = (result?.product_name || form.product_name || 'optimization').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
  link.href = url;
  link.download = `${slug || 'optimization'}-price-report.csv`;
  link.click();
  URL.revokeObjectURL(url);
}
