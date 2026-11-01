const defaults = {
  quality: 'standard',
  available_units: '',
  num_competitors: '100',
  competitor_prices: '',
  rating: '4.2',
  min_markup_pct: '5',
  max_markup_pct: '80',
};

export const sampleProducts = [
  { id: 'wireless-earbuds', label: 'Wireless Earbuds', product_name: 'Wireless Earbuds', category: 'Electronics', cost_price: '1000', fixed_costs: '80', ad_spend: '1200', month: '6', ...defaults },
  { id: 'running-shoes', label: 'Running Shoes', product_name: 'Running Shoes', category: 'Sports', cost_price: '1800', fixed_costs: '120', ad_spend: '1600', month: '7', ...defaults },
  { id: 'face-serum', label: 'Face Serum', product_name: 'Face Serum', category: 'Beauty', cost_price: '450', fixed_costs: '35', ad_spend: '900', month: '4', ...defaults },
  { id: 'cotton-kurta', label: 'Cotton Kurta', product_name: 'Cotton Kurta', category: 'Clothing', cost_price: '600', fixed_costs: '45', ad_spend: '700', month: '10', ...defaults },
  { id: 'yoga-mat', label: 'Yoga Mat', product_name: 'Yoga Mat', category: 'Home', cost_price: '700', fixed_costs: '55', ad_spend: '650', month: '1', ...defaults },
];
