export class ApiError extends Error {
  constructor(message, fields = {}) {
    super(message);
    this.name = 'ApiError';
    this.fields = fields;
  }
}

export async function optimizePrice(payload) {
  const response = await fetch('/api/optimize', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  let body;
  try {
    body = await response.json();
  } catch {
    throw new ApiError(`The server returned an unreadable response (${response.status}).`);
  }
  if (!response.ok) {
    throw new ApiError(body.error || 'Please correct the highlighted fields.', body.errors || {});
  }
  if (!body.success) throw new ApiError(body.error || 'Optimization failed.');
  return body;
}
