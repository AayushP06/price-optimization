const STORAGE_KEY = 'priceoptimizer.saved-presets.v1';

function browserStorage() {
  try { return window.localStorage; } catch { return null; }
}

export function readPresets(storage = browserStorage()) {
  try {
    if (!storage) return { presets: [], error: 'Saved presets are unavailable in this browser.' };
    const raw = storage.getItem(STORAGE_KEY);
    if (!raw) return { presets: [], error: null };
    const value = JSON.parse(raw);
    return { presets: Array.isArray(value) ? value.filter((item) => item && item.id && item.name && item.form) : [], error: null };
  } catch {
    return { presets: [], error: 'Could not read saved presets from this browser.' };
  }
}

export function writePresets(presets, storage = browserStorage()) {
  try {
    if (!storage) return { ok: false, error: 'Saved presets are unavailable in this browser.' };
    storage.setItem(STORAGE_KEY, JSON.stringify(presets));
    return { ok: true, error: null };
  } catch {
    return { ok: false, error: 'Could not save presets. Check browser storage availability.' };
  }
}

export function savePreset(presets, name, form, storage = browserStorage()) {
  const trimmedName = String(name || '').trim();
  if (!trimmedName) return { presets, ok: false, error: 'Enter a name for this preset.' };
  const next = [...presets, { id: `preset-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`, name: trimmedName, form: { ...form } }];
  const saved = writePresets(next, storage);
  return { presets: saved.ok ? next : presets, ...saved };
}

export function deletePreset(presets, id, storage = browserStorage()) {
  const next = presets.filter((preset) => preset.id !== id);
  const saved = writePresets(next, storage);
  return { presets: saved.ok ? next : presets, ...saved };
}
