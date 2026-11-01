import { describe, expect, it, vi } from 'vitest';
import { deletePreset, readPresets, savePreset } from './presetStorage';

function memoryStorage(initial = null) {
  let value = initial;
  return {
    getItem: vi.fn(() => value),
    setItem: vi.fn((key, next) => { value = next; }),
    inspect: () => value,
  };
}

describe('saved presets storage', () => {
  it('handles empty storage and saves and deletes a named form', () => {
    const storage = memoryStorage();
    expect(readPresets(storage)).toEqual({ presets: [], error: null });
    const saved = savePreset([], 'Earbuds', { cost_price: '1000' }, storage);
    expect(saved.ok).toBe(true);
    expect(JSON.parse(storage.inspect())[0]).toMatchObject({ name: 'Earbuds', form: { cost_price: '1000' } });
    const removed = deletePreset(saved.presets, saved.presets[0].id, storage);
    expect(removed.ok).toBe(true);
    expect(JSON.parse(storage.inspect())).toEqual([]);
  });

  it('reports browser storage read and write failures without throwing', () => {
    const unavailable = { getItem: () => { throw new Error('blocked'); }, setItem: () => { throw new Error('blocked'); } };
    expect(readPresets(unavailable).error).toMatch(/Could not read/);
    expect(savePreset([], 'Test', {}, unavailable)).toMatchObject({ ok: false, presets: [] });
    expect(savePreset([], '', {}, memoryStorage()).error).toMatch(/name/);
  });
});
