import { useEffect, useState } from 'react';
import { deletePreset, readPresets, savePreset } from '../utils/presetStorage';

export function useSavedPresets() {
  const [presets, setPresets] = useState([]);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');

  useEffect(() => {
    const result = readPresets();
    setPresets(result.presets);
    setError(result.error || '');
  }, []);

  function save(name, form) {
    const result = savePreset(presets, name, form);
    if (result.ok) {
      setPresets(result.presets);
      setNotice('Preset saved on this device.');
      setError('');
    } else {
      setError(result.error);
      setNotice('');
    }
    return result.ok;
  }

  function remove(id) {
    const result = deletePreset(presets, id);
    if (result.ok) {
      setPresets(result.presets);
      setError('');
    } else setError(result.error);
  }

  return { presets, error, notice, save, remove };
}
