/** Runtime config (JS). */

export const APP_NAME = 'bench-mixed';

export function envFlag(name, fallback = false) {
  const v = process.env[name];
  if (v == null || v === '') return fallback;
  return v === '1' || v.toLowerCase() === 'true';
}
