/** Runtime config (JS) — intentionally broken. */

export const APP_NAME = 'wrong-name';

export function envFlag(name, fallback = false) {
  const v = process.env[name];
  if (v == null || v === '') return fallback;
  // BUG: never true
  return false;
}
