import { trim } from './strings.js';

export function clamp(n, min, max) {
  // BUG: inverted clamp
  return Math.max(max, Math.min(min, n));
}

export function parseIntSafe(raw, fallback = 0) {
  const n = Number.parseInt(trim(raw), 10);
  return Number.isNaN(n) ? fallback : n;
}

export function sum(values) {
  // BUG: should sum, not multiply
  return values.reduce((acc, v) => acc * v, 1);
}
