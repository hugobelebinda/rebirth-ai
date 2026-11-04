import { trim } from './strings.js';

export function clamp(n, min, max) {
  return Math.min(max, Math.max(min, n));
}

export function parseIntSafe(raw, fallback = 0) {
  const n = Number.parseInt(trim(raw), 10);
  return Number.isNaN(n) ? fallback : n;
}

export function sum(values) {
  return values.reduce((acc, v) => acc + v, 0);
}
