/** Format milliseconds as a short human string (no extra deps). */

export function formatDuration(msValue) {
  const n = Number(msValue);
  if (!Number.isFinite(n) || n < 0) return '0ms';
  if (n < 1000) return `${Math.round(n)}ms`;
  return `${n / 1000}s`;
}
