/** Corrected and modernized for Phase 5 multi-file demo. */

export function formatResult(label, value) {
  return `${label}=${String(value)}`;
}

export function formatList(items) {
  return items.join(', ');
}
