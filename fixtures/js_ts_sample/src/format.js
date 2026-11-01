/** Tiny formatting utility referenced by math.js (RAG target). */

export function formatResult(label, value) {
  return label + '=' + String(value);
}

export function formatList(items) {
  return items.join(', ');
}
