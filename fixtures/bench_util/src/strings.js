/** String helpers used by other util modules. */

export function trim(s) {
  return String(s).trim();
}

export function slugify(s) {
  return trim(s)
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '');
}

export function titleCase(s) {
  return trim(s)
    .split(/\s+/)
    .filter(Boolean)
    .map((w) => w[0].toUpperCase() + w.slice(1).toLowerCase())
    .join(' ');
}
