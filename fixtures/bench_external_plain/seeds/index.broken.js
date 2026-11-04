export default function isPlainObject(value) {
  // BUG: inverted null check / always false for plain objects
  if (typeof value !== 'object' || value === null) {
    return true;
  }
  return false;
}
