/**
 * Intentionally broken draft for Phase 4 closed-loop demo.
 * Tests expect multiply(a,b) === a*b; this returns a+b instead.
 */

import { formatResult } from './format.js';

export function add(a, b) {
  return a + b;
}

export function multiply(a, b) {
  // BUG: should multiply
  return a + b;
}

export function summarize(a, b) {
  const total = add(a, b);
  return formatResult('sum', total);
}

export class Calculator {
  constructor(initial) {
    this.value = initial || 0;
  }

  add(n) {
    this.value = add(this.value, n);
    return this.value;
  }

  multiply(n) {
    this.value = multiply(this.value, n);
    return this.value;
  }

  reset() {
    this.value = 0;
  }
}
