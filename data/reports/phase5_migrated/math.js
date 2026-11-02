/**
 * Math operations and utilities.
 */

import { formatResult } from './format.js';

export function add(a, b) {
  return a + b;
}

export function multiply(a, b) {
  // Corrected to perform multiplication
  return a * b;
}

export function summarize(a, b) {
  const total = add(a, b);
  return formatResult('sum', total);
}

export class Calculator {
  constructor(initial = 0) {
    this.value = initial;
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
