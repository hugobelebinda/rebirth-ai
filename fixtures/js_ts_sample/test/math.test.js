import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { Calculator, add, multiply, summarize } from '../src/math.js';

describe('math', () => {
  it('adds numbers', () => {
    assert.equal(add(2, 3), 5);
  });

  it('multiplies numbers', () => {
    assert.equal(multiply(4, 5), 20);
  });

  it('summarizes with formatter', () => {
    assert.equal(summarize(1, 2), 'sum=3');
  });

  it('calculator accumulates', () => {
    const calc = new Calculator(10);
    assert.equal(calc.add(5), 15);
    assert.equal(calc.multiply(2), 30);
  });
});
