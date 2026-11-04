import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { clamp, parseIntSafe, sum } from '../src/numbers.js';

describe('numbers', () => {
  it('clamps', () => {
    assert.equal(clamp(5, 0, 10), 5);
    assert.equal(clamp(-1, 0, 10), 0);
    assert.equal(clamp(99, 0, 10), 10);
  });
  it('parses ints safely', () => {
    assert.equal(parseIntSafe('42'), 42);
    assert.equal(parseIntSafe('x', 7), 7);
  });
  it('sums', () => {
    assert.equal(sum([1, 2, 3]), 6);
  });
});
