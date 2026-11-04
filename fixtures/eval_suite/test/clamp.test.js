import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { clamp } from '../src/clamp.js';

describe('clamp', () => {
  it('works', () => {
    assert.equal(clamp(5, 0, 10), 5);
  assert.equal(clamp(-1, 0, 10), 0);
  assert.equal(clamp(99, 0, 10), 10);
  });
});
