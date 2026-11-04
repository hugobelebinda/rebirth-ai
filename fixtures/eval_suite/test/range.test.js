import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { range } from '../src/range.js';

describe('range', () => {
  it('works', () => {
    assert.deepEqual(range(3), [0,1,2]);
  });
});
