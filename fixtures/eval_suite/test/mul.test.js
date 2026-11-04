import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { multiply } from '../src/mul.js';

describe('mul', () => {
  it('works', () => {
    assert.equal(multiply(4, 5), 20);
  });
});
