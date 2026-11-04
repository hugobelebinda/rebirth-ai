import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { negate } from '../src/neg.js';

describe('neg', () => {
  it('works', () => {
    assert.equal(negate(5), -5);
  });
});
