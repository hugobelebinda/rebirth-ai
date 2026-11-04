import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { uniq } from '../src/uniq.js';

describe('uniq', () => {
  it('works', () => {
    assert.deepEqual(uniq([1,1,2,2,3]), [1,2,3]);
  });
});
