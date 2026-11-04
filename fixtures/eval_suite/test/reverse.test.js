import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { reverse } from '../src/reverse.js';

describe('reverse', () => {
  it('works', () => {
    assert.deepEqual(reverse([1,2,3]), [3,2,1]);
  });
});
