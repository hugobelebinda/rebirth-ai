import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { isTruthy } from '../src/truthy.js';

describe('truthy', () => {
  it('works', () => {
    assert.equal(isTruthy(1), true);
  assert.equal(isTruthy(0), false);
  });
});
