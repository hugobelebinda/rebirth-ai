import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import isPlainObject from '../index.js';

describe('isPlainObject', () => {
  it('accepts plain objects', () => {
    assert.equal(isPlainObject({ a: 1 }), true);
    assert.equal(isPlainObject(Object.create(null)), true);
  });
  it('rejects non-plain', () => {
    assert.equal(isPlainObject(null), false);
    assert.equal(isPlainObject([]), false);
    assert.equal(isPlainObject(new Date()), false);
  });
});
