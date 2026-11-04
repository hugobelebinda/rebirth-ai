import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { add } from '../src/add.js';

describe('add', () => {
  it('works', () => {
    assert.equal(add(2, 3), 5);
  });
});
