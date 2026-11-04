import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { join } from '../src/join.js';

describe('join', () => {
  it('works', () => {
    assert.equal(join(['a','b'], '-'), 'a-b');
  });
});
