import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { formatKv } from '../src/kv.js';

describe('kv', () => {
  it('works', () => {
    assert.equal(formatKv('a', 1), 'a=1');
  });
});
