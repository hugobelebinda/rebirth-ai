import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { formatList, formatResult } from '../src/format.js';

describe('format', () => {
  it('formats label=value', () => {
    assert.equal(formatResult('sum', 3), 'sum=3');
  });

  it('joins lists', () => {
    assert.equal(formatList(['a', 'b']), 'a, b');
  });
});
