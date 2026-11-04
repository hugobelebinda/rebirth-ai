import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { formatDuration } from '../src/duration.js';

describe('formatDuration', () => {
  it('formats sub-second', () => {
    assert.equal(formatDuration(20), '20ms');
  });
  it('formats seconds', () => {
    assert.equal(formatDuration(5000), '5s');
  });
});
