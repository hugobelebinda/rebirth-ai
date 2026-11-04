import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { padLeft } from '../src/pad.js';

describe('pad', () => {
  it('works', () => {
    assert.equal(padLeft('7', 3), '  7');
  });
});
