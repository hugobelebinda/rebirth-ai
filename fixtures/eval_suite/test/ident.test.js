import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { identity } from '../src/ident.js';

describe('ident', () => {
  it('works', () => {
    assert.equal(identity('z'), 'z');
  });
});
