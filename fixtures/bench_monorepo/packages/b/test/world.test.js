import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { world } from '../src/world.js';

describe('world', () => {
  it('returns world', () => {
    assert.equal(world(), 'world');
  });
});
