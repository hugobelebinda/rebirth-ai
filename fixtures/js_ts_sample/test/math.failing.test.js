/**
 * Deliberately failing suite used by sandbox integration tests.
 * Not included in `npm test` (only math.test.js runs by default).
 */
import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { add } from '../src/math.js';

describe('math failing fixture', () => {
  it('expects the wrong sum on purpose', () => {
    assert.equal(add(2, 3), 999);
  });
});
