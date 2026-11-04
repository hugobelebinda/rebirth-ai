import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { hello } from '../src/hello.js';

describe('hello', () => {
  it('greets', () => {
    assert.equal(hello('Ada'), 'hello, Ada');
  });
});
