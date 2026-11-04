import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { slugify, titleCase, trim } from '../src/strings.js';

describe('strings', () => {
  it('trims', () => {
    assert.equal(trim('  hi  '), 'hi');
  });
  it('slugifies', () => {
    assert.equal(slugify('Hello World!'), 'hello-world');
  });
  it('title cases', () => {
    assert.equal(titleCase('hello world'), 'Hello World');
  });
});
