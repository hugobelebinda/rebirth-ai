import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { APP_NAME, envFlag } from '../src/config.js';

describe('config', () => {
  it('exposes app name', () => {
    assert.equal(APP_NAME, 'bench-mixed');
  });
  it('parses env flags', () => {
    process.env.REBIRTH_FLAG = 'true';
    assert.equal(envFlag('REBIRTH_FLAG'), true);
    delete process.env.REBIRTH_FLAG;
  });
});
