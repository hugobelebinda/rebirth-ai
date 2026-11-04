/** TypeScript greeter depending on JS config. */

import { APP_NAME } from './config.js';

export function greet(name: string): string {
  return `${APP_NAME}: hello, ${name}`;
}

export function shout(name: string): string {
  return greet(name).toUpperCase();
}
