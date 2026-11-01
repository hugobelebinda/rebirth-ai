/** TypeScript sample for AST coverage. */

export interface Person {
  name: string;
  age?: number;
}

export function greet(person: Person): string {
  return `Hello, ${person.name}`;
}

export class Greeter {
  constructor(private readonly prefix: string = 'Hi') {}

  say(name: string): string {
    return `${this.prefix}, ${name}`;
  }
}
