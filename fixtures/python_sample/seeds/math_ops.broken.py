"""Intentionally broken draft for Phase 8 closed-loop demo.
Tests expect multiply(a, b) == a * b; this returns a + b instead.
"""


def add(a: int, b: int) -> int:
    return a + b


def multiply(a: int, b: int) -> int:
    # BUG: should multiply
    return a + b


def summarize(a: int, b: int) -> str:
    total = add(a, b)
    return f"sum={total}"


class Calculator:
    def __init__(self, initial: int = 0) -> None:
        self.value = initial or 0

    def add(self, n: int) -> int:
        self.value = add(self.value, n)
        return self.value

    def multiply(self, n: int) -> int:
        self.value = multiply(self.value, n)
        return self.value

    def reset(self) -> None:
        self.value = 0
