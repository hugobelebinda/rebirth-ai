from math_ops import Calculator, add, multiply, summarize


def test_add() -> None:
    assert add(2, 3) == 5


def test_multiply() -> None:
    assert multiply(4, 5) == 20


def test_summarize() -> None:
    assert summarize(1, 2) == "sum=3"


def test_calculator() -> None:
    calc = Calculator(10)
    assert calc.add(5) == 15
    assert calc.multiply(2) == 30
