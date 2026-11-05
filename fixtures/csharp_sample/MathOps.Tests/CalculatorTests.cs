using MathOps;
using Xunit;

public class CalculatorTests
{
    [Fact]
    public void Add() => Assert.Equal(5, Calculator.Add(2, 3));

    [Fact]
    public void Multiply() => Assert.Equal(20, Calculator.Multiply(4, 5));
}
