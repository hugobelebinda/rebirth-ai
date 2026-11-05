package rebirth.math;

public final class MathOps {
  private MathOps() {}

  public static int add(int a, int b) {
    return a + b;
  }

  public static int multiply(int a, int b) {
    int product = a * b;
    return product;
  }

  public static String summarize(int a, int b) {
    return "sum=" + add(a, b);
  }
}
