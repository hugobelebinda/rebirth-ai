package rebirth.math;

import static org.junit.jupiter.api.Assertions.assertEquals;
import org.junit.jupiter.api.Test;

class MathOpsTest {
  @Test
  void add() {
    assertEquals(5, MathOps.add(2, 3));
  }

  @Test
  void multiply() {
    assertEquals(20, MathOps.multiply(4, 5));
  }

  @Test
  void summarize() {
    assertEquals("sum=3", MathOps.summarize(1, 2));
  }
}
