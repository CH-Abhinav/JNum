package jnum.internal.kernel.linalg;

import static org.junit.jupiter.api.Assertions.*;

import java.lang.foreign.Arena;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import jnum.DType;
import jnum.JNum;
import jnum.NDArray;
import jnum.testutil.TestArrayFactory;

public class CondTest {

    @Test
    @DisplayName("Private constructor throws AssertionError")
    void testPrivateConstructor() throws Exception {
        Constructor<Cond> constructor = Cond.class.getDeclaredConstructor();
        constructor.setAccessible(true);
        InvocationTargetException ex = assertThrows(InvocationTargetException.class, constructor::newInstance);
        assertInstanceOf(AssertionError.class, ex.getCause());
    }

    @Test
    @DisplayName("Non-2D matrix throws IllegalArgumentException")
    void testInvalidShape() {
        try (Arena arena = Arena.ofConfined()) {
            NDArray v1 = JNum.zeros(arena, DType.f32, 4);
            assertThrows(IllegalArgumentException.class, () -> Cond.cond(v1, arena));

            NDArray t3 = JNum.zeros(arena, DType.f32, 2, 2, 2);
            assertThrows(IllegalArgumentException.class, () -> Cond.cond(t3, arena));
        }
    }

    @Test
    @DisplayName("Zero dimension matrix returns 0.0")
    void testZeroDimension() {
        try (Arena arena = Arena.ofConfined()) {
            NDArray empty = JNum.zeros(arena, DType.f32, 0, 4);
            assertEquals(0.0, Cond.cond(empty, arena));
        }
    }

    @Test
    @DisplayName("Identity matrix condition number is 1.0: Float and Double")
    void testIdentityCondition() {
        try (Arena arena = Arena.ofConfined()) {
            NDArray eyeF = TestArrayFactory.eye(4, DType.f32);
            assertEquals(1.0, Cond.cond(eyeF, arena), 1e-5);

            NDArray eyeD = TestArrayFactory.eye(4, DType.f64);
            assertEquals(1.0, Cond.cond(eyeD, arena), 1e-10);
        }
    }

    @Test
    @DisplayName("Known diagonal matrix condition number")
    void testDiagonalCondition() {
        try (Arena arena = Arena.ofConfined()) {
            // diag(10, 2) -> sigma_max = 10, sigma_min = 2 -> cond = 5.0
            NDArray diagF = TestArrayFactory.matrix(new float[][]{
                {10f, 0f},
                {0f, 2f}
            });
            assertEquals(5.0, Cond.cond(diagF, arena), 1e-4);

            NDArray diagD = TestArrayFactory.matrix(new double[][]{
                {100.0, 0.0},
                {0.0, 4.0}
            });
            assertEquals(25.0, Cond.cond(diagD, arena), 1e-9);
        }
    }

    @Test
    @DisplayName("Singular matrix condition number is POSITIVE_INFINITY")
    void testSingularCondition() {
        try (Arena arena = Arena.ofConfined()) {
            NDArray singularF = TestArrayFactory.matrix(new float[][]{
                {1f, 2f},
                {2f, 4f}
            });
            assertEquals(Double.POSITIVE_INFINITY, Cond.cond(singularF, arena));

            NDArray singularD = TestArrayFactory.matrix(new double[][]{
                {1.0, 2.0},
                {2.0, 4.0}
            });
            assertEquals(Double.POSITIVE_INFINITY, Cond.cond(singularD, arena));
        }
    }
}
