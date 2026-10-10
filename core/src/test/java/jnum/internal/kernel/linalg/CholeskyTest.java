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
import jnum.testutil.TestAssertions;

public class CholeskyTest {

    @Test
    @DisplayName("Private constructor throws AssertionError")
    void testPrivateConstructor() throws Exception {
        Constructor<Cholesky> constructor = Cholesky.class.getDeclaredConstructor();
        constructor.setAccessible(true);
        InvocationTargetException ex = assertThrows(InvocationTargetException.class, constructor::newInstance);
        assertInstanceOf(AssertionError.class, ex.getCause());
    }

    @Test
    @DisplayName("Invalid shape throws IllegalArgumentException")
    void testInvalidShape() {
        try (Arena arena = Arena.ofConfined()) {
            NDArray rect = JNum.zeros(arena, DType.f32, 2, 3);
            assertThrows(IllegalArgumentException.class, () -> Cholesky.cholesky(rect, arena));

            NDArray t3 = JNum.zeros(arena, DType.f32, 2, 2, 2);
            assertThrows(IllegalArgumentException.class, () -> Cholesky.cholesky(t3, arena));
        }
    }

    @Test
    @DisplayName("Non-positive definite matrix throws ArithmeticException")
    void testNonPositiveDefinite() {
        try (Arena arena = Arena.ofConfined()) {
            // Negative diagonal entry
            NDArray nonSPD = TestArrayFactory.matrix(new float[][]{
                {-1.0f, 0.0f},
                {0.0f, 1.0f}
            });
            assertThrows(ArithmeticException.class, () -> Cholesky.cholesky(nonSPD, arena));

            // Positive diagonal, but negative determinant (indefinite)
            NDArray indefinite = TestArrayFactory.matrix(new double[][]{
                {1.0, 5.0},
                {5.0, 1.0}
            });
            assertThrows(ArithmeticException.class, () -> Cholesky.cholesky(indefinite, arena));
        }
    }

    @Test
    @DisplayName("1x1 matrix Cholesky")
    void test1x1Matrix() {
        try (Arena arena = Arena.ofConfined()) {
            NDArray mF = TestArrayFactory.matrix(new float[][]{{9.0f}});
            NDArray lF = Cholesky.cholesky(mF, arena);
            assertEquals(3.0f, lF.getFloat(0, 0), 1e-6f);

            NDArray mD = TestArrayFactory.matrix(new double[][]{{16.0}});
            NDArray lD = Cholesky.cholesky(mD, arena);
            assertEquals(4.0, lD.getDouble(0, 0), 1e-12);
        }
    }

    @Test
    @DisplayName("Known 2x2 SPD matrix decomposition")
    void test2x2SPD() {
        try (Arena arena = Arena.ofConfined()) {
            // A = [[4, 12], [12, 45]]
            // L = [[2, 0], [6, 3]]
            NDArray aF = TestArrayFactory.matrix(new float[][]{
                {4f, 12f},
                {12f, 45f}
            });
            NDArray lF = Cholesky.cholesky(aF, arena);
            assertEquals(2.0f, lF.getFloat(0, 0), 1e-5f);
            assertEquals(0.0f, lF.getFloat(0, 1), 1e-5f);
            assertEquals(6.0f, lF.getFloat(1, 0), 1e-5f);
            assertEquals(3.0f, lF.getFloat(1, 1), 1e-5f);
        }
    }

    @Test
    @DisplayName("Cholesky L * L^T == A for random SPD matrix: Float and Double")
    void testReconstructionSPD() {
        try (Arena arena = Arena.ofConfined()) {
            int n = 8;
            NDArray aF = TestArrayFactory.spd(n).cast(DType.f32);
            NDArray lF = Cholesky.cholesky(aF, arena);

            // Verify L is lower triangular
            for (int i = 0; i < n; i++) {
                for (int j = i + 1; j < n; j++) {
                    assertEquals(0.0f, lF.getFloat(i, j), 1e-6f);
                }
            }

            NDArray reconF = JNum.zeros(arena, DType.f32, n, n);
            MatMul.matmulFloat(lF, lF.transpose(), reconF);
            TestAssertions.assertNDArrayClose(aF, reconF, 1e-3, 1e-3);

            // Double
            NDArray aD = TestArrayFactory.spd(n);
            NDArray lD = Cholesky.cholesky(aD, arena);

            NDArray reconD = JNum.zeros(arena, DType.f64, n, n);
            MatMul.matmulDouble(lD, lD.transpose(), reconD);
            TestAssertions.assertNDArrayClose(aD, reconD, 1e-6, 1e-6);
        }
    }
}
