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

public class SolveTest {

    @Test
    @DisplayName("Private constructor throws AssertionError")
    void testPrivateConstructor() throws Exception {
        Constructor<Solve> constructor = Solve.class.getDeclaredConstructor();
        constructor.setAccessible(true);
        InvocationTargetException ex = assertThrows(InvocationTargetException.class, constructor::newInstance);
        assertInstanceOf(AssertionError.class, ex.getCause());
    }

    @Test
    @DisplayName("Invalid shapes throw IllegalArgumentException")
    void testInvalidShapes() {
        try (Arena arena = Arena.ofConfined()) {
            NDArray nonSquareA = JNum.zeros(arena, DType.f32, 2, 3);
            NDArray b = JNum.zeros(arena, DType.f32, 2);
            assertThrows(IllegalArgumentException.class, () -> Solve.solve(nonSquareA, b, arena));

            NDArray a = JNum.zeros(arena, DType.f32, 3, 3);
            NDArray b3D = JNum.zeros(arena, DType.f32, 3, 1, 1);
            assertThrows(IllegalArgumentException.class, () -> Solve.solve(a, b3D, arena));

            NDArray bMismatch = JNum.zeros(arena, DType.f32, 4);
            assertThrows(IllegalArgumentException.class, () -> Solve.solve(a, bMismatch, arena));
        }
    }

    @Test
    @DisplayName("Singular matrix throws ArithmeticException")
    void testSingularMatrixThrows() {
        try (Arena arena = Arena.ofConfined()) {
            NDArray singularA = TestArrayFactory.matrix(new float[][]{{1f, 2f}, {2f, 4f}});
            NDArray b = JNum.from(new float[]{3f, 6f}, 2);
            assertThrows(ArithmeticException.class, () -> Solve.solve(singularA, b, arena));
        }
    }

    @Test
    @DisplayName("Solve 1D right-hand side Ax = b: Float and Double")
    void testSolve1DVector() {
        try (Arena arena = Arena.ofConfined()) {
            // 2x + y = 5, -3x + 4y = 9 -> x = 1, y = 3
            NDArray aF = TestArrayFactory.matrix(new float[][]{{2f, 1f}, {-3f, 4f}});
            NDArray bF = JNum.from(new float[]{5f, 9f}, 2);
            NDArray xF = Solve.solve(aF, bF, arena);

            assertEquals(1, xF.dim());
            assertEquals(2, xF.getSize());
            assertEquals(1.0f, xF.getFloat(0), 1e-5f);
            assertEquals(3.0f, xF.getFloat(1), 1e-5f);

            // Double
            NDArray aD = TestArrayFactory.matrix(new double[][]{{2.0, 1.0}, {-3.0, 4.0}});
            NDArray bD = JNum.from(new double[]{5.0, 9.0}, 2);
            NDArray xD = Solve.solve(aD, bD, arena);

            assertEquals(1, xD.dim());
            assertEquals(1.0, xD.getDouble(0), 1e-10);
            assertEquals(3.0, xD.getDouble(1), 1e-10);
        }
    }

    @Test
    @DisplayName("Solve 2D right-hand side AX = B (multiple RHS vectors)")
    void testSolve2DMatrix() {
        try (Arena arena = Arena.ofConfined()) {
            // A * X = I => X should equal A^-1
            int n = 4;
            NDArray aF = TestArrayFactory.spd(n).cast(DType.f32);
            NDArray bF = TestArrayFactory.eye(n, DType.f32);
            NDArray xF = Solve.solve(aF, bF, arena);

            NDArray prodF = JNum.zeros(arena, DType.f32, n, n);
            MatMul.matmulFloat(aF, xF, prodF);
            TestAssertions.assertNDArrayClose(bF, prodF, 1e-4, 1e-4);

            NDArray aD = TestArrayFactory.spd(n);
            NDArray bD = TestArrayFactory.eye(n, DType.f64);
            NDArray xD = Solve.solve(aD, bD, arena);

            NDArray prodD = JNum.zeros(arena, DType.f64, n, n);
            MatMul.matmulDouble(aD, xD, prodD);
            TestAssertions.assertNDArrayClose(bD, prodD, 1e-7, 1e-7);
        }
    }

    @Test
    @DisplayName("Non-contiguous inputs solve correctly")
    void testNonContiguousSolve() {
        try (Arena arena = Arena.ofConfined()) {
            NDArray a = TestArrayFactory.matrix(new double[][]{{3.0, 1.0}, {1.0, 2.0}});
            NDArray b = TestArrayFactory.matrix(new double[][]{{9.0, 0.0}, {8.0, 0.0}});
            // Take slice of b: column 0 as 1D
            NDArray bCol = b.slice(":, 0");
            NDArray x = Solve.solve(a, bCol, arena);
            // 3x + y = 9, x + 2y = 8 -> x = 2, y = 3
            assertEquals(2.0, x.getDouble(0), 1e-6);
            assertEquals(3.0, x.getDouble(1), 1e-6);
        }
    }
}
