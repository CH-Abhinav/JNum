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

public class InvTest {

    @Test
    @DisplayName("Private constructor throws AssertionError")
    void testPrivateConstructor() throws Exception {
        Constructor<Inv> constructor = Inv.class.getDeclaredConstructor();
        constructor.setAccessible(true);
        InvocationTargetException ex = assertThrows(InvocationTargetException.class, constructor::newInstance);
        assertInstanceOf(AssertionError.class, ex.getCause());
    }

    @Test
    @DisplayName("Non-square or non-2D matrices throw IllegalArgumentException")
    void testInvalidShape() {
        try (Arena arena = Arena.ofConfined()) {
            NDArray rect = JNum.zeros(arena, DType.f32, 2, 3);
            assertThrows(IllegalArgumentException.class, () -> Inv.inv(rect, arena));

            NDArray tensor3D = JNum.zeros(arena, DType.f32, 2, 2, 2);
            assertThrows(IllegalArgumentException.class, () -> Inv.inv(tensor3D, arena));
        }
    }

    @Test
    @DisplayName("Singular matrix throws ArithmeticException")
    void testSingularMatrixThrows() {
        try (Arena arena = Arena.ofConfined()) {
            NDArray singularF = TestArrayFactory.matrix(new float[][]{
                {1f, 2f},
                {2f, 4f}
            });
            assertThrows(ArithmeticException.class, () -> Inv.invFloat(singularF, arena));

            NDArray singularD = TestArrayFactory.matrix(new double[][]{
                {1.0, 2.0},
                {2.0, 4.0}
            });
            assertThrows(ArithmeticException.class, () -> Inv.invDouble(singularD, arena));
        }
    }

    @Test
    @DisplayName("1x1 matrix inversion")
    void test1x1Matrix() {
        try (Arena arena = Arena.ofConfined()) {
            NDArray mF = TestArrayFactory.matrix(new float[][]{{4.0f}});
            NDArray invF = Inv.invFloat(mF, arena);
            assertEquals(0.25f, invF.getFloat(0, 0), 1e-6f);

            NDArray mD = TestArrayFactory.matrix(new double[][]{{-2.5}});
            NDArray invD = Inv.invDouble(mD, arena);
            assertEquals(-0.4, invD.getDouble(0, 0), 1e-12);
        }
    }

    @Test
    @DisplayName("2x2 known analytical matrix inversion")
    void test2x2Analytical() {
        try (Arena arena = Arena.ofConfined()) {
            // A = [[4, 7], [2, 6]], det = 24 - 14 = 10
            // A^-1 = [[0.6, -0.7], [-0.2, 0.4]]
            NDArray aF = TestArrayFactory.matrix(new float[][]{{4f, 7f}, {2f, 6f}});
            NDArray invF = Inv.inv(aF, arena);
            assertEquals(0.6f, invF.getFloat(0, 0), 1e-5f);
            assertEquals(-0.7f, invF.getFloat(0, 1), 1e-5f);
            assertEquals(-0.2f, invF.getFloat(1, 0), 1e-5f);
            assertEquals(0.4f, invF.getFloat(1, 1), 1e-5f);

            NDArray aD = TestArrayFactory.matrix(new double[][]{{4.0, 7.0}, {2.0, 6.0}});
            NDArray invD = Inv.inv(aD, arena);
            assertEquals(0.6, invD.getDouble(0, 0), 1e-10);
            assertEquals(-0.7, invD.getDouble(0, 1), 1e-10);
            assertEquals(-0.2, invD.getDouble(1, 0), 1e-10);
            assertEquals(0.4, invD.getDouble(1, 1), 1e-10);
        }
    }

    @Test
    @DisplayName("A * A^-1 == I for well-conditioned matrix")
    void testIdentityProperty() {
        try (Arena arena = Arena.ofConfined()) {
            int n = 8;
            NDArray aF = TestArrayFactory.spd(n).cast(DType.f32);
            NDArray invF = Inv.inv(aF, arena);
            NDArray prodF = JNum.zeros(arena, DType.f32, n, n);
            MatMul.matmulFloat(aF, invF, prodF);
            NDArray eyeF = TestArrayFactory.eye(n, DType.f32);
            TestAssertions.assertNDArrayClose(eyeF, prodF, 1e-3, 1e-3);

            NDArray aD = TestArrayFactory.spd(n);
            NDArray invD = Inv.inv(aD, arena);
            NDArray prodD = JNum.zeros(arena, DType.f64, n, n);
            MatMul.matmulDouble(aD, invD, prodD);
            NDArray eyeD = TestArrayFactory.eye(n, DType.f64);
            TestAssertions.assertNDArrayClose(eyeD, prodD, 1e-6, 1e-6);
        }
    }

    @Test
    @DisplayName("Transposed non-contiguous matrix inversion")
    void testTransposedMatrixInversion() {
        try (Arena arena = Arena.ofConfined()) {
            NDArray a = TestArrayFactory.matrix(new double[][]{
                {5.0, 2.0},
                {1.0, 3.0}
            });
            NDArray aT = a.transpose();
            NDArray invAT = Inv.inv(aT, arena);
            NDArray invA = Inv.inv(a, arena);
            NDArray invATExpected = invA.transpose();
            TestAssertions.assertNDArrayClose(invATExpected, invAT, 1e-9, 1e-9);
        }
    }
}
