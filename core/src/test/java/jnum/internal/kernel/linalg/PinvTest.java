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

public class PinvTest {

    @Test
    @DisplayName("Private constructor throws AssertionError")
    void testPrivateConstructor() throws Exception {
        Constructor<Pinv> constructor = Pinv.class.getDeclaredConstructor();
        constructor.setAccessible(true);
        InvocationTargetException ex = assertThrows(InvocationTargetException.class, constructor::newInstance);
        assertInstanceOf(AssertionError.class, ex.getCause());
    }

    @Test
    @DisplayName("Non-2D matrix throws IllegalArgumentException")
    void testInvalidShape() {
        try (Arena arena = Arena.ofConfined()) {
            NDArray v1 = JNum.zeros(arena, DType.f32, 4);
            assertThrows(IllegalArgumentException.class, () -> Pinv.pinv(v1, arena));

            NDArray t3 = JNum.zeros(arena, DType.f32, 2, 2, 2);
            assertThrows(IllegalArgumentException.class, () -> Pinv.pinv(t3, arena));
        }
    }

    @Test
    @DisplayName("Square invertible matrix: pinv(A) == inv(A)")
    void testSquareInvertible() {
        try (Arena arena = Arena.ofConfined()) {
            NDArray aF = TestArrayFactory.matrix(new float[][]{{4f, 7f}, {2f, 6f}});
            NDArray pinvF = Pinv.pinv(aF, arena);
            NDArray invF = Inv.inv(aF, arena);
            TestAssertions.assertNDArrayClose(invF, pinvF, 1e-4, 1e-4);

            NDArray aD = TestArrayFactory.matrix(new double[][]{{4.0, 7.0}, {2.0, 6.0}});
            NDArray pinvD = Pinv.pinv(aD, arena);
            NDArray invD = Inv.inv(aD, arena);
            TestAssertions.assertNDArrayClose(invD, pinvD, 1e-9, 1e-9);
        }
    }

    @Test
    @DisplayName("Moore-Penrose conditions on rectangular matrices: A * A^+ * A == A")
    void testMoorePenroseProperty() {
        try (Arena arena = Arena.ofConfined()) {
            // Tall: 4x2
            int m = 4, n = 2;
            NDArray aF = TestArrayFactory.random(1101L, DType.f32, m, n);
            NDArray pinvF = Pinv.pinv(aF, arena);
            assertEquals(n, pinvF.internalShapeUnsafe()[0]);
            assertEquals(m, pinvF.internalShapeUnsafe()[1]);

            // A * A^+ -> (4, 4)
            NDArray aPinvF = JNum.zeros(arena, DType.f32, m, m);
            MatMul.matmulFloat(aF, pinvF, aPinvF);
            // (A * A^+) * A -> (4, 2)
            NDArray reconF = JNum.zeros(arena, DType.f32, m, n);
            MatMul.matmulFloat(aPinvF, aF, reconF);
            TestAssertions.assertNDArrayClose(aF, reconF, 1e-3, 1e-3);

            // Wide: 2x4
            NDArray aD = TestArrayFactory.random(1102L, DType.f64, 2, 4);
            NDArray pinvD = Pinv.pinv(aD, arena);
            assertEquals(4, pinvD.internalShapeUnsafe()[0]);
            assertEquals(2, pinvD.internalShapeUnsafe()[1]);

            NDArray aPinvD = JNum.zeros(arena, DType.f64, 2, 2);
            MatMul.matmulDouble(aD, pinvD, aPinvD);
            NDArray reconD = JNum.zeros(arena, DType.f64, 2, 4);
            MatMul.matmulDouble(aPinvD, aD, reconD);
            TestAssertions.assertNDArrayClose(aD, reconD, 1e-6, 1e-6);
        }
    }

    @Test
    @DisplayName("Singular / rank-deficient matrix pseudo-inverse with rcond")
    void testSingularPinv() {
        try (Arena arena = Arena.ofConfined()) {
            // Rank-1 2x2 matrix
            NDArray rank1 = TestArrayFactory.matrix(new double[][]{
                {1.0, 2.0},
                {2.0, 4.0}
            });
            NDArray pinv = Pinv.pinv(rank1, 1e-4, arena);
            // Reconstructed A * pinv(A) * A == A
            NDArray step1 = JNum.zeros(arena, DType.f64, 2, 2);
            MatMul.matmulDouble(rank1, pinv, step1);
            NDArray recon = JNum.zeros(arena, DType.f64, 2, 2);
            MatMul.matmulDouble(step1, rank1, recon);
            TestAssertions.assertNDArrayClose(rank1, recon, 1e-6, 1e-6);
        }
    }
}
