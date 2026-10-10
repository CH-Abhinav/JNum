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

public class SVDTest {

    @Test
    @DisplayName("Private constructor throws AssertionError")
    void testPrivateConstructor() throws Exception {
        Constructor<SVD> constructor = SVD.class.getDeclaredConstructor();
        constructor.setAccessible(true);
        InvocationTargetException ex = assertThrows(InvocationTargetException.class, constructor::newInstance);
        assertInstanceOf(AssertionError.class, ex.getCause());
    }

    @Test
    @DisplayName("Non-2D matrix throws IllegalArgumentException")
    void testInvalidShape() {
        try (Arena arena = Arena.ofConfined()) {
            NDArray v1 = JNum.zeros(arena, DType.f32, 4);
            assertThrows(IllegalArgumentException.class, () -> SVD.svd(v1, arena));

            NDArray t3 = JNum.zeros(arena, DType.f32, 2, 2, 2);
            assertThrows(IllegalArgumentException.class, () -> SVD.svd(t3, arena));
        }
    }

    @Test
    @DisplayName("Square matrix SVD: Float and Double")
    void testSquareSVD() {
        try (Arena arena = Arena.ofConfined()) {
            int n = 4;
            NDArray aF = TestArrayFactory.random(801L, DType.f32, n, n);
            SVD.SVDResult svdF = SVD.svd(aF, arena);
            NDArray uF = svdF.u();
            NDArray sF = svdF.s();
            NDArray vtF = svdF.vt();

            assertEquals(n, uF.internalShapeUnsafe()[0]);
            assertEquals(n, uF.internalShapeUnsafe()[1]);
            assertEquals(n, sF.internalShapeUnsafe()[0]);
            assertEquals(n, vtF.internalShapeUnsafe()[0]);
            assertEquals(n, vtF.internalShapeUnsafe()[1]);

            // Singular values non-negative and non-increasing
            for (int i = 0; i < n; i++) {
                assertTrue(sF.getFloat(i) >= -1e-6f, "Singular values must be non-negative");
                if (i > 0) {
                    assertTrue(sF.getFloat(i - 1) >= sF.getFloat(i) - 1e-5f, "Singular values must be sorted");
                }
            }

            // Reconstruct: A = U * diag(S) * Vt
            NDArray sDiagF = JNum.zeros(arena, DType.f32, n, n);
            for (int i = 0; i < n; i++) sDiagF.setFloat(sF.getFloat(i), i, i);
            NDArray uSF = JNum.zeros(arena, DType.f32, n, n);
            MatMul.matmulFloat(uF, sDiagF, uSF);
            NDArray reconF = JNum.zeros(arena, DType.f32, n, n);
            MatMul.matmulFloat(uSF, vtF, reconF);
            TestAssertions.assertNDArrayClose(aF, reconF, 1e-3, 1e-3);

            // Double
            NDArray aD = TestArrayFactory.random(802L, DType.f64, n, n);
            SVD.SVDResult svdD = SVD.svd(aD, arena);
            NDArray uD = svdD.u();
            NDArray sD = svdD.s();
            NDArray vtD = svdD.vt();

            NDArray sDiagD = JNum.zeros(arena, DType.f64, n, n);
            for (int i = 0; i < n; i++) sDiagD.setDouble(sD.getDouble(i), i, i);
            NDArray uSD = JNum.zeros(arena, DType.f64, n, n);
            MatMul.matmulDouble(uD, sDiagD, uSD);
            NDArray reconD = JNum.zeros(arena, DType.f64, n, n);
            MatMul.matmulDouble(uSD, vtD, reconD);
            TestAssertions.assertNDArrayClose(aD, reconD, 1e-6, 1e-6);
        }
    }

    @Test
    @DisplayName("Rectangular SVD: Tall (m > n) and Wide (m < n)")
    void testRectangularSVD() {
        try (Arena arena = Arena.ofConfined()) {
            // Tall: 5x3
            int m = 5, n = 3;
            NDArray aTall = TestArrayFactory.random(803L, DType.f64, m, n);
            SVD.SVDResult svdTall = SVD.svd(aTall, arena);
            assertEquals(m, svdTall.u().internalShapeUnsafe()[0]);
            assertEquals(m, svdTall.u().internalShapeUnsafe()[1]);
            assertEquals(n, svdTall.s().internalShapeUnsafe()[0]);
            assertEquals(n, svdTall.vt().internalShapeUnsafe()[0]);
            assertEquals(n, svdTall.vt().internalShapeUnsafe()[1]);

            // Wide: 3x5
            NDArray aWide = TestArrayFactory.random(804L, DType.f64, 3, 5);
            SVD.SVDResult svdWide = SVD.svd(aWide, arena);
            assertEquals(3, svdWide.u().internalShapeUnsafe()[0]);
            assertEquals(3, svdWide.u().internalShapeUnsafe()[1]);
            assertEquals(3, svdWide.s().internalShapeUnsafe()[0]);
            assertEquals(5, svdWide.vt().internalShapeUnsafe()[0]);
            assertEquals(5, svdWide.vt().internalShapeUnsafe()[1]);
        }
    }
}
