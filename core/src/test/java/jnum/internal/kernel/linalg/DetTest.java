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

public class DetTest {

    @Test
    @DisplayName("Private constructor throws AssertionError")
    void testPrivateConstructor() throws Exception {
        Constructor<Det> constructor = Det.class.getDeclaredConstructor();
        constructor.setAccessible(true);
        InvocationTargetException ex = assertThrows(InvocationTargetException.class, constructor::newInstance);
        assertInstanceOf(AssertionError.class, ex.getCause());
    }

    @Test
    @DisplayName("Invalid shape throws IllegalArgumentException")
    void testInvalidShape() {
        try (Arena arena = Arena.ofConfined()) {
            NDArray nonSquare = JNum.zeros(arena, DType.f32, 3, 4);
            assertThrows(IllegalArgumentException.class, () -> Det.detFloat(nonSquare));
            assertThrows(IllegalArgumentException.class, () -> Det.detDouble(nonSquare));
            assertThrows(IllegalArgumentException.class, () -> Det.slogdet(nonSquare));

            NDArray non2D = JNum.zeros(arena, DType.f32, 2, 2, 2);
            assertThrows(IllegalArgumentException.class, () -> Det.detFloat(non2D));
            assertThrows(IllegalArgumentException.class, () -> Det.detDouble(non2D));
            assertThrows(IllegalArgumentException.class, () -> Det.slogdet(non2D));
        }
    }

    @Test
    @DisplayName("Boundary dimensions: 0x0 and 1x1 matrices")
    void testBoundaryDimensions() {
        try (Arena arena = Arena.ofConfined()) {
            NDArray empty = JNum.zeros(arena, DType.f32, 0, 0);
            assertEquals(1.0f, Det.detFloat(empty));
            assertEquals(1.0, Det.detDouble(empty));
            Det.SlogdetResult sEmpty = Det.slogdet(empty);
            assertEquals(1.0, sEmpty.sign());
            assertEquals(0.0, sEmpty.logAbsDet());

            NDArray singleF = TestArrayFactory.matrix(new float[][]{{5.5f}});
            assertEquals(5.5f, Det.detFloat(singleF), 1e-6f);

            NDArray singleD = TestArrayFactory.matrix(new double[][]{{-4.2}});
            assertEquals(-4.2, Det.detDouble(singleD), 1e-12);
            Det.SlogdetResult sSingle = Det.slogdet(singleD);
            assertEquals(-1.0, sSingle.sign());
            assertEquals(Math.log(4.2), sSingle.logAbsDet(), 1e-12);
        }
    }

    @Test
    @DisplayName("2x2 and 3x3 known determinants: Float and Double")
    void testKnownDeterminants() {
        try (Arena arena = Arena.ofConfined()) {
            // [ [1, 2], [3, 4] ] -> det = 1*4 - 2*3 = -2
            NDArray m2F = TestArrayFactory.matrix(new float[][]{{1f, 2f}, {3f, 4f}});
            assertEquals(-2.0f, Det.detFloat(m2F), 1e-5f);

            NDArray m2D = TestArrayFactory.matrix(new double[][]{{1.0, 2.0}, {3.0, 4.0}});
            assertEquals(-2.0, Det.detDouble(m2D), 1e-10);

            // [ [6, 1, 1], [4, -2, 5], [2, 8, 7] ] -> det = -306
            NDArray m3F = TestArrayFactory.matrix(new float[][]{
                {6f, 1f, 1f},
                {4f, -2f, 5f},
                {2f, 8f, 7f}
            });
            assertEquals(-306.0f, Det.detFloat(m3F), 1e-4f);

            NDArray m3D = TestArrayFactory.matrix(new double[][]{
                {6.0, 1.0, 1.0},
                {4.0, -2.0, 5.0},
                {2.0, 8.0, 7.0}
            });
            assertEquals(-306.0, Det.detDouble(m3D), 1e-9);

            Det.SlogdetResult slog = Det.slogdet(m3D);
            assertEquals(-1.0, slog.sign());
            assertEquals(Math.log(306.0), slog.logAbsDet(), 1e-9);
        }
    }

    @Test
    @DisplayName("Singular matrix yields det = 0")
    void testSingularMatrix() {
        try (Arena arena = Arena.ofConfined()) {
            // Linearly dependent rows: row1 = 2 * row0
            NDArray singularF = TestArrayFactory.matrix(new float[][]{
                {1f, 2f, 3f},
                {2f, 4f, 6f},
                {7f, 8f, 9f}
            });
            assertEquals(0.0f, Det.detFloat(singularF), 1e-6f);

            NDArray singularD = TestArrayFactory.matrix(new double[][]{
                {1.0, 2.0, 3.0},
                {2.0, 4.0, 6.0},
                {7.0, 8.0, 9.0}
            });
            assertEquals(0.0, Det.detDouble(singularD), 1e-12);
            Det.SlogdetResult res = Det.slogdet(singularD);
            assertEquals(0.0, res.sign());
            assertEquals(Double.NEGATIVE_INFINITY, res.logAbsDet());
        }
    }

    @Test
    @DisplayName("Identity matrix yields det = 1")
    void testIdentityMatrix() {
        NDArray eyeF = TestArrayFactory.eye(16, DType.f32);
        assertEquals(1.0f, Det.detFloat(eyeF), 1e-5f);

        NDArray eyeD = TestArrayFactory.eye(16, DType.f64);
        assertEquals(1.0, Det.detDouble(eyeD), 1e-12);
    }

    @Test
    @DisplayName("Non-contiguous transposed view preserves determinant")
    void testNonContiguousTransposed() {
        NDArray m = TestArrayFactory.matrix(new double[][]{
            {2.0, 1.0, 3.0},
            {0.0, 4.0, 5.0},
            {1.0, 1.0, 2.0}
        });
        double detOrig = Det.detDouble(m);
        double detT = Det.detDouble(m.transpose());
        assertEquals(detOrig, detT, 1e-9);
    }
}
