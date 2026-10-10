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

public class NormTest {

    @Test
    @DisplayName("Private constructor throws AssertionError")
    void testPrivateConstructor() throws Exception {
        Constructor<Norm> constructor = Norm.class.getDeclaredConstructor();
        constructor.setAccessible(true);
        InvocationTargetException ex = assertThrows(InvocationTargetException.class, constructor::newInstance);
        assertInstanceOf(AssertionError.class, ex.getCause());
    }

    @Test
    @DisplayName("Unsupported norm order throws UnsupportedOperationException")
    void testUnsupportedOrder() {
        try (Arena arena = Arena.ofConfined()) {
            NDArray a = JNum.from(new float[]{1f, 2f, 3f}, 3);
            assertThrows(UnsupportedOperationException.class, () -> Norm.norm(a, 3));
            assertThrows(UnsupportedOperationException.class, () -> Norm.normFloat(a, 0));
            assertThrows(UnsupportedOperationException.class, () -> Norm.normDouble(a.cast(DType.f64), -1));
        }
    }

    @Test
    @DisplayName("Empty array norm is 0.0")
    void testEmptyArray() {
        try (Arena arena = Arena.ofConfined()) {
            NDArray emptyF = JNum.zeros(arena, DType.f32, 0);
            assertEquals(0.0f, Norm.normFloat(emptyF, 2));
            assertEquals(0.0f, Norm.normFloat(emptyF, 1));
            assertEquals(0.0f, Norm.normFloat(emptyF, Integer.MAX_VALUE));

            NDArray emptyD = JNum.zeros(arena, DType.f64, 0);
            assertEquals(0.0, Norm.normDouble(emptyD, 2));
        }
    }

    @Test
    @DisplayName("L1, L2, Linf norms on contiguous arrays: Float and Double")
    void testContiguousNorms() {
        // [3, -4] -> L1 = 7, L2 = 5, Linf = 4
        NDArray vF = JNum.from(new float[]{3.0f, -4.0f}, 2);
        assertEquals(7.0f, Norm.normFloat(vF, 1), 1e-6f);
        assertEquals(5.0f, Norm.normFloat(vF, 2), 1e-6f);
        assertEquals(4.0f, Norm.normFloat(vF, Integer.MAX_VALUE), 1e-6f);

        NDArray vD = JNum.from(new double[]{-1.5, 2.0, -2.5}, 3);
        // L1 = 1.5 + 2.0 + 2.5 = 6.0
        // L2 = sqrt(2.25 + 4 + 6.25) = sqrt(12.5) = 3.5355339...
        // Linf = 2.5
        assertEquals(6.0, Norm.normDouble(vD, 1), 1e-12);
        assertEquals(Math.sqrt(12.5), Norm.normDouble(vD, 2), 1e-12);
        assertEquals(2.5, Norm.normDouble(vD, Integer.MAX_VALUE), 1e-12);
    }

    @Test
    @DisplayName("SIMD boundary lane sizes (unrolled loop tails)")
    void testLaneBoundaries() {
        for (int size : new int[]{1, 7, 8, 9, 31, 32, 33, 64, 65}) {
            NDArray arrF = TestArrayFactory.random(1000L + size, DType.f32, size);
            float[] rawF = TestArrayFactory.toFloatArray(arrF);

            float expectedL1 = 0.0f;
            float expectedL2Sq = 0.0f;
            float expectedLinf = 0.0f;
            for (float v : rawF) {
                float abs = Math.abs(v);
                expectedL1 += abs;
                expectedL2Sq += v * v;
                expectedLinf = Math.max(expectedLinf, abs);
            }

            assertEquals(expectedL1, Norm.normFloat(arrF, 1), 1e-3f * size);
            assertEquals((float) Math.sqrt(expectedL2Sq), Norm.normFloat(arrF, 2), 1e-3f * size);
            assertEquals(expectedLinf, Norm.normFloat(arrF, Integer.MAX_VALUE), 1e-4f);
        }
    }

    @Test
    @DisplayName("Non-contiguous views norm (NDIter path)")
    void testNonContiguousNorm() {
        NDArray m = TestArrayFactory.matrix(new float[][]{
            {1f, -2f, 3f},
            {-4f, 5f, -6f}
        });
        NDArray col1 = m.slice(":, 1:2"); // [-2, 5], non-contiguous
        assertFalse(col1.isContiguous());

        assertEquals(7.0f, Norm.normFloat(col1, 1), 1e-6f);
        assertEquals((float) Math.sqrt(4 + 25), Norm.normFloat(col1, 2), 1e-5f);
        assertEquals(5.0f, Norm.normFloat(col1, Integer.MAX_VALUE), 1e-6f);

        // Double
        NDArray mD = TestArrayFactory.matrix(new double[][]{
            {1.0, -2.0, 3.0},
            {-4.0, 5.0, -6.0}
        });
        NDArray col1D = mD.slice(":, 1:2");
        assertEquals(7.0, Norm.normDouble(col1D, 1), 1e-12);
        assertEquals(Math.sqrt(29.0), Norm.normDouble(col1D, 2), 1e-12);
        assertEquals(5.0, Norm.normDouble(col1D, Integer.MAX_VALUE), 1e-12);
    }
}
