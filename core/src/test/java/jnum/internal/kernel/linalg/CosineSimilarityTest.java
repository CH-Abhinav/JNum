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

public class CosineSimilarityTest {

    @Test
    @DisplayName("Private constructor throws AssertionError")
    void testPrivateConstructor() throws Exception {
        Constructor<CosineSimilarity> constructor = CosineSimilarity.class.getDeclaredConstructor();
        constructor.setAccessible(true);
        InvocationTargetException ex = assertThrows(InvocationTargetException.class, constructor::newInstance);
        assertInstanceOf(AssertionError.class, ex.getCause());
    }

    @Test
    @DisplayName("Invalid dimensions or length mismatch throws IllegalArgumentException")
    void testInvalidDimensions() {
        try (Arena arena = Arena.ofConfined()) {
            NDArray m2D = JNum.zeros(arena, DType.f32, 2, 2);
            NDArray v1D = JNum.zeros(arena, DType.f32, 4);
            assertThrows(IllegalArgumentException.class, () -> CosineSimilarity.compute(m2D, v1D));

            NDArray vMismatch = JNum.zeros(arena, DType.f32, 5);
            assertThrows(IllegalArgumentException.class, () -> CosineSimilarity.compute(v1D, vMismatch));
        }
    }

    @Test
    @DisplayName("Length 0 and zero vectors return 0.0")
    void testBoundaryVectors() {
        try (Arena arena = Arena.ofConfined()) {
            NDArray emptyA = JNum.zeros(arena, DType.f32, 0);
            NDArray emptyB = JNum.zeros(arena, DType.f32, 0);
            assertEquals(0.0, CosineSimilarity.compute(emptyA, emptyB));

            NDArray zeroA = JNum.zeros(arena, DType.f32, 4);
            NDArray nonzeroB = JNum.from(new float[]{1f, 2f, 3f, 4f}, 4);
            assertEquals(0.0, CosineSimilarity.compute(zeroA, nonzeroB));
        }
    }

    @Test
    @DisplayName("Collinear, opposite, and orthogonal vectors")
    void testGeometricRelations() {
        // Collinear (same direction) -> 1.0
        NDArray a = JNum.from(new float[]{1f, 2f, 3f}, 3);
        NDArray b = JNum.from(new float[]{2f, 4f, 6f}, 3);
        assertEquals(1.0, CosineSimilarity.compute(a, b), 1e-6);

        // Opposite direction -> -1.0
        NDArray c = JNum.from(new float[]{-1f, -2f, -3f}, 3);
        assertEquals(-1.0, CosineSimilarity.compute(a, c), 1e-6);

        // Orthogonal -> 0.0
        NDArray d = JNum.from(new float[]{1f, 0f, 0f}, 3);
        NDArray e = JNum.from(new float[]{0f, 1f, 0f}, 3);
        assertEquals(0.0, CosineSimilarity.compute(d, e), 1e-6);

        // 45 degrees: [1, 0] and [1, 1] -> cos(45) = 1 / sqrt(2)
        NDArray v1 = JNum.from(new double[]{1.0, 0.0}, 2);
        NDArray v2 = JNum.from(new double[]{1.0, 1.0}, 2);
        assertEquals(1.0 / Math.sqrt(2.0), CosineSimilarity.compute(v1, v2), 1e-12);
    }

    @Test
    @DisplayName("SIMD lane loop bounds and non-contiguous views")
    void testSIMDAndNonContiguous() {
        for (int len : new int[]{1, 7, 8, 9, 16, 17, 32, 33}) {
            float[] dataA = new float[len];
            float[] dataB = new float[len];
            for (int i = 0; i < len; i++) {
                dataA[i] = (i + 1) * 0.5f;
                dataB[i] = (i + 1) * 0.5f;
            }
            NDArray arrA = JNum.from(dataA, len);
            NDArray arrB = JNum.from(dataB, len);
            assertEquals(1.0, CosineSimilarity.compute(arrA, arrB), 1e-5);
        }

        // Sliced non-contiguous 1D views
        NDArray matrix = TestArrayFactory.matrix(new double[][]{
            {1.0, 2.0},
            {2.0, 4.0}
        });
        NDArray row0 = matrix.subview(0); // [1, 2]
        NDArray row1 = matrix.subview(1); // [2, 4]
        assertEquals(1.0, CosineSimilarity.compute(row0, row1), 1e-12);
    }
}
