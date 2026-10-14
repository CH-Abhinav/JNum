package jnum.internal.kernel.linalg.matmul;

import static org.junit.jupiter.api.Assertions.*;

import java.lang.foreign.Arena;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import jnum.DType;
import jnum.JNum;
import jnum.NDArray;
import jnum.testutil.ReferenceOps;
import jnum.testutil.TestArrayFactory;
import jnum.testutil.TestAssertions;

public class MatMulTier1DirectTiledTest {

    @Test
    @DisplayName("Private constructor throws AssertionError")
    void testPrivateConstructor() throws Exception {
        Constructor<MatMulTier1DirectTiled> constructor = MatMulTier1DirectTiled.class.getDeclaredConstructor();
        constructor.setAccessible(true);
        InvocationTargetException ex = assertThrows(InvocationTargetException.class, constructor::newInstance);
        assertInstanceOf(AssertionError.class, ex.getCause());
    }

    @Test
    @DisplayName("Tier 1 Direct Tiled Float: Multiple shapes with tails")
    void testDirectTiledFloat() {
        try (Arena arena = Arena.ofConfined()) {
            int[][] shapes = {{16, 24, 12}, {33, 31, 29}, {64, 64, 64}};
            long seed = 100L;
            for (int[] s : shapes) {
                int m = s[0], k = s[1], n = s[2];
                NDArray a = TestArrayFactory.random(seed++, DType.f32, m, k);
                NDArray b = TestArrayFactory.random(seed++, DType.f32, k, n);
                NDArray c = JNum.zeros(arena, DType.f32, m, n);
                MatMulTier1DirectTiled.matmulFloat(a, b, c, m, k, n);

                float[] ref = ReferenceOps.matmulFloat(TestArrayFactory.toFloatArray(a), TestArrayFactory.toFloatArray(b), m, k, n);
                TestAssertions.assertArrayClose(ref, TestArrayFactory.toFloatArray(c), 1e-4f, 1e-4f);
            }
        }
    }

    @Test
    @DisplayName("Tier 1 Direct Tiled Double: Multiple shapes with tails")
    void testDirectTiledDouble() {
        try (Arena arena = Arena.ofConfined()) {
            int[][] shapes = {{16, 24, 12}, {27, 19, 25}, {48, 32, 40}};
            long seed = 200L;
            for (int[] s : shapes) {
                int m = s[0], k = s[1], n = s[2];
                NDArray a = TestArrayFactory.random(seed++, DType.f64, m, k);
                NDArray b = TestArrayFactory.random(seed++, DType.f64, k, n);
                NDArray c = JNum.zeros(arena, DType.f64, m, n);
                MatMulTier1DirectTiled.matmulDouble(a, b, c, m, k, n);

                double[] ref = ReferenceOps.matmulDouble(TestArrayFactory.toDoubleArray(a), TestArrayFactory.toDoubleArray(b), m, k, n);
                TestAssertions.assertArrayClose(ref, TestArrayFactory.toDoubleArray(c), 1e-9, 1e-9);
            }
        }
    }

    @Test
    @DisplayName("Tier 1 Direct Tiled Int: Small and medium matrices")
    void testDirectTiledInt() {
        try (Arena arena = Arena.ofConfined()) {
            int m = 18, k = 22, n = 15;
            NDArray a = JNum.zeros(DType.i32, m, k);
            NDArray b = JNum.zeros(DType.i32, k, n);
            for (int r = 0; r < m; r++) for (int c = 0; c < k; c++) a.setInt((r + c) % 7, r, c);
            for (int r = 0; r < k; r++) for (int c = 0; c < n; c++) b.setInt((r * 2 + c) % 5, r, c);
            NDArray c = JNum.zeros(arena, DType.i32, m, n);
            MatMulTier1DirectTiled.matmulInt(a, b, c, m, k, n);

            for (int i = 0; i < m; i += 5) {
                for (int j = 0; j < n; j += 4) {
                    int expected = 0;
                    for (int x = 0; x < k; x++) expected += a.getInt(i, x) * b.getInt(x, j);
                    assertEquals(expected, c.getInt(i, j));
                }
            }
        }
    }

    @Test
    @DisplayName("Tier 1 Direct Tiled Strided: Transposed A and B fallback")
    void testDirectTiledStrided() {
        try (Arena arena = Arena.ofConfined()) {
            NDArray origA = TestArrayFactory.random(300L, DType.f32, 20, 15);
            NDArray origB = TestArrayFactory.random(301L, DType.f32, 10, 20);
            NDArray a = origA.transpose(); // 15x20
            NDArray b = origB.transpose(); // 20x10, non-contiguous b_s1 != 1
            NDArray c = JNum.zeros(arena, DType.f32, 15, 10);
            MatMulTier1DirectTiled.matmulFloat(a, b, c, 15, 20, 10);

            for (int i = 0; i < 15; i++) {
                for (int j = 0; j < 10; j++) {
                    float expected = 0f;
                    for (int k = 0; k < 20; k++) {
                        expected += a.getFloat(i, k) * b.getFloat(k, j);
                    }
                    assertEquals(expected, c.getFloat(i, j), 1e-4f);
                }
            }
        }
    }
}
