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
import jnum.testutil.ReferenceOps;
import jnum.testutil.TestArrayFactory;
import jnum.testutil.TestAssertions;

public class MatMulTest {

    @Test
    @DisplayName("Private constructor throws AssertionError")
    void testPrivateConstructor() throws Exception {
        Constructor<MatMul> constructor = MatMul.class.getDeclaredConstructor();
        constructor.setAccessible(true);
        InvocationTargetException ex = assertThrows(InvocationTargetException.class, constructor::newInstance);
        assertInstanceOf(AssertionError.class, ex.getCause());
    }

    @Test
    @DisplayName("Nano kernel (maxDim <= 4): Float, Double, Int")
    void testNanoKernel() {
        try (Arena arena = Arena.ofConfined()) {
            // 2x3 and 3x2
            NDArray aF = TestArrayFactory.matrix(new float[][]{{1, 2, 3}, {4, 5, 6}});
            NDArray bF = TestArrayFactory.matrix(new float[][]{{7, 8}, {9, 1}, {2, 3}});
            NDArray cF = JNum.zeros(arena, DType.f32, 2, 2);
            MatMul.matmulFloat(aF, bF, cF);
            // [1*7+2*9+3*2, 1*8+2*1+3*3] = [31, 19]
            // [4*7+5*9+6*2, 4*8+5*1+6*3] = [85, 55]
            assertEquals(31.0f, cF.getFloat(0, 0), 1e-5f);
            assertEquals(19.0f, cF.getFloat(0, 1), 1e-5f);
            assertEquals(85.0f, cF.getFloat(1, 0), 1e-5f);
            assertEquals(55.0f, cF.getFloat(1, 1), 1e-5f);

            // Double nano (1x1)
            NDArray aD = TestArrayFactory.matrix(new double[][]{{3.5}});
            NDArray bD = TestArrayFactory.matrix(new double[][]{{2.0}});
            NDArray cD = JNum.zeros(arena, DType.f64, 1, 1);
            MatMul.matmulDouble(aD, bD, cD);
            assertEquals(7.0, cD.getDouble(0, 0), 1e-12);

            // Int nano (3x3 identity)
            NDArray aI = TestArrayFactory.eye(3, DType.i32);
            NDArray bI = TestArrayFactory.matrix(new int[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
            NDArray cI = JNum.zeros(arena, DType.i32, 3, 3);
            MatMul.matmulInt(aI, bI, cI);
            assertArrayEquals(new int[]{1, 2, 3}, new int[]{cI.getInt(0, 0), cI.getInt(0, 1), cI.getInt(0, 2)});
            assertArrayEquals(new int[]{4, 5, 6}, new int[]{cI.getInt(1, 0), cI.getInt(1, 1), cI.getInt(1, 2)});
            assertArrayEquals(new int[]{7, 8, 9}, new int[]{cI.getInt(2, 0), cI.getInt(2, 1), cI.getInt(2, 2)});
        }
    }

    @Test
    @DisplayName("Direct tiled kernel (4 < maxDim <= 128): Float, Double, Int")
    void testDirectTiledKernel() {
        try (Arena arena = Arena.ofConfined()) {
            int m = 16, k = 24, n = 12;
            NDArray aF = TestArrayFactory.random(42L, DType.f32, m, k);
            NDArray bF = TestArrayFactory.random(43L, DType.f32, k, n);
            NDArray cF = JNum.zeros(arena, DType.f32, m, n);
            MatMul.matmulFloat(aF, bF, cF);

            float[] refF = ReferenceOps.matmulFloat(TestArrayFactory.toFloatArray(aF), TestArrayFactory.toFloatArray(bF), m, k, n);
            TestAssertions.assertArrayClose(refF, TestArrayFactory.toFloatArray(cF), 1e-4f, 1e-4f);

            // Double
            NDArray aD = TestArrayFactory.random(44L, DType.f64, m, k);
            NDArray bD = TestArrayFactory.random(45L, DType.f64, k, n);
            NDArray cD = JNum.zeros(arena, DType.f64, m, n);
            MatMul.matmulDouble(aD, bD, cD);

            double[] refD = ReferenceOps.matmulDouble(TestArrayFactory.toDoubleArray(aD), TestArrayFactory.toDoubleArray(bD), m, k, n);
            TestAssertions.assertArrayClose(refD, TestArrayFactory.toDoubleArray(cD), 1e-9, 1e-9);

            // Int
            NDArray aI = JNum.zeros(DType.i32, m, k);
            NDArray bI = JNum.zeros(DType.i32, k, n);
            for (int r = 0; r < m; r++) for (int c = 0; c < k; c++) aI.setInt(r % 5, r, c);
            for (int r = 0; r < k; r++) for (int c = 0; c < n; c++) bI.setInt(c % 3, r, c);
            NDArray cI = JNum.zeros(arena, DType.i32, m, n);
            MatMul.matmulInt(aI, bI, cI);

            int expected00 = 0;
            for (int x = 0; x < k; x++) expected00 += aI.getInt(0, x) * bI.getInt(x, 0);
            assertEquals(expected00, cI.getInt(0, 0));
        }
    }

    @Test
    @DisplayName("Single-threaded BLIS tier (128 < maxDim <= 256)")
    void testBlisSingleThreadTier() {
        try (Arena arena = Arena.ofConfined()) {
            int dim = 130;
            NDArray aF = TestArrayFactory.random(101L, DType.f32, dim, dim);
            NDArray bF = TestArrayFactory.eye(dim, DType.f32);
            NDArray cF = JNum.zeros(arena, DType.f32, dim, dim);
            MatMul.matmulFloat(aF, bF, cF);
            TestAssertions.assertNDArrayClose(aF, cF, 1e-5, 1e-5);

            // Double
            NDArray aD = TestArrayFactory.random(102L, DType.f64, dim, dim);
            NDArray bD = TestArrayFactory.eye(dim, DType.f64);
            NDArray cD = JNum.zeros(arena, DType.f64, dim, dim);
            MatMul.matmulDouble(aD, bD, cD);
            TestAssertions.assertNDArrayClose(aD, cD, 1e-11, 1e-11);
        }
    }

    @Test
    @DisplayName("Multi-threaded BLIS tier (maxDim > 256)")
    void testBlisMacroParallelTier() {
        try (Arena arena = Arena.ofShared()) {
            int dim = 260;
            NDArray aF = TestArrayFactory.random(201L, DType.f32, dim, dim);
            NDArray bF = TestArrayFactory.eye(dim, DType.f32);
            NDArray cF = JNum.zeros(arena, DType.f32, dim, dim);
            MatMul.matmulFloat(aF, bF, cF);
            TestAssertions.assertNDArrayClose(aF, cF, 1e-5, 1e-5);

            NDArray aD = TestArrayFactory.random(202L, DType.f64, dim, dim);
            NDArray bD = TestArrayFactory.eye(dim, DType.f64);
            NDArray cD = JNum.zeros(arena, DType.f64, dim, dim);
            MatMul.matmulDouble(aD, bD, cD);
            TestAssertions.assertNDArrayClose(aD, cD, 1e-11, 1e-11);
        }
    }

    @Test
    @DisplayName("Non-contiguous views: Transposed and sliced")
    void testNonContiguousViews() {
        try (Arena arena = Arena.ofConfined()) {
            NDArray origA = TestArrayFactory.random(301L, DType.f32, 10, 8);
            NDArray aT = origA.transpose(); // shape: (8, 10), non-contiguous
            NDArray b = TestArrayFactory.random(302L, DType.f32, 10, 6);
            NDArray c = JNum.zeros(arena, DType.f32, 8, 6);

            MatMul.matmulFloat(aT, b, c);

            // Reference check: A_T is (8, 10), B is (10, 6)
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 6; j++) {
                    float expected = 0.0f;
                    for (int k = 0; k < 10; k++) {
                        expected += origA.getFloat(k, i) * b.getFloat(k, j);
                    }
                    assertEquals(expected, c.getFloat(i, j), 1e-4f);
                }
            }
        }
    }
}
