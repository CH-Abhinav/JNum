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

public class MatMulTier2BlisSingleTest {

    @Test
    @DisplayName("Private constructor throws AssertionError")
    void testPrivateConstructor() throws Exception {
        Constructor<MatMulTier2BlisSingle> constructor = MatMulTier2BlisSingle.class.getDeclaredConstructor();
        constructor.setAccessible(true);
        InvocationTargetException ex = assertThrows(InvocationTargetException.class, constructor::newInstance);
        assertInstanceOf(AssertionError.class, ex.getCause());
    }

    @Test
    @DisplayName("Tier 2 BLIS Single-Threaded Float: 130x130 identity and random")
    void testBlisSingleFloat() {
        try (Arena arena = Arena.ofConfined()) {
            int dim = 130;
            NDArray a = TestArrayFactory.random(401L, DType.f32, dim, dim);
            NDArray eye = TestArrayFactory.eye(dim, DType.f32);
            NDArray c = JNum.zeros(arena, DType.f32, dim, dim);
            MatMulTier2BlisSingle.matmulFloat(a, eye, c, dim, dim, dim);
            TestAssertions.assertNDArrayClose(a, c, 1e-5, 1e-5);

            // Rectangular with tails
            int m = 135, k = 140, n = 130;
            NDArray aRect = TestArrayFactory.random(402L, DType.f32, m, k);
            NDArray bRect = TestArrayFactory.random(403L, DType.f32, k, n);
            NDArray cRect = JNum.zeros(arena, DType.f32, m, n);
            MatMulTier2BlisSingle.matmulFloat(aRect, bRect, cRect, m, k, n);

            float[] ref = ReferenceOps.matmulFloat(TestArrayFactory.toFloatArray(aRect), TestArrayFactory.toFloatArray(bRect), m, k, n);
            TestAssertions.assertArrayClose(ref, TestArrayFactory.toFloatArray(cRect), 1e-4f, 1e-4f);
        }
    }

    @Test
    @DisplayName("Tier 2 BLIS Single-Threaded Double: 130x130 identity and rectangular")
    void testBlisSingleDouble() {
        try (Arena arena = Arena.ofConfined()) {
            int dim = 130;
            NDArray a = TestArrayFactory.random(501L, DType.f64, dim, dim);
            NDArray eye = TestArrayFactory.eye(dim, DType.f64);
            NDArray c = JNum.zeros(arena, DType.f64, dim, dim);
            MatMulTier2BlisSingle.matmulDouble(a, eye, c, dim, dim, dim);
            TestAssertions.assertNDArrayClose(a, c, 1e-11, 1e-11);

            int m = 132, k = 135, n = 130;
            NDArray aRect = TestArrayFactory.random(502L, DType.f64, m, k);
            NDArray bRect = TestArrayFactory.random(503L, DType.f64, k, n);
            NDArray cRect = JNum.zeros(arena, DType.f64, m, n);
            MatMulTier2BlisSingle.matmulDouble(aRect, bRect, cRect, m, k, n);

            double[] ref = ReferenceOps.matmulDouble(TestArrayFactory.toDoubleArray(aRect), TestArrayFactory.toDoubleArray(bRect), m, k, n);
            TestAssertions.assertArrayClose(ref, TestArrayFactory.toDoubleArray(cRect), 1e-9, 1e-9);
        }
    }

    @Test
    @DisplayName("Tier 2 BLIS Single-Threaded Int: Identity matrix")
    void testBlisSingleInt() {
        try (Arena arena = Arena.ofConfined()) {
            int dim = 130;
            NDArray a = JNum.zeros(DType.i32, dim, dim);
            for (int r = 0; r < dim; r++) {
                for (int c = 0; c < dim; c++) {
                    a.setInt((r + c) % 10, r, c);
                }
            }
            NDArray eye = TestArrayFactory.eye(dim, DType.i32);
            NDArray c = JNum.zeros(arena, DType.i32, dim, dim);
            MatMulTier2BlisSingle.matmulInt(a, eye, c, dim, dim, dim);

            for (int r = 0; r < dim; r += 13) {
                for (int col = 0; col < dim; col += 17) {
                    assertEquals(a.getInt(r, col), c.getInt(r, col));
                }
            }
        }
    }
}
