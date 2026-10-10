package jnum.internal.ops;

import static org.junit.jupiter.api.Assertions.*;

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import jnum.DType;
import jnum.JNum;
import jnum.NDArray;
import jnum.testutil.TestArrayFactory;

public class ReduceOpsTest {

    @Test
    @DisplayName("Private constructor throws AssertionError")
    void testPrivateConstructor() throws Exception {
        Constructor<ReduceOps> constructor = ReduceOps.class.getDeclaredConstructor();
        constructor.setAccessible(true);
        InvocationTargetException ex = assertThrows(InvocationTargetException.class, constructor::newInstance);
        assertInstanceOf(AssertionError.class, ex.getCause());
    }

    @Test
    @DisplayName("Global reductions: sum, max, min for f32, f64, i32")
    void testGlobalReductions() {
        NDArray fArr = JNum.from(new float[]{1f, 5f, 2f, 4f}, 4);
        assertEquals(12.0, ReduceOps.sum(fArr), 1e-6);
        assertEquals(5.0, ReduceOps.max(fArr), 1e-6);
        assertEquals(1.0, ReduceOps.min(fArr), 1e-6);

        NDArray dArr = JNum.from(new double[]{10.0, -2.5, 3.5}, 3);
        assertEquals(11.0, ReduceOps.sum(dArr), 1e-12);
        assertEquals(10.0, ReduceOps.max(dArr), 1e-12);
        assertEquals(-2.5, ReduceOps.min(dArr), 1e-12);

        NDArray iArr = JNum.from(new int[]{7, 12, -3}, 3);
        assertEquals(16.0, ReduceOps.sum(iArr), 1e-6);
        assertEquals(12.0, ReduceOps.max(iArr), 1e-6);
        assertEquals(-3.0, ReduceOps.min(iArr), 1e-6);
    }

    @Test
    @DisplayName("Axis reductions: sum, max, min along axis 0 and axis 1")
    void testAxisReductions() {
        NDArray m = TestArrayFactory.matrix(new float[][]{
            {1f, 2f, 3f},
            {4f, 5f, 6f}
        });

        // Sum axis 0 -> [5, 7, 9]
        NDArray sum0 = ReduceOps.sum(m, 0);
        assertArrayEquals(new long[]{3}, sum0.getShape());
        assertEquals(5f, sum0.getFloat(0), 1e-6f);
        assertEquals(7f, sum0.getFloat(1), 1e-6f);
        assertEquals(9f, sum0.getFloat(2), 1e-6f);

        // Sum axis 1 -> [6, 15]
        NDArray sum1 = ReduceOps.sum(m, 1);
        assertArrayEquals(new long[]{2}, sum1.getShape());
        assertEquals(6f, sum1.getFloat(0), 1e-6f);
        assertEquals(15f, sum1.getFloat(1), 1e-6f);

        // Max axis 0 -> [4, 5, 6]
        NDArray max0 = ReduceOps.max(m, 0);
        assertEquals(4f, max0.getFloat(0), 1e-6f);
        assertEquals(6f, max0.getFloat(2), 1e-6f);

        // Min axis 1 -> [1, 4]
        NDArray min1 = ReduceOps.min(m, 1);
        assertEquals(1f, min1.getFloat(0), 1e-6f);
        assertEquals(4f, min1.getFloat(1), 1e-6f);
    }

    @Test
    @DisplayName("Dot product with promotion: i32 . f32 -> f64/f32")
    void testDotProduct() {
        NDArray a = JNum.from(new int[]{1, 2, 3}, 3);
        NDArray b = JNum.from(new float[]{4f, 5f, 6f}, 3);
        double dotVal = ReduceOps.dot(a, b);
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assertEquals(32.0, dotVal, 1e-6);
    }

    @Test
    @DisplayName("Unsupported dtypes throw UnsupportedOperationException")
    void testUnsupportedDType() {
        NDArray bArr = JNum.from(new boolean[]{true, false}, 2);
        assertThrows(UnsupportedOperationException.class, () -> ReduceOps.sum(bArr));
        assertThrows(UnsupportedOperationException.class, () -> ReduceOps.max(bArr));
        assertThrows(UnsupportedOperationException.class, () -> ReduceOps.min(bArr));
    }
}
