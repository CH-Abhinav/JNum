package jnum.internal.ops;

import static org.junit.jupiter.api.Assertions.*;

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import jnum.DType;
import jnum.JNum;
import jnum.NDArray;

public class CompareOpsTest {

    @Test
    @DisplayName("Private constructor throws AssertionError")
    void testPrivateConstructor() throws Exception {
        Constructor<CompareOps> constructor = CompareOps.class.getDeclaredConstructor();
        constructor.setAccessible(true);
        InvocationTargetException ex = assertThrows(InvocationTargetException.class, constructor::newInstance);
        assertInstanceOf(AssertionError.class, ex.getCause());
    }

    @Test
    @DisplayName("Maximum and minimum between arrays with type promotion and broadcasting")
    void testArrayArrayComparisons() {
        NDArray a = JNum.from(new int[]{1, 10}, 2, 1);
        NDArray b = JNum.from(new float[]{5f, 2f}, 1, 2);

        NDArray maxRes = CompareOps.maximum(a, b);
        assertEquals(DType.f32, maxRes.getDType());
        assertArrayEquals(new long[]{2, 2}, maxRes.getShape());
        assertEquals(5f, maxRes.getFloat(0, 0), 1e-6f);
        assertEquals(2f, maxRes.getFloat(0, 1), 1e-6f);
        assertEquals(10f, maxRes.getFloat(1, 0), 1e-6f);
        assertEquals(10f, maxRes.getFloat(1, 1), 1e-6f);

        NDArray minRes = CompareOps.minimum(a, b);
        assertEquals(DType.f32, minRes.getDType());
        assertEquals(1f, minRes.getFloat(0, 0), 1e-6f);
        assertEquals(1f, minRes.getFloat(0, 1), 1e-6f);
        assertEquals(5f, minRes.getFloat(1, 0), 1e-6f);
        assertEquals(2f, minRes.getFloat(1, 1), 1e-6f);
    }

    @Test
    @DisplayName("Scalar comparisons: float, int, double")
    void testScalarComparisons() {
        NDArray a = JNum.from(new float[]{2f, 8f}, 2);
        NDArray maxF = CompareOps.maximum(a, 5f);
        assertEquals(5f, maxF.getFloat(0), 1e-6f);
        assertEquals(8f, maxF.getFloat(1), 1e-6f);

        NDArray minF = CompareOps.minimum(a, 5f);
        assertEquals(2f, minF.getFloat(0), 1e-6f);
        assertEquals(5f, minF.getFloat(1), 1e-6f);

        NDArray aInt = JNum.from(new int[]{3, 7}, 2);
        NDArray maxI = CompareOps.maximum(aInt, 5);
        assertEquals(5, maxI.getInt(0));
        assertEquals(7, maxI.getInt(1));

        NDArray aD = JNum.from(new double[]{1.5, 4.5}, 2);
        NDArray maxD = CompareOps.maximum(aD, 3.0);
        assertEquals(3.0, maxD.getDouble(0), 1e-12);
        assertEquals(4.5, maxD.getDouble(1), 1e-12);
    }

    @Test
    @DisplayName("Unsupported dtype (bool) throws UnsupportedOperationException")
    void testUnsupportedDType() {
        NDArray bArr = JNum.from(new boolean[]{true, false}, 2);
        assertThrows(UnsupportedOperationException.class, () -> CompareOps.maximum(bArr, bArr));
        assertThrows(UnsupportedOperationException.class, () -> CompareOps.minimum(bArr, bArr));
    }
}
