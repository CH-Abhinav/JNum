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
import jnum.testutil.TestAssertions;

public class ArithmeticOpsTest {

    @Test
    @DisplayName("Private constructor throws AssertionError")
    void testPrivateConstructor() throws Exception {
        Constructor<ArithmeticOps> constructor = ArithmeticOps.class.getDeclaredConstructor();
        constructor.setAccessible(true);
        InvocationTargetException ex = assertThrows(InvocationTargetException.class, constructor::newInstance);
        assertInstanceOf(AssertionError.class, ex.getCause());
    }

    @Test
    @DisplayName("Type promotion: i32 + f32 -> f32, f32 + f64 -> f64")
    void testTypePromotion() {
        NDArray iArr = JNum.from(new int[]{1, 2}, 2);
        NDArray fArr = JNum.from(new float[]{1.5f, 2.5f}, 2);
        NDArray res1 = ArithmeticOps.add(iArr, fArr);
        assertEquals(DType.f32, res1.getDType());
        assertEquals(2.5f, res1.getFloat(0), 1e-6f);
        assertEquals(4.5f, res1.getFloat(1), 1e-6f);

        NDArray dArr = JNum.from(new double[]{0.5, 1.5}, 2);
        NDArray res2 = ArithmeticOps.mul(fArr, dArr);
        assertEquals(DType.f64, res2.getDType());
        assertEquals(0.75, res2.getDouble(0), 1e-12);
        assertEquals(3.75, res2.getDouble(1), 1e-12);
    }

    @Test
    @DisplayName("Broadcasting shapes: (2, 1) and (1, 3) -> (2, 3)")
    void testBroadcasting() {
        NDArray a = JNum.from(new float[]{1f, 2f}, 2, 1);
        NDArray b = JNum.from(new float[]{10f, 20f, 30f}, 1, 3);
        NDArray res = ArithmeticOps.add(a, b);

        assertArrayEquals(new long[]{2, 3}, res.getShape());
        assertEquals(11f, res.getFloat(0, 0), 1e-6f);
        assertEquals(21f, res.getFloat(0, 1), 1e-6f);
        assertEquals(31f, res.getFloat(0, 2), 1e-6f);
        assertEquals(12f, res.getFloat(1, 0), 1e-6f);
        assertEquals(22f, res.getFloat(1, 1), 1e-6f);
        assertEquals(32f, res.getFloat(1, 2), 1e-6f);
    }

    @Test
    @DisplayName("Incompatible shapes throw IllegalArgumentException")
    void testIncompatibleShapes() {
        NDArray a = JNum.zeros(DType.f32, 2, 3);
        NDArray b = JNum.zeros(DType.f32, 2, 4);
        assertThrows(IllegalArgumentException.class, () -> ArithmeticOps.add(a, b));
    }

    @Test
    @DisplayName("Array-scalar operations: float, int, double")
    void testArrayScalar() {
        NDArray a = JNum.from(new float[]{10f, 20f}, 2);
        NDArray r1 = ArithmeticOps.add(a, 5f);
        assertEquals(15f, r1.getFloat(0), 1e-6f);

        NDArray r2 = ArithmeticOps.sub(a, 2);
        assertEquals(8f, r2.getFloat(0), 1e-6f);

        NDArray r3 = ArithmeticOps.mul(a, 2.5);
        assertEquals(DType.f64, r3.getDType());
        assertEquals(25.0, r3.getDouble(0), 1e-12);

        NDArray r4 = ArithmeticOps.div(a, 2f);
        assertEquals(5f, r4.getFloat(0), 1e-6f);
    }

    @Test
    @DisplayName("Explicit result array overloads")
    void testResultArrayOverloads() {
        NDArray a = JNum.from(new float[]{2f, 4f}, 2);
        NDArray b = JNum.from(new float[]{3f, 5f}, 2);
        NDArray res = JNum.zeros(DType.f32, 2);
        ArithmeticOps.add(a, b, res);
        assertEquals(5f, res.getFloat(0), 1e-6f);
        assertEquals(9f, res.getFloat(1), 1e-6f);

        ArithmeticOps.sub(a, 1f, res);
        assertEquals(1f, res.getFloat(0), 1e-6f);
        assertEquals(3f, res.getFloat(1), 1e-6f);
    }

    @Test
    @DisplayName("Unsupported dtypes throw UnsupportedOperationException")
    void testUnsupportedDType() {
        NDArray boolArr = JNum.from(new boolean[]{true, false}, 2);
        assertThrows(UnsupportedOperationException.class, () -> ArithmeticOps.add(boolArr, boolArr));
    }
}
