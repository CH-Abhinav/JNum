package jnum.internal.ops;

import static org.junit.jupiter.api.Assertions.*;

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import jnum.DType;
import jnum.JNum;
import jnum.NDArray;

public class SetOpsTest {

    @Test
    @DisplayName("Private constructor throws AssertionError")
    void testPrivateConstructor() throws Exception {
        Constructor<SetOps> constructor = SetOps.class.getDeclaredConstructor();
        constructor.setAccessible(true);
        InvocationTargetException ex = assertThrows(InvocationTargetException.class, constructor::newInstance);
        assertInstanceOf(AssertionError.class, ex.getCause());
    }

    @Test
    @DisplayName("Set by indices across dtypes: f32, f64, i32, bool")
    void testSetAcrossDTypes() {
        NDArray fArr = JNum.zeros(DType.f32, 2, 2);
        SetOps.set(fArr, 3.14, 0, 1);
        assertEquals(3.14f, fArr.getFloat(0, 1), 1e-6f);

        NDArray dArr = JNum.zeros(DType.f64, 2, 2);
        SetOps.set(dArr, 2.718281828, 1, 0);
        assertEquals(2.718281828, dArr.getDouble(1, 0), 1e-12);

        NDArray iArr = JNum.zeros(DType.i32, 2, 2);
        SetOps.set(iArr, 42, 1, 1);
        assertEquals(42, iArr.getInt(1, 1));

        NDArray bArr = JNum.zeros(DType.bool, 2, 2);
        SetOps.set(bArr, 1.0, 0, 0);
        assertTrue(bArr.getBoolean(0, 0));
        assertFalse(bArr.getBoolean(0, 1));
    }

    @Test
    @DisplayName("setFloat, setDouble, setInt, setBoolean 1D/2D/3D with negative indices")
    void testTypedSetters() {
        NDArray a1D = JNum.zeros(DType.f32, 5);
        SetOps.setFloat(a1D, 99.0f, -1); // last element
        assertEquals(99.0f, a1D.getFloat(4), 1e-6f);

        NDArray a2D = JNum.zeros(DType.f64, 3, 3);
        SetOps.setDouble(a2D, 123.456, -1, -1); // bottom-right element
        assertEquals(123.456, a2D.getDouble(2, 2), 1e-12);

        NDArray a3D = JNum.zeros(DType.i32, 2, 2, 2);
        SetOps.setInt(a3D, 777, 1, 0, 1);
        assertEquals(777, a3D.getInt(1, 0, 1));

        NDArray b2D = JNum.zeros(DType.bool, 2, 2);
        SetOps.setBoolean(b2D, true, 0, 1);
        assertTrue(b2D.getBoolean(0, 1));
    }

    @Test
    @DisplayName("Invalid indices count or out of bounds throws exceptions")
    void testBoundsExceptions() {
        NDArray m = JNum.zeros(DType.f32, 2, 2);
        assertThrows(IllegalArgumentException.class, () -> SetOps.set(m, 1.0, 0)); // 1 index for 2D
        assertThrows(IndexOutOfBoundsException.class, () -> SetOps.set(m, 1.0, 5, 0)); // index 5 out of bounds
        assertThrows(IndexOutOfBoundsException.class, () -> SetOps.setFloat(m, 1.0f, 10, 0));
    }
}
