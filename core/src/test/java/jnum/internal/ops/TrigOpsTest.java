package jnum.internal.ops;

import static org.junit.jupiter.api.Assertions.*;

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import jnum.DType;
import jnum.JNum;
import jnum.NDArray;

public class TrigOpsTest {

    @Test
    @DisplayName("Private constructor throws AssertionError")
    void testPrivateConstructor() throws Exception {
        Constructor<TrigOps> constructor = TrigOps.class.getDeclaredConstructor();
        constructor.setAccessible(true);
        InvocationTargetException ex = assertThrows(InvocationTargetException.class, constructor::newInstance);
        assertInstanceOf(AssertionError.class, ex.getCause());
    }

    @Test
    @DisplayName("Trig functions for float, double, int: sin, cos, tan, sinh, cosh, tanh")
    void testTrigFunctions() {
        // Float 0.0
        NDArray fZero = JNum.from(new float[]{0.0f}, 1);
        assertEquals(0.0f, TrigOps.sin(fZero).getFloat(0), 1e-6f);
        assertEquals(1.0f, TrigOps.cos(fZero).getFloat(0), 1e-6f);
        assertEquals(0.0f, TrigOps.tan(fZero).getFloat(0), 1e-6f);
        assertEquals(0.0f, TrigOps.sinh(fZero).getFloat(0), 1e-6f);
        assertEquals(1.0f, TrigOps.cosh(fZero).getFloat(0), 1e-6f);
        assertEquals(0.0f, TrigOps.tanh(fZero).getFloat(0), 1e-6f);

        // Double
        NDArray dArr = JNum.from(new double[]{Math.PI / 4.0}, 1);
        assertEquals(Math.sin(Math.PI / 4.0), TrigOps.sin(dArr).getDouble(0), 1e-6);
        assertEquals(Math.cos(Math.PI / 4.0), TrigOps.cos(dArr).getDouble(0), 1e-6);
        assertEquals(Math.tan(Math.PI / 4.0), TrigOps.tan(dArr).getDouble(0), 1e-6);

        // Int input promotes to Float output
        NDArray iArr = JNum.from(new int[]{0}, 1);
        NDArray sinInt = TrigOps.sin(iArr);
        assertEquals(DType.f32, sinInt.getDType());
        assertEquals(0.0f, sinInt.getFloat(0), 1e-6f);
    }

    @Test
    @DisplayName("Non-contiguous input view handling")
    void testNonContiguous() {
        NDArray m = JNum.from(new float[]{0f, 1f, 2f, 3f}, 2, 2);
        NDArray col = m.slice(":, 1:2");
        assertFalse(col.isContiguous());
        NDArray sinCol = TrigOps.sin(col);
        assertEquals((float) Math.sin(1.0), sinCol.getFloat(0, 0), 1e-5f);
        assertEquals((float) Math.sin(3.0), sinCol.getFloat(1, 0), 1e-5f);
    }

    @Test
    @DisplayName("Unsupported dtype (bool) throws UnsupportedOperationException")
    void testUnsupportedDType() {
        NDArray bArr = JNum.from(new boolean[]{true, false}, 2);
        assertThrows(UnsupportedOperationException.class, () -> TrigOps.sin(bArr));
    }
}
