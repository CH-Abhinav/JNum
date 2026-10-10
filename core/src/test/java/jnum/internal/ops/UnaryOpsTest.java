package jnum.internal.ops;

import static org.junit.jupiter.api.Assertions.*;

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import jnum.DType;
import jnum.JNum;
import jnum.NDArray;

public class UnaryOpsTest {

    @Test
    @DisplayName("Private constructor throws AssertionError")
    void testPrivateConstructor() throws Exception {
        Constructor<UnaryOps> constructor = UnaryOps.class.getDeclaredConstructor();
        constructor.setAccessible(true);
        InvocationTargetException ex = assertThrows(InvocationTargetException.class, constructor::newInstance);
        assertInstanceOf(AssertionError.class, ex.getCause());
    }

    @Test
    @DisplayName("Unary operations: sqrt, abs, exp, log, log10, sigmoid")
    void testUnaryOperations() {
        NDArray fArr = JNum.from(new float[]{4.0f, 100.0f}, 2);
        assertEquals(2.0f, UnaryOps.sqrt(fArr).getFloat(0), 1e-6f);
        assertEquals(2.0f, UnaryOps.log10(fArr).getFloat(1), 1e-6f);

        NDArray fArr2 = JNum.from(new float[]{-5.0f, 0.0f}, 2);
        assertEquals(5.0f, UnaryOps.abs(fArr2).getFloat(0), 1e-6f);
        assertEquals(1.0f, UnaryOps.exp(fArr2).getFloat(1), 1e-6f); // e^0 = 1
        assertEquals(0.5f, UnaryOps.sigmoid(fArr2).getFloat(1), 1e-6f); // sigmoid(0) = 0.5

        NDArray dArr = JNum.from(new double[]{Math.E}, 1);
        assertEquals(1.0, UnaryOps.log(dArr).getDouble(0), 1e-6);

        // Int input: abs keeps i32, others promote to f32
        NDArray iArr = JNum.from(new int[]{-42, 16}, 2);
        NDArray absInt = UnaryOps.abs(iArr);
        assertEquals(DType.i32, absInt.getDType());
        assertEquals(42, absInt.getInt(0));

        NDArray sqrtInt = UnaryOps.sqrt(iArr);
        assertEquals(DType.f32, sqrtInt.getDType());
        assertEquals(4.0f, sqrtInt.getFloat(1), 1e-6f);
    }

    @Test
    @DisplayName("Non-contiguous input views")
    void testNonContiguous() {
        NDArray m = JNum.from(new float[]{1f, -4f, 9f, -16f}, 2, 2);
        NDArray col = m.slice(":, 1:2"); // [-4, -16]
        NDArray absCol = UnaryOps.abs(col);
        assertEquals(4.0f, absCol.getFloat(0, 0), 1e-6f);
        assertEquals(16.0f, absCol.getFloat(1, 0), 1e-6f);
    }

    @Test
    @DisplayName("Unsupported dtype (bool) throws UnsupportedOperationException")
    void testUnsupportedDType() {
        NDArray bArr = JNum.from(new boolean[]{true, false}, 2);
        assertThrows(UnsupportedOperationException.class, () -> UnaryOps.sqrt(bArr));
    }
}
