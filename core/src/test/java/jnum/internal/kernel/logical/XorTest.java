package jnum.internal.kernel.logical;

import jnum.DType;
import jnum.JNum;
import jnum.NDArray;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("Xor Kernel - Elementwise Boolean Xor Tests")
class XorTest {

    @Test
    @DisplayName("Private constructor throws AssertionError upon reflection")
    void constructorThrows() throws Exception {
        Constructor<Xor> constructor = Xor.class.getDeclaredConstructor();
        constructor.setAccessible(true);
        InvocationTargetException ex = assertThrows(InvocationTargetException.class, constructor::newInstance);
        assertInstanceOf(AssertionError.class, ex.getCause());
    }

    @Test
    @DisplayName("xor satisfies standard boolean truth table")
    void xorTruthTable() {
        NDArray a = JNum.from(new boolean[]{true, true, false, false}, 4);
        NDArray b = JNum.from(new boolean[]{true, false, true, false}, 4);
        NDArray res = JNum.zeros(DType.bool, 4);

        Xor.xor(a, b, res);

        assertFalse(res.getBoolean(0));
        assertTrue(res.getBoolean(1));
        assertTrue(res.getBoolean(2));
        assertFalse(res.getBoolean(3));
    }

    @Test
    @DisplayName("xor works on non-contiguous transposed views")
    void xorTransposed() {
        NDArray a = JNum.from(new boolean[]{true, false, false, true}, 2, 2).transpose();
        NDArray b = JNum.from(new boolean[]{true, true, false, false}, 2, 2).transpose();
        NDArray res = JNum.zeros(DType.bool, 2, 2);

        Xor.xor(a, b, res);

        assertFalse(res.getBoolean(0, 0));
        assertFalse(res.getBoolean(0, 1));
        assertTrue(res.getBoolean(1, 0));
        assertTrue(res.getBoolean(1, 1));
    }
}
