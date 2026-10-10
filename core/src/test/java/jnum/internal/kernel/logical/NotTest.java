package jnum.internal.kernel.logical;

import jnum.DType;
import jnum.JNum;
import jnum.NDArray;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("Not Kernel - Elementwise Boolean Not Tests")
class NotTest {

    @Test
    @DisplayName("Private constructor throws AssertionError upon reflection")
    void constructorThrows() throws Exception {
        Constructor<Not> constructor = Not.class.getDeclaredConstructor();
        constructor.setAccessible(true);
        InvocationTargetException ex = assertThrows(InvocationTargetException.class, constructor::newInstance);
        assertInstanceOf(AssertionError.class, ex.getCause());
    }

    @Test
    @DisplayName("not inverts boolean elements")
    void notInversion() {
        NDArray a = JNum.from(new boolean[]{true, false, true, false}, 4);
        NDArray res = JNum.zeros(DType.bool, 4);

        Not.not(a, res);

        assertFalse(res.getBoolean(0));
        assertTrue(res.getBoolean(1));
        assertFalse(res.getBoolean(2));
        assertTrue(res.getBoolean(3));
    }

    @Test
    @DisplayName("not works on non-contiguous transposed views")
    void notTransposed() {
        NDArray a = JNum.from(new boolean[]{true, false, false, true}, 2, 2).transpose();
        NDArray res = JNum.zeros(DType.bool, 2, 2);

        Not.not(a, res);

        assertFalse(res.getBoolean(0, 0));
        assertTrue(res.getBoolean(0, 1));
        assertTrue(res.getBoolean(1, 0));
        assertFalse(res.getBoolean(1, 1));
    }
}
