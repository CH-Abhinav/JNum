package jnum.internal.kernel.logical;

import jnum.DType;
import jnum.JNum;
import jnum.NDArray;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("And Kernel - Elementwise Boolean And Tests")
class AndTest {

    @Test
    @DisplayName("Private constructor throws AssertionError upon reflection")
    void constructorThrows() throws Exception {
        Constructor<And> constructor = And.class.getDeclaredConstructor();
        constructor.setAccessible(true);
        InvocationTargetException ex = assertThrows(InvocationTargetException.class, constructor::newInstance);
        assertInstanceOf(AssertionError.class, ex.getCause());
    }

    @Test
    @DisplayName("and satisfies standard boolean truth table")
    void andTruthTable() {
        NDArray a = JNum.from(new boolean[]{true, true, false, false}, 4);
        NDArray b = JNum.from(new boolean[]{true, false, true, false}, 4);
        NDArray res = JNum.zeros(DType.bool, 4);

        And.and(a, b, res);

        assertTrue(res.getBoolean(0));
        assertFalse(res.getBoolean(1));
        assertFalse(res.getBoolean(2));
        assertFalse(res.getBoolean(3));
    }

    @Test
    @DisplayName("and works on non-contiguous transposed views")
    void andTransposed() {
        NDArray a = JNum.from(new boolean[]{true, false, true, false}, 2, 2).transpose();
        NDArray b = JNum.from(new boolean[]{true, true, false, false}, 2, 2).transpose();
        NDArray res = JNum.zeros(DType.bool, 2, 2);

        And.and(a, b, res);

        assertTrue(res.getBoolean(0, 0));
        assertFalse(res.getBoolean(0, 1));
        assertFalse(res.getBoolean(1, 0));
        assertFalse(res.getBoolean(1, 1));
    }
}
