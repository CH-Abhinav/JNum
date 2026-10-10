package jnum.internal.kernel.logical;

import jnum.DType;
import jnum.JNum;
import jnum.NDArray;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("Any Kernel - Boolean Reduction Any Tests")
class AnyTest {

    @Test
    @DisplayName("Private constructor throws AssertionError upon reflection")
    void constructorThrows() throws Exception {
        Constructor<Any> constructor = Any.class.getDeclaredConstructor();
        constructor.setAccessible(true);
        InvocationTargetException ex = assertThrows(InvocationTargetException.class, constructor::newInstance);
        assertInstanceOf(AssertionError.class, ex.getCause());
    }

    @Test
    @DisplayName("any returns true when at least one element is true")
    void anyTrue() {
        NDArray a = JNum.from(new boolean[]{false, false, true, false}, 4);
        assertTrue(Any.any(a));

        NDArray b = JNum.from(new boolean[]{true, false, false, false}, 4);
        assertTrue(Any.any(b));
    }

    @Test
    @DisplayName("any returns false when all elements are false")
    void anyFalse() {
        NDArray a = JNum.from(new boolean[]{false, false, false, false}, 4);
        assertFalse(Any.any(a));
    }

    @Test
    @DisplayName("any works on non-contiguous transposed views")
    void anyTransposed() {
        NDArray a = JNum.from(new boolean[]{false, false, false, true}, 2, 2).transpose();
        assertTrue(Any.any(a));

        NDArray b = JNum.from(new boolean[]{false, false, false, false}, 2, 2).transpose();
        assertFalse(Any.any(b));
    }

    @Test
    @DisplayName("any returns false on empty array")
    void anyEmpty() {
        NDArray empty = JNum.zeros(DType.bool, 0);
        assertFalse(Any.any(empty));
    }
}
