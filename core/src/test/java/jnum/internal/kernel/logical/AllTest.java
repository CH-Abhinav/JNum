package jnum.internal.kernel.logical;

import jnum.DType;
import jnum.JNum;
import jnum.NDArray;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("All Kernel - Boolean Reduction All Tests")
class AllTest {

    @Test
    @DisplayName("Private constructor throws AssertionError upon reflection")
    void constructorThrows() throws Exception {
        Constructor<All> constructor = All.class.getDeclaredConstructor();
        constructor.setAccessible(true);
        InvocationTargetException ex = assertThrows(InvocationTargetException.class, constructor::newInstance);
        assertInstanceOf(AssertionError.class, ex.getCause());
    }

    @Test
    @DisplayName("all returns true when all elements are true")
    void allTrue() {
        NDArray a = JNum.from(new boolean[]{true, true, true, true}, 4);
        assertTrue(All.all(a));
    }

    @Test
    @DisplayName("all returns false when at least one element is false")
    void allWithFalse() {
        NDArray a = JNum.from(new boolean[]{true, false, true, true}, 4);
        assertFalse(All.all(a));

        NDArray b = JNum.from(new boolean[]{false, true, true, true}, 4);
        assertFalse(All.all(b));

        NDArray c = JNum.from(new boolean[]{true, true, true, false}, 4);
        assertFalse(All.all(c));
    }

    @Test
    @DisplayName("all works on non-contiguous transposed views")
    void allTransposed() {
        NDArray a = JNum.from(new boolean[]{true, true, false, true}, 2, 2).transpose();
        assertFalse(All.all(a));

        NDArray b = JNum.from(new boolean[]{true, true, true, true}, 2, 2).transpose();
        assertTrue(All.all(b));
    }

    @Test
    @DisplayName("all returns true on empty array (standard reduction identity)")
    void allEmpty() {
        NDArray empty = JNum.zeros(DType.bool, 0);
        assertTrue(All.all(empty));
    }
}
