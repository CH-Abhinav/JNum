package jnum.internal.ops;

import static org.junit.jupiter.api.Assertions.*;

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import jnum.DType;
import jnum.JNum;
import jnum.NDArray;

public class BooleanOpsTest {

    @Test
    @DisplayName("Private constructor throws AssertionError")
    void testPrivateConstructor() throws Exception {
        Constructor<BooleanOps> constructor = BooleanOps.class.getDeclaredConstructor();
        constructor.setAccessible(true);
        InvocationTargetException ex = assertThrows(InvocationTargetException.class, constructor::newInstance);
        assertInstanceOf(AssertionError.class, ex.getCause());
    }

    @Test
    @DisplayName("Boolean and, or, xor, not basic truth table")
    void testBasicTruthTable() {
        NDArray a = JNum.from(new boolean[]{true, true, false, false}, 4);
        NDArray b = JNum.from(new boolean[]{true, false, true, false}, 4);

        NDArray andRes = BooleanOps.and(a, b);
        assertArrayEquals(new boolean[]{true, false, false, false},
                new boolean[]{andRes.getBoolean(0), andRes.getBoolean(1), andRes.getBoolean(2), andRes.getBoolean(3)});

        NDArray orRes = BooleanOps.or(a, b);
        assertArrayEquals(new boolean[]{true, true, true, false},
                new boolean[]{orRes.getBoolean(0), orRes.getBoolean(1), orRes.getBoolean(2), orRes.getBoolean(3)});

        NDArray xorRes = BooleanOps.xor(a, b);
        assertArrayEquals(new boolean[]{false, true, true, false},
                new boolean[]{xorRes.getBoolean(0), xorRes.getBoolean(1), xorRes.getBoolean(2), xorRes.getBoolean(3)});

        NDArray notRes = BooleanOps.not(a);
        assertArrayEquals(new boolean[]{false, false, true, true},
                new boolean[]{notRes.getBoolean(0), notRes.getBoolean(1), notRes.getBoolean(2), notRes.getBoolean(3)});
    }

    @Test
    @DisplayName("Broadcasting in boolean operations: (2, 1) and (1, 2)")
    void testBroadcasting() {
        NDArray a = JNum.from(new boolean[]{true, false}, 2, 1);
        NDArray b = JNum.from(new boolean[]{true, false}, 1, 2);

        NDArray res = BooleanOps.and(a, b);
        assertArrayEquals(new long[]{2, 2}, res.getShape());
        assertTrue(res.getBoolean(0, 0));
        assertFalse(res.getBoolean(0, 1));
        assertFalse(res.getBoolean(1, 0));
        assertFalse(res.getBoolean(1, 1));
    }

    @Test
    @DisplayName("Automatic casting of numeric arrays to boolean")
    void testNumericCasting() {
        NDArray numsA = JNum.from(new int[]{0, 1, 2}, 3); // [false, true, true]
        NDArray numsB = JNum.from(new float[]{0f, 0f, 5f}, 3); // [false, false, true]

        NDArray res = BooleanOps.or(numsA, numsB);
        assertEquals(DType.bool, res.getDType());
        assertFalse(res.getBoolean(0));
        assertTrue(res.getBoolean(1));
        assertTrue(res.getBoolean(2));
    }

    @Test
    @DisplayName("any and all reductions")
    void testAnyAndAll() {
        NDArray allTrue = JNum.from(new boolean[]{true, true, true}, 3);
        assertTrue(BooleanOps.all(allTrue));
        assertTrue(BooleanOps.any(allTrue));

        NDArray mixed = JNum.from(new boolean[]{true, false, true}, 3);
        assertFalse(BooleanOps.all(mixed));
        assertTrue(BooleanOps.any(mixed));

        NDArray allFalse = JNum.from(new boolean[]{false, false, false}, 3);
        assertFalse(BooleanOps.all(allFalse));
        assertFalse(BooleanOps.any(allFalse));
    }

    @Test
    @DisplayName("In-place result array overloads")
    void testResultOverloads() {
        NDArray a = JNum.from(new boolean[]{true, false}, 2);
        NDArray b = JNum.from(new boolean[]{true, true}, 2);
        NDArray res = JNum.zeros(DType.bool, 2);

        BooleanOps.and(a, b, res);
        assertTrue(res.getBoolean(0));
        assertFalse(res.getBoolean(1));

        BooleanOps.not(a, res);
        assertFalse(res.getBoolean(0));
        assertTrue(res.getBoolean(1));
    }
}
