package jnum;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.lang.foreign.Arena;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("JNum - Facade, Factories, Random & Arange Tests")
class JNumTest {

    // =========================================================================
    // 01. Construction & Reflection Protection
    // =========================================================================
    @Test
    @DisplayName("Private constructor throws AssertionError upon reflection")
    void constructorCannotBeInstantiated() throws Exception {
        Constructor<JNum> constructor = JNum.class.getDeclaredConstructor();
        constructor.setAccessible(true);
        InvocationTargetException ex = assertThrows(InvocationTargetException.class, constructor::newInstance);
        assertInstanceOf(AssertionError.class, ex.getCause());
    }

    // =========================================================================
    // 02. Zeros & Ones Across All DTypes
    // =========================================================================
    @Test
    @DisplayName("zeros creates correctly sized and zero-initialized arrays")
    void zerosAllDtypes() {
        NDArray zF32 = JNum.zeros(2, 3);
        assertEquals(DType.f32, zF32.getDType());
        assertEquals(0.0f, zF32.getFloat(0, 0));

        NDArray zF64 = JNum.zeros(DType.f64, 4);
        assertEquals(0.0, zF64.getDouble(0));

        NDArray zI32 = JNum.zeros(DType.i32, 2, 2);
        assertEquals(0, zI32.getInt(1, 1));

        NDArray zBool = JNum.zeros(DType.bool, 3);
        assertFalse(zBool.getBoolean(0));
    }

    @Test
    @DisplayName("ones creates correctly initialized unity arrays across all dtypes")
    void onesAllDtypes() {
        NDArray oF32 = JNum.ones(2, 2);
        assertEquals(1.0f, oF32.getFloat(0, 0));

        NDArray oF64 = JNum.ones(DType.f64, 2);
        assertEquals(1.0, oF64.getDouble(0));

        NDArray oI32 = JNum.ones(DType.i32, 2);
        assertEquals(1, oI32.getInt(0));

        NDArray oBool = JNum.ones(DType.bool, 2);
        assertTrue(oBool.getBoolean(0));
    }

    // =========================================================================
    // 03. From Array Factories & Size Mismatch
    // =========================================================================
    @Test
    @DisplayName("from creates NDArray from primitive arrays and validates size")
    void fromPrimitives() {
        NDArray f = JNum.from(new float[]{1f, 2f, 3f, 4f}, 2, 2);
        assertEquals(1f, f.getFloat(0, 0));
        assertEquals(4f, f.getFloat(1, 1));

        NDArray d = JNum.from(new double[]{5.0, 6.0}, 2);
        assertEquals(5.0, d.getDouble(0));

        NDArray i = JNum.from(new int[]{10, 20}, 2);
        assertEquals(10, i.getInt(0));

        NDArray b = JNum.from(new boolean[]{true, false}, 2);
        assertTrue(b.getBoolean(0));
        assertFalse(b.getBoolean(1));

        // Mismatched shape size vs array length
        assertThrows(IllegalArgumentException.class, () -> JNum.from(new float[]{1f, 2f}, 3));
        assertThrows(IllegalArgumentException.class, () -> JNum.from(new double[]{1.0, 2.0}, 3));
        assertThrows(IllegalArgumentException.class, () -> JNum.from(new int[]{1, 2}, 3));
        assertThrows(IllegalArgumentException.class, () -> JNum.from(new boolean[]{true, false}, 3));
    }

    // =========================================================================
    // 04. Arange Generation & Boundaries
    // =========================================================================
    @Test
    @DisplayName("arange creates 1D sequences correctly")
    void arangeSequence() {
        NDArray a1 = JNum.arange(5.0);
        assertEquals(5, a1.getSize());
        assertEquals(0.0f, a1.getFloat(0));
        assertEquals(4.0f, a1.getFloat(4));

        NDArray a2 = JNum.arange(2.0, 5.0);
        assertEquals(3, a2.getSize());
        assertEquals(2.0f, a2.getFloat(0));
        assertEquals(4.0f, a2.getFloat(2));

        NDArray a3 = JNum.arange(1.0, 10.0, 2.0);
        assertEquals(5, a3.getSize()); // 1, 3, 5, 7, 9
        assertEquals(1.0f, a3.getFloat(0));
        assertEquals(9.0f, a3.getFloat(4));
    }

    @Test
    @DisplayName("arange throws on zero step and unsupported boolean dtype")
    void arangeExceptions() {
        assertThrows(IllegalArgumentException.class, () -> JNum.arange(0.0, 10.0, 0.0));
        assertThrows(IllegalArgumentException.class,
            () -> JNum.arange(0.0, 5.0, 1.0, DType.bool, Arena.ofAuto()));
    }

    @Test
    @DisplayName("arange returns empty array when bounds are reversed with positive step")
    void arangeEmpty() {
        NDArray empty = JNum.arange(10.0, 5.0, 1.0);
        assertEquals(0, empty.getSize());
    }

    // =========================================================================
    // 05. Random Tensor Generation
    // =========================================================================
    @Test
    @DisplayName("rand generates values within expected bounds")
    void randGeneration() {
        NDArray r1 = JNum.rand(10L);
        assertEquals(10L, r1.getSize());

        NDArray r2 = JNum.rand(DType.i32, 5L);
        assertEquals(5L, r2.getSize());

        NDArray rBounded = JNum.rand(10, 20, DType.i32, 10L);
        for (int idx = 0; idx < 10; idx++) {
            int val = rBounded.getInt(idx);
            assertTrue(val >= 10 && val < 20);
        }
    }

    @Test
    @DisplayName("rand throws when generating invalid types into mismatching dtype")
    void randExceptions() {
        assertThrows(IllegalArgumentException.class,
            () -> JNum.rand(5.0f, DType.i32, 2L));
        assertThrows(IllegalArgumentException.class,
            () -> JNum.rand(5.0, DType.f32, 2L));
    }

    // =========================================================================
    // 06. Eval Facade Checks
    // =========================================================================
    @Test
    @DisplayName("eval evaluates expressions and rejects empty arguments")
    void evalFacade() {
        NDArray x = JNum.from(new float[]{2f, 3f}, 2);
        NDArray y = JNum.from(new float[]{4f, 5f}, 2);

        NDArray resPositional = JNum.eval("$0 + $1", x, y);
        assertEquals(6f, resPositional.getFloat(0));
        assertEquals(8f, resPositional.getFloat(1));

        NDArray resNamed = JNum.eval("a * b", Map.of("a", x, "b", y));
        assertEquals(8f, resNamed.getFloat(0));
        assertEquals(15f, resNamed.getFloat(1));

        assertThrows(IllegalArgumentException.class, () -> JNum.eval("x + 1"));
        assertThrows(IllegalArgumentException.class, () -> JNum.eval("x + 1", Map.of()));
    }
}
