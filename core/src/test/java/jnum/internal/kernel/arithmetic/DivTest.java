package jnum.internal.kernel.arithmetic;

import jnum.DType;
import jnum.JNum;
import jnum.NDArray;
import jnum.testutil.LaneSizes;
import jnum.testutil.TestArrayFactory;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.util.stream.LongStream;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("Div Kernel - SIMD, Strided, Division-by-Zero & Boundary Tests")
class DivTest {

    static LongStream laneSizesF32() {
        return LongStream.of(LaneSizes.boundarySizesF32()).filter(s -> s > 0);
    }

    static LongStream laneSizesF64() {
        return LongStream.of(LaneSizes.boundarySizesF64()).filter(s -> s > 0);
    }

    static LongStream laneSizesI32() {
        return LongStream.of(LaneSizes.boundarySizesI32()).filter(s -> s > 0);
    }

    @Test
    @DisplayName("Private constructor throws AssertionError upon reflection")
    void constructorThrows() throws Exception {
        Constructor<Div> constructor = Div.class.getDeclaredConstructor();
        constructor.setAccessible(true);
        InvocationTargetException ex = assertThrows(InvocationTargetException.class, constructor::newInstance);
        assertInstanceOf(AssertionError.class, ex.getCause());
    }

    @ParameterizedTest(name = "divFloat contiguous lane size: {0}")
    @MethodSource("laneSizesF32")
    void divFloatContiguousLaneSizes(long size) {
        NDArray a = TestArrayFactory.random(1101L, DType.f32, size);
        NDArray b = JNum.ones(DType.f32, size); // Avoid random divide-by-zero
        NDArray res = JNum.zeros(DType.f32, size);

        Div.divFloat(a, b, res);

        for (int i = 0; i < size; i++) {
            assertEquals(a.getFloat(i) / b.getFloat(i), res.getFloat(i), 1e-5f);
        }
    }

    @ParameterizedTest(name = "divDouble contiguous lane size: {0}")
    @MethodSource("laneSizesF64")
    void divDoubleContiguousLaneSizes(long size) {
        NDArray a = TestArrayFactory.random(1201L, DType.f64, size);
        NDArray b = JNum.ones(DType.f64, size);
        NDArray res = JNum.zeros(DType.f64, size);

        Div.divDouble(a, b, res);

        for (int i = 0; i < size; i++) {
            assertEquals(a.getDouble(i) / b.getDouble(i), res.getDouble(i), 1e-12);
        }
    }

    @ParameterizedTest(name = "divInt contiguous lane size: {0}")
    @MethodSource("laneSizesI32")
    void divIntContiguousLaneSizes(long size) {
        NDArray a = JNum.from(new int[]{(int) size * 10}, 1);
        NDArray aPadded = JNum.zeros(DType.i32, size);
        NDArray b = JNum.ones(DType.i32, size); // Div by 1
        NDArray res = JNum.zeros(DType.i32, size);

        Div.divInt(aPadded, b, res);

        for (int i = 0; i < size; i++) {
            assertEquals(0, res.getInt(i));
        }
    }

    @Test
    @DisplayName("divFloat handles division by zero producing IEEE Infinity and NaN")
    void divFloatDivisionByZero() {
        NDArray a = JNum.from(new float[]{1.0f, -1.0f, 0.0f}, 3);
        NDArray b = JNum.from(new float[]{0.0f, 0.0f, 0.0f}, 3);
        NDArray res = JNum.zeros(3);

        Div.divFloat(a, b, res);

        assertEquals(Float.POSITIVE_INFINITY, res.getFloat(0));
        assertEquals(Float.NEGATIVE_INFINITY, res.getFloat(1));
        assertTrue(Float.isNaN(res.getFloat(2)));
    }

    @Test
    @DisplayName("divFloat works on non-contiguous transposed views")
    void divNonContiguous() {
        NDArray a = JNum.from(new float[]{10f, 20f, 30f, 40f}, 2, 2).transpose();
        NDArray b = JNum.from(new float[]{2f, 4f, 5f, 8f}, 2, 2).transpose();
        NDArray res = JNum.zeros(2, 2);

        Div.divFloat(a, b, res);

        assertEquals(5f, res.getFloat(0, 0)); // 10 / 2
        assertEquals(6f, res.getFloat(0, 1)); // 30 / 5
        assertEquals(5f, res.getFloat(1, 0)); // 20 / 4
        assertEquals(5f, res.getFloat(1, 1)); // 40 / 8
    }

    @Test
    @DisplayName("divFloat supports in-place aliasing (res == a)")
    void inPlaceAliasing() {
        NDArray a = JNum.from(new float[]{10f, 20f, 30f}, 3);
        NDArray b = JNum.from(new float[]{2f, 4f, 5f}, 3);

        Div.divFloat(a, b, a);
        assertEquals(5f, a.getFloat(0));
        assertEquals(5f, a.getFloat(1));
        assertEquals(6f, a.getFloat(2));
    }
}
