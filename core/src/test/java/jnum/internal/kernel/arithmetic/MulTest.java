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

@DisplayName("Mul Kernel - SIMD, Strided & Boundary Tests")
class MulTest {

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
        Constructor<Mul> constructor = Mul.class.getDeclaredConstructor();
        constructor.setAccessible(true);
        InvocationTargetException ex = assertThrows(InvocationTargetException.class, constructor::newInstance);
        assertInstanceOf(AssertionError.class, ex.getCause());
    }

    @ParameterizedTest(name = "mulFloat contiguous lane size: {0}")
    @MethodSource("laneSizesF32")
    void mulFloatContiguousLaneSizes(long size) {
        NDArray a = TestArrayFactory.random(701L, DType.f32, size);
        NDArray b = TestArrayFactory.random(702L, DType.f32, size);
        NDArray res = JNum.zeros(DType.f32, size);

        Mul.mulFloat(a, b, res);

        for (int i = 0; i < size; i++) {
            assertEquals(a.getFloat(i) * b.getFloat(i), res.getFloat(i), 1e-5f);
        }
    }

    @ParameterizedTest(name = "mulDouble contiguous lane size: {0}")
    @MethodSource("laneSizesF64")
    void mulDoubleContiguousLaneSizes(long size) {
        NDArray a = TestArrayFactory.random(801L, DType.f64, size);
        NDArray b = TestArrayFactory.random(802L, DType.f64, size);
        NDArray res = JNum.zeros(DType.f64, size);

        Mul.mulDouble(a, b, res);

        for (int i = 0; i < size; i++) {
            assertEquals(a.getDouble(i) * b.getDouble(i), res.getDouble(i), 1e-12);
        }
    }

    @ParameterizedTest(name = "mulInt contiguous lane size: {0}")
    @MethodSource("laneSizesI32")
    void mulIntContiguousLaneSizes(long size) {
        NDArray a = TestArrayFactory.random(901L, DType.i32, size);
        NDArray b = TestArrayFactory.random(902L, DType.i32, size);
        NDArray res = JNum.zeros(DType.i32, size);

        Mul.mulInt(a, b, res);

        for (int i = 0; i < size; i++) {
            assertEquals(a.getInt(i) * b.getInt(i), res.getInt(i));
        }
    }

    @Test
    @DisplayName("mulFloat scalar multiplication produces expected results")
    void mulScalars() {
        NDArray a = JNum.from(new float[]{2f, 4f, 6f}, 3);
        NDArray res = JNum.zeros(3);
        Mul.mulFloat(a, 3f, res);
        assertEquals(6f, res.getFloat(0));
        assertEquals(18f, res.getFloat(2));
    }

    @Test
    @DisplayName("mulFloat works on non-contiguous transposed views")
    void mulNonContiguous() {
        NDArray a = JNum.from(new float[]{1f, 2f, 3f, 4f}, 2, 2).transpose();
        NDArray b = JNum.from(new float[]{2f, 3f, 4f, 5f}, 2, 2).transpose();
        NDArray res = JNum.zeros(2, 2);

        Mul.mulFloat(a, b, res);

        assertEquals(2f, res.getFloat(0, 0));  // 1 * 2
        assertEquals(12f, res.getFloat(0, 1)); // 3 * 4
        assertEquals(6f, res.getFloat(1, 0));  // 2 * 3
        assertEquals(20f, res.getFloat(1, 1)); // 4 * 5
    }

    @Test
    @DisplayName("mulFloat supports in-place aliasing (res == a)")
    void inPlaceAliasing() {
        NDArray a = JNum.from(new float[]{2f, 3f, 4f}, 3);
        NDArray b = JNum.from(new float[]{5f, 6f, 7f}, 3);

        Mul.mulFloat(a, b, a);
        assertEquals(10f, a.getFloat(0));
        assertEquals(18f, a.getFloat(1));
        assertEquals(28f, a.getFloat(2));
    }
}
