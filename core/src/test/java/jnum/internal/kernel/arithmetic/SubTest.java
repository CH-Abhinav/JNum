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

@DisplayName("Sub Kernel - SIMD, Strided & Boundary Tests")
class SubTest {

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
        Constructor<Sub> constructor = Sub.class.getDeclaredConstructor();
        constructor.setAccessible(true);
        InvocationTargetException ex = assertThrows(InvocationTargetException.class, constructor::newInstance);
        assertInstanceOf(AssertionError.class, ex.getCause());
    }

    @ParameterizedTest(name = "subFloat contiguous lane size: {0}")
    @MethodSource("laneSizesF32")
    void subFloatContiguousLaneSizes(long size) {
        NDArray a = TestArrayFactory.random(111L, DType.f32, size);
        NDArray b = TestArrayFactory.random(222L, DType.f32, size);
        NDArray res = JNum.zeros(DType.f32, size);

        Sub.subFloat(a, b, res);

        for (int i = 0; i < size; i++) {
            assertEquals(a.getFloat(i) - b.getFloat(i), res.getFloat(i), 1e-6f);
        }
    }

    @ParameterizedTest(name = "subDouble contiguous lane size: {0}")
    @MethodSource("laneSizesF64")
    void subDoubleContiguousLaneSizes(long size) {
        NDArray a = TestArrayFactory.random(333L, DType.f64, size);
        NDArray b = TestArrayFactory.random(444L, DType.f64, size);
        NDArray res = JNum.zeros(DType.f64, size);

        Sub.subDouble(a, b, res);

        for (int i = 0; i < size; i++) {
            assertEquals(a.getDouble(i) - b.getDouble(i), res.getDouble(i), 1e-12);
        }
    }

    @ParameterizedTest(name = "subInt contiguous lane size: {0}")
    @MethodSource("laneSizesI32")
    void subIntContiguousLaneSizes(long size) {
        NDArray a = TestArrayFactory.random(555L, DType.i32, size);
        NDArray b = TestArrayFactory.random(666L, DType.i32, size);
        NDArray res = JNum.zeros(DType.i32, size);

        Sub.subInt(a, b, res);

        for (int i = 0; i < size; i++) {
            assertEquals(a.getInt(i) - b.getInt(i), res.getInt(i));
        }
    }

    @Test
    @DisplayName("subFloat, subDouble, subInt scalar subtraction produce exact results")
    void subScalars() {
        NDArray aF = JNum.from(new float[]{10f, 20f, 30f}, 3);
        NDArray resF = JNum.zeros(DType.f32, 3);
        Sub.subFloat(aF, 2f, resF);
        assertEquals(8f, resF.getFloat(0));
        assertEquals(28f, resF.getFloat(2));

        NDArray aD = JNum.from(new double[]{10.0, 20.0, 30.0}, 3);
        NDArray resD = JNum.zeros(DType.f64, 3);
        Sub.subDouble(aD, 5.0, resD);
        assertEquals(5.0, resD.getDouble(0));
        assertEquals(25.0, resD.getDouble(2));

        NDArray aI = JNum.from(new int[]{10, 20, 30}, 3);
        NDArray resI = JNum.zeros(DType.i32, 3);
        Sub.subInt(aI, 3, resI);
        assertEquals(7, resI.getInt(0));
        assertEquals(27, resI.getInt(2));
    }

    @Test
    @DisplayName("subFloat works on non-contiguous transposed views")
    void subNonContiguousStrided() {
        NDArray a = JNum.from(new float[]{10f, 20f, 30f, 40f}, 2, 2).transpose();
        NDArray b = JNum.from(new float[]{1f, 2f, 3f, 4f}, 2, 2).transpose();
        NDArray res = JNum.zeros(2, 2);

        Sub.subFloat(a, b, res);

        assertEquals(9f, res.getFloat(0, 0));  // 10 - 1
        assertEquals(27f, res.getFloat(0, 1)); // 30 - 3
        assertEquals(18f, res.getFloat(1, 0)); // 20 - 2
        assertEquals(36f, res.getFloat(1, 1)); // 40 - 4
    }

    @Test
    @DisplayName("subFloat supports in-place aliasing (res == a)")
    void inPlaceAliasing() {
        NDArray a = JNum.from(new float[]{10f, 20f, 30f}, 3);
        NDArray b = JNum.from(new float[]{1f, 2f, 3f}, 3);

        Sub.subFloat(a, b, a);
        assertEquals(9f, a.getFloat(0));
        assertEquals(18f, a.getFloat(1));
        assertEquals(27f, a.getFloat(2));
    }
}
