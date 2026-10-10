package jnum.internal.kernel.arithmetic;

import jnum.DType;
import jnum.JNum;
import jnum.NDArray;
import jnum.testutil.LaneSizes;
import jnum.testutil.ReferenceOps;
import jnum.testutil.TestArrayFactory;
import jnum.testutil.TestAssertions;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.util.stream.LongStream;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("Add Kernel - SIMD, Strided & Boundary Tests")
class AddTest {

    static LongStream laneSizesF32() {
        return LongStream.of(LaneSizes.boundarySizesF32()).filter(s -> s > 0);
    }

    static LongStream laneSizesF64() {
        return LongStream.of(LaneSizes.boundarySizesF64()).filter(s -> s > 0);
    }

    static LongStream laneSizesI32() {
        return LongStream.of(LaneSizes.boundarySizesI32()).filter(s -> s > 0);
    }

    // =========================================================================
    // 01. Private Constructor Check
    // =========================================================================
    @Test
    @DisplayName("Private constructor throws AssertionError upon reflection")
    void constructorThrows() throws Exception {
        Constructor<Add> constructor = Add.class.getDeclaredConstructor();
        constructor.setAccessible(true);
        InvocationTargetException ex = assertThrows(InvocationTargetException.class, constructor::newInstance);
        assertInstanceOf(AssertionError.class, ex.getCause());
    }

    // =========================================================================
    // 02. Vector Lane Boundaries (Float, Double, Int)
    // =========================================================================
    @ParameterizedTest(name = "addFloat contiguous lane size: {0}")
    @MethodSource("laneSizesF32")
    void addFloatContiguousLaneSizes(long size) {
        NDArray a = TestArrayFactory.random(101L, DType.f32, size);
        NDArray b = TestArrayFactory.random(202L, DType.f32, size);
        NDArray res = JNum.zeros(DType.f32, size);

        Add.addFloat(a, b, res);

        for (int i = 0; i < size; i++) {
            assertEquals(a.getFloat(i) + b.getFloat(i), res.getFloat(i), 1e-6f);
        }
    }

    @ParameterizedTest(name = "addDouble contiguous lane size: {0}")
    @MethodSource("laneSizesF64")
    void addDoubleContiguousLaneSizes(long size) {
        NDArray a = TestArrayFactory.random(303L, DType.f64, size);
        NDArray b = TestArrayFactory.random(404L, DType.f64, size);
        NDArray res = JNum.zeros(DType.f64, size);

        Add.addDouble(a, b, res);

        for (int i = 0; i < size; i++) {
            assertEquals(a.getDouble(i) + b.getDouble(i), res.getDouble(i), 1e-12);
        }
    }

    @ParameterizedTest(name = "addInt contiguous lane size: {0}")
    @MethodSource("laneSizesI32")
    void addIntContiguousLaneSizes(long size) {
        NDArray a = TestArrayFactory.random(505L, DType.i32, size);
        NDArray b = TestArrayFactory.random(606L, DType.i32, size);
        NDArray res = JNum.zeros(DType.i32, size);

        Add.addInt(a, b, res);

        for (int i = 0; i < size; i++) {
            assertEquals(a.getInt(i) + b.getInt(i), res.getInt(i));
        }
    }

    // =========================================================================
    // 03. Array + Scalar Variants
    // =========================================================================
    @Test
    @DisplayName("addFloat, addDouble, addInt scalar additions produce exact results")
    void addScalars() {
        NDArray aF = JNum.from(new float[]{1f, 2f, 3f}, 3);
        NDArray resF = JNum.zeros(DType.f32, 3);
        Add.addFloat(aF, 10f, resF);
        assertEquals(11f, resF.getFloat(0));
        assertEquals(13f, resF.getFloat(2));

        NDArray aD = JNum.from(new double[]{1.0, 2.0, 3.0}, 3);
        NDArray resD = JNum.zeros(DType.f64, 3);
        Add.addDouble(aD, 5.0, resD);
        assertEquals(6.0, resD.getDouble(0));
        assertEquals(8.0, resD.getDouble(2));

        NDArray aI = JNum.from(new int[]{1, 2, 3}, 3);
        NDArray resI = JNum.zeros(DType.i32, 3);
        Add.addInt(aI, 100, resI);
        assertEquals(101, resI.getInt(0));
        assertEquals(103, resI.getInt(2));
    }

    // =========================================================================
    // 04. Strided / Non-Contiguous Fallback Path
    // =========================================================================
    @Test
    @DisplayName("addFloat works identically on non-contiguous transposed views")
    void addNonContiguousStrided() {
        NDArray a = JNum.from(new float[]{1f, 2f, 3f, 4f}, 2, 2).transpose();
        NDArray b = JNum.from(new float[]{10f, 20f, 30f, 40f}, 2, 2).transpose();
        NDArray res = JNum.zeros(2, 2);

        assertFalse(a.isContiguous());
        assertFalse(b.isContiguous());

        Add.addFloat(a, b, res);

        assertEquals(11f, res.getFloat(0, 0)); // a(0,0)=1, b(0,0)=10
        assertEquals(33f, res.getFloat(0, 1)); // a(0,1)=3, b(0,1)=30
        assertEquals(22f, res.getFloat(1, 0)); // a(1,0)=2, b(1,0)=20
        assertEquals(44f, res.getFloat(1, 1)); // a(1,1)=4, b(1,1)=40
    }

    // =========================================================================
    // 05. Buffer In-Place Aliasing (out == in)
    // =========================================================================
    @Test
    @DisplayName("addFloat and addDouble support in-place aliasing (res == a)")
    void inPlaceAliasing() {
        NDArray a = JNum.from(new float[]{1f, 2f, 3f}, 3);
        NDArray b = JNum.from(new float[]{10f, 20f, 30f}, 3);

        Add.addFloat(a, b, a); // destination is a
        assertEquals(11f, a.getFloat(0));
        assertEquals(22f, a.getFloat(1));
        assertEquals(33f, a.getFloat(2));
    }

    // =========================================================================
    // 06. IEEE 754 & Boundary Values
    // =========================================================================
    @Test
    @DisplayName("addFloat handles NaN and Infinities according to IEEE 754")
    void ieeeBoundaries() {
        NDArray a = JNum.from(new float[]{Float.NaN, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY, 0.0f}, 4);
        NDArray b = JNum.from(new float[]{1.0f, 5.0f, -5.0f, -0.0f}, 4);
        NDArray res = JNum.zeros(4);

        Add.addFloat(a, b, res);

        assertTrue(Float.isNaN(res.getFloat(0)));
        assertEquals(Float.POSITIVE_INFINITY, res.getFloat(1));
        assertEquals(Float.NEGATIVE_INFINITY, res.getFloat(2));
        assertEquals(0.0f, res.getFloat(3));
    }
}
