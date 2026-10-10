package jnum.internal.kernel.reduce;

import jnum.DType;
import jnum.JNum;
import jnum.NDArray;
import jnum.testutil.LaneSizes;
import jnum.testutil.ReferenceOps;
import jnum.testutil.TestArrayFactory;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.util.stream.LongStream;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("Dot Kernel - Vector Dot Product Tests")
class DotTest {

    static LongStream laneSizesF32() {
        return LongStream.of(LaneSizes.boundarySizesF32()).filter(s -> s > 0);
    }

    @Test
    @DisplayName("Private constructor throws AssertionError upon reflection")
    void constructorThrows() throws Exception {
        Constructor<Dot> constructor = Dot.class.getDeclaredConstructor();
        constructor.setAccessible(true);
        InvocationTargetException ex = assertThrows(InvocationTargetException.class, constructor::newInstance);
        assertInstanceOf(AssertionError.class, ex.getCause());
    }

    @ParameterizedTest(name = "dotFloat lane size: {0}")
    @MethodSource("laneSizesF32")
    void dotFloatLaneSizes(long size) {
        float[] aData = new float[(int) size];
        float[] bData = new float[(int) size];
        for (int i = 0; i < size; i++) {
            aData[i] = (i % 5) + 1;
            bData[i] = 1.0f;
        }
        NDArray a = JNum.from(aData, size);
        NDArray b = JNum.from(bData, size);

        double actual = Dot.dotFloat(a, b);
        double expected = ReferenceOps.dot(aData, bData);

        assertEquals(expected, actual, 1e-4);
    }

    @Test
    @DisplayName("dotDouble computes accurate double dot product")
    void dotDoubleStandard() {
        NDArray a = JNum.from(new double[]{1.0, 2.0, 3.0, 4.0}, 4);
        NDArray b = JNum.from(new double[]{0.5, 1.0, 1.5, 2.0}, 4);

        // 0.5 + 2.0 + 4.5 + 8.0 = 15.0
        double actual = Dot.dotDouble(a, b);
        assertEquals(15.0, actual, 1e-9);
    }

    @Test
    @DisplayName("dotInt computes exact integer dot product")
    void dotIntStandard() {
        NDArray a = JNum.from(new int[]{1, 2, 3, 4}, 4);
        NDArray b = JNum.from(new int[]{10, 20, 30, 40}, 4);

        // 10 + 40 + 90 + 160 = 300
        double actual = Dot.dotInt(a, b);
        assertEquals(300.0, actual);
    }

    @Test
    @DisplayName("dotFloat works on non-contiguous transposed views")
    void dotNonContiguous() {
        NDArray a = JNum.from(new float[]{1f, 2f, 3f, 4f}, 2, 2).transpose();
        NDArray b = JNum.from(new float[]{5f, 6f, 7f, 8f}, 2, 2).transpose();

        // Elements in column-major order: a=[1, 3, 2, 4], b=[5, 7, 6, 8]
        // 1*5 + 3*7 + 2*6 + 4*8 = 5 + 21 + 12 + 32 = 70
        double actual = Dot.dotFloat(a, b);
        assertEquals(70.0, actual, 1e-4);
    }
}
