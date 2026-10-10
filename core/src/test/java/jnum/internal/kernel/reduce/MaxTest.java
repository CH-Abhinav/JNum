package jnum.internal.kernel.reduce;

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

@DisplayName("Max Kernel - Global & Axis Maximum Reductions Tests")
class MaxTest {

    static LongStream laneSizesF32() {
        return LongStream.of(LaneSizes.boundarySizesF32()).filter(s -> s > 0);
    }

    @Test
    @DisplayName("Private constructor throws AssertionError upon reflection")
    void constructorThrows() throws Exception {
        Constructor<Max> constructor = Max.class.getDeclaredConstructor();
        constructor.setAccessible(true);
        InvocationTargetException ex = assertThrows(InvocationTargetException.class, constructor::newInstance);
        assertInstanceOf(AssertionError.class, ex.getCause());
    }

    @ParameterizedTest(name = "maxFloat global lane size: {0}")
    @MethodSource("laneSizesF32")
    void maxFloatGlobal(long size) {
        float[] data = new float[(int) size];
        for (int i = 0; i < size; i++) data[i] = (float) i;
        NDArray a = JNum.from(data, size);

        double actual = Max.maxFloat(a);
        assertEquals((double) (size - 1), actual, 1e-5);
    }

    @Test
    @DisplayName("maxDouble and maxInt global reductions")
    void maxDoubleAndInt() {
        NDArray aD = JNum.from(new double[]{1.0, 99.0, 3.0}, 3);
        assertEquals(99.0, Max.maxDouble(aD), 1e-9);

        NDArray aI = JNum.from(new int[]{10, -50, 42}, 3);
        assertEquals(42.0, Max.maxInt(aI));
    }

    @Test
    @DisplayName("maxFloatAxis reduces along axes correctly")
    void maxAxis() {
        NDArray a = JNum.from(new float[]{
            10f, 2f,
            4f, 25f
        }, 2, 2);

        // Max along axis 0 -> [10, 25]
        NDArray res0 = JNum.zeros(2);
        Max.maxFloatAxis(a, 0, res0);
        assertEquals(10f, res0.getFloat(0));
        assertEquals(25f, res0.getFloat(1));

        // Max along axis 1 -> [10, 25]
        NDArray res1 = JNum.zeros(2);
        Max.maxFloatAxis(a, 1, res1);
        assertEquals(10f, res1.getFloat(0));
        assertEquals(25f, res1.getFloat(1));
    }
}
