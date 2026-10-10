package jnum.internal.kernel.reduce;

import jnum.DType;
import jnum.JNum;
import jnum.NDArray;
import jnum.testutil.LaneSizes;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.util.stream.LongStream;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("Min Kernel - Global & Axis Minimum Reductions Tests")
class MinTest {

    static LongStream laneSizesF32() {
        return LongStream.of(LaneSizes.boundarySizesF32()).filter(s -> s > 0);
    }

    @Test
    @DisplayName("Private constructor throws AssertionError upon reflection")
    void constructorThrows() throws Exception {
        Constructor<Min> constructor = Min.class.getDeclaredConstructor();
        constructor.setAccessible(true);
        InvocationTargetException ex = assertThrows(InvocationTargetException.class, constructor::newInstance);
        assertInstanceOf(AssertionError.class, ex.getCause());
    }

    @ParameterizedTest(name = "minFloat global lane size: {0}")
    @MethodSource("laneSizesF32")
    void minFloatGlobal(long size) {
        float[] data = new float[(int) size];
        for (int i = 0; i < size; i++) data[i] = (float) (i + 10);
        NDArray a = JNum.from(data, size);

        double actual = Min.minFloat(a);
        assertEquals(10.0, actual, 1e-5);
    }

    @Test
    @DisplayName("minDouble and minInt global reductions")
    void minDoubleAndInt() {
        NDArray aD = JNum.from(new double[]{10.0, -99.0, 3.0}, 3);
        assertEquals(-99.0, Min.minDouble(aD), 1e-9);

        NDArray aI = JNum.from(new int[]{10, -50, 42}, 3);
        assertEquals(-50.0, Min.minInt(aI));
    }

    @Test
    @DisplayName("minFloatAxis reduces along axes correctly")
    void minAxis() {
        NDArray a = JNum.from(new float[]{
            10f, 2f,
            4f, 25f
        }, 2, 2);

        // Min along axis 0 -> [4, 2]
        NDArray res0 = JNum.zeros(2);
        Min.minFloatAxis(a, 0, res0);
        assertEquals(4f, res0.getFloat(0));
        assertEquals(2f, res0.getFloat(1));

        // Min along axis 1 -> [2, 4]
        NDArray res1 = JNum.zeros(2);
        Min.minFloatAxis(a, 1, res1);
        assertEquals(2f, res1.getFloat(0));
        assertEquals(4f, res1.getFloat(1));
    }
}
