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

@DisplayName("Sum Kernel - Global & Axis Reductions Tests")
class SumTest {

    static LongStream laneSizesF32() {
        return LongStream.of(LaneSizes.boundarySizesF32()).filter(s -> s > 0);
    }

    @Test
    @DisplayName("Private constructor throws AssertionError upon reflection")
    void constructorThrows() throws Exception {
        Constructor<Sum> constructor = Sum.class.getDeclaredConstructor();
        constructor.setAccessible(true);
        InvocationTargetException ex = assertThrows(InvocationTargetException.class, constructor::newInstance);
        assertInstanceOf(AssertionError.class, ex.getCause());
    }

    @ParameterizedTest(name = "sumFloat global lane size: {0}")
    @MethodSource("laneSizesF32")
    void sumFloatGlobal(long size) {
        float[] data = new float[(int) size];
        double expected = 0.0;
        for (int i = 0; i < size; i++) {
            data[i] = 1.0f;
            expected += 1.0;
        }
        NDArray a = JNum.from(data, size);
        double actual = Sum.sumFloat(a);
        assertEquals(expected, actual, 1e-4);
    }

    @Test
    @DisplayName("sumDouble and sumInt global reductions")
    void sumDoubleAndInt() {
        NDArray aD = JNum.from(new double[]{1.0, 2.0, 3.0, 4.0}, 4);
        assertEquals(10.0, Sum.sumDouble(aD), 1e-9);

        NDArray aI = JNum.from(new int[]{10, 20, 30}, 3);
        assertEquals(60.0, Sum.sumInt(aI));
    }

    @Test
    @DisplayName("sumFloat on non-contiguous transposed views")
    void sumNonContiguous() {
        NDArray a = JNum.from(new float[]{1f, 2f, 3f, 4f}, 2, 2).transpose();
        assertEquals(10.0, Sum.sumFloat(a), 1e-6);
    }

    @Test
    @DisplayName("sumFloatAxis reduces along axis 0 and axis 1 correctly")
    void sumAxis() {
        NDArray a = JNum.from(new float[]{
            1f, 2f, 3f,
            4f, 5f, 6f
        }, 2, 3);

        // Sum along axis 0 (columns sum) -> shape [3]
        NDArray res0 = JNum.zeros(3);
        Sum.sumFloatAxis(a, 0, res0);
        assertEquals(5f, res0.getFloat(0)); // 1 + 4
        assertEquals(7f, res0.getFloat(1)); // 2 + 5
        assertEquals(9f, res0.getFloat(2)); // 3 + 6

        // Sum along axis 1 (rows sum) -> shape [2]
        NDArray res1 = JNum.zeros(2);
        Sum.sumFloatAxis(a, 1, res1);
        assertEquals(6f, res1.getFloat(0));  // 1 + 2 + 3
        assertEquals(15f, res1.getFloat(1)); // 4 + 5 + 6
    }
}
