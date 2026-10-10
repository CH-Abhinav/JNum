package jnum.internal.kernel.compare;

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

@DisplayName("Maximum Kernel - Elementwise Maximum Tests")
class MaximumTest {

    static LongStream laneSizesF32() {
        return LongStream.of(LaneSizes.boundarySizesF32()).filter(s -> s > 0);
    }

    @Test
    @DisplayName("Private constructor throws AssertionError upon reflection")
    void constructorThrows() throws Exception {
        Constructor<Maximum> constructor = Maximum.class.getDeclaredConstructor();
        constructor.setAccessible(true);
        InvocationTargetException ex = assertThrows(InvocationTargetException.class, constructor::newInstance);
        assertInstanceOf(AssertionError.class, ex.getCause());
    }

    @ParameterizedTest(name = "maximumFloat contiguous lane size: {0}")
    @MethodSource("laneSizesF32")
    void maximumFloatLaneSizes(long size) {
        NDArray a = TestArrayFactory.random(2001L, DType.f32, size);
        NDArray b = TestArrayFactory.random(2002L, DType.f32, size);
        NDArray res = JNum.zeros(DType.f32, size);

        Maximum.maximumFloat(a, b, res);

        for (int i = 0; i < size; i++) {
            assertEquals(Math.max(a.getFloat(i), b.getFloat(i)), res.getFloat(i));
        }
    }

    @Test
    @DisplayName("maximumFloat handles signed zeros according to Math.max")
    void maximumSignedZeros() {
        NDArray a = JNum.from(new float[]{0.0f, -0.0f}, 2);
        NDArray b = JNum.from(new float[]{-0.0f, 0.0f}, 2);
        NDArray res = JNum.zeros(2);

        Maximum.maximumFloat(a, b, res);

        assertEquals(0.0f, res.getFloat(0));
        assertEquals(0.0f, res.getFloat(1));
    }

    @Test
    @DisplayName("maximumDouble and maximumInt scalar operations produce expected values")
    void maximumScalars() {
        NDArray aD = JNum.from(new double[]{1.0, 5.0, 10.0}, 3);
        NDArray resD = JNum.zeros(DType.f64, 3);
        Maximum.maximumDouble(aD, 6.0, resD);
        assertEquals(6.0, resD.getDouble(0));
        assertEquals(6.0, resD.getDouble(1));
        assertEquals(10.0, resD.getDouble(2));

        NDArray aI = JNum.from(new int[]{1, 5, 10}, 3);
        NDArray resI = JNum.zeros(DType.i32, 3);
        Maximum.maximumInt(aI, 6, resI);
        assertEquals(6, resI.getInt(0));
        assertEquals(6, resI.getInt(1));
        assertEquals(10, resI.getInt(2));
    }
}
