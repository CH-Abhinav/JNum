package jnum.internal.kernel.unary;

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

@DisplayName("Sqrt Kernel - Square Root Tests")
class SqrtTest {

    static LongStream laneSizesF32() {
        return LongStream.of(LaneSizes.boundarySizesF32()).filter(s -> s > 0);
    }

    @Test
    @DisplayName("Private constructor throws AssertionError upon reflection")
    void constructorThrows() throws Exception {
        Constructor<Sqrt> constructor = Sqrt.class.getDeclaredConstructor();
        constructor.setAccessible(true);
        InvocationTargetException ex = assertThrows(InvocationTargetException.class, constructor::newInstance);
        assertInstanceOf(AssertionError.class, ex.getCause());
    }

    @ParameterizedTest(name = "sqrtFloat lane size: {0}")
    @MethodSource("laneSizesF32")
    void sqrtFloatLaneSizes(long size) {
        float[] data = new float[(int) size];
        for (int i = 0; i < size; i++) data[i] = (float) ((i + 1) * (i + 1));
        NDArray a = JNum.from(data, size);
        NDArray res = JNum.zeros(DType.f32, size);

        Sqrt.sqrtFloat(a, res);

        for (int i = 0; i < size; i++) {
            assertEquals((float) (i + 1), res.getFloat(i), 1e-4f);
        }
    }

    @Test
    @DisplayName("sqrtFloat handles 0.0, positive infinity, and negative numbers (NaN)")
    void sqrtBoundaries() {
        NDArray a = JNum.from(new float[]{0.0f, Float.POSITIVE_INFINITY, -4.0f}, 3);
        NDArray res = JNum.zeros(3);

        Sqrt.sqrtFloat(a, res);

        assertEquals(0.0f, res.getFloat(0));
        assertEquals(Float.POSITIVE_INFINITY, res.getFloat(1));
        assertTrue(Float.isNaN(res.getFloat(2)));
    }
}
