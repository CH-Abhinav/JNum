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

@DisplayName("Log Kernel - Natural Logarithm Tests")
class LogTest {

    static LongStream laneSizesF32() {
        return LongStream.of(LaneSizes.boundarySizesF32()).filter(s -> s > 0);
    }

    @Test
    @DisplayName("Private constructor throws AssertionError upon reflection")
    void constructorThrows() throws Exception {
        Constructor<Log> constructor = Log.class.getDeclaredConstructor();
        constructor.setAccessible(true);
        InvocationTargetException ex = assertThrows(InvocationTargetException.class, constructor::newInstance);
        assertInstanceOf(AssertionError.class, ex.getCause());
    }

    @ParameterizedTest(name = "logFloat lane size: {0}")
    @MethodSource("laneSizesF32")
    void logFloatLaneSizes(long size) {
        float[] data = new float[(int) size];
        for (int i = 0; i < size; i++) data[i] = (float) (i + 1.0);
        NDArray a = JNum.from(data, size);
        NDArray res = JNum.zeros(DType.f32, size);

        Log.logFloat(a, res);

        for (int i = 0; i < size; i++) {
            assertEquals((float) Math.log(data[i]), res.getFloat(i), 1e-4f);
        }
    }

    @Test
    @DisplayName("logFloat handles zero producing -Infinity and negative producing NaN")
    void logBoundaries() {
        NDArray a = JNum.from(new float[]{0.0f, -1.0f, 1.0f}, 3);
        NDArray res = JNum.zeros(3);

        Log.logFloat(a, res);

        assertEquals(Float.NEGATIVE_INFINITY, res.getFloat(0));
        assertTrue(Float.isNaN(res.getFloat(1)));
        assertEquals(0.0f, res.getFloat(2));
    }
}
