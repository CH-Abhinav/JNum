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

@DisplayName("Log10 Kernel - Base-10 Logarithm Tests")
class Log10Test {

    static LongStream laneSizesF32() {
        return LongStream.of(LaneSizes.boundarySizesF32()).filter(s -> s > 0);
    }

    @Test
    @DisplayName("Private constructor throws AssertionError upon reflection")
    void constructorThrows() throws Exception {
        Constructor<Log10> constructor = Log10.class.getDeclaredConstructor();
        constructor.setAccessible(true);
        InvocationTargetException ex = assertThrows(InvocationTargetException.class, constructor::newInstance);
        assertInstanceOf(AssertionError.class, ex.getCause());
    }

    @ParameterizedTest(name = "log10Float lane size: {0}")
    @MethodSource("laneSizesF32")
    void log10FloatLaneSizes(long size) {
        float[] data = new float[(int) size];
        for (int i = 0; i < size; i++) data[i] = (float) (i + 1.0);
        NDArray a = JNum.from(data, size);
        NDArray res = JNum.zeros(DType.f32, size);

        Log10.log10Float(a, res);

        for (int i = 0; i < size; i++) {
            assertEquals((float) Math.log10(data[i]), res.getFloat(i), 1e-4f);
        }
    }

    @Test
    @DisplayName("log10Double computes exact powers of 10")
    void log10DoublePowers() {
        NDArray a = JNum.from(new double[]{1.0, 10.0, 100.0, 1000.0}, 4);
        NDArray res = JNum.zeros(DType.f64, 4);

        Log10.log10Double(a, res);

        assertEquals(0.0, res.getDouble(0), 1e-9);
        assertEquals(1.0, res.getDouble(1), 1e-9);
        assertEquals(2.0, res.getDouble(2), 1e-9);
        assertEquals(3.0, res.getDouble(3), 1e-9);
    }
}
