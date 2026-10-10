package jnum.internal.kernel.trig;

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

@DisplayName("Tanh Kernel - Hyperbolic Tangent Tests")
class TanhTest {

    static LongStream laneSizesF32() {
        return LongStream.of(LaneSizes.boundarySizesF32()).filter(s -> s > 0);
    }

    @Test
    @DisplayName("Private constructor throws AssertionError upon reflection")
    void constructorThrows() throws Exception {
        Constructor<Tanh> constructor = Tanh.class.getDeclaredConstructor();
        constructor.setAccessible(true);
        InvocationTargetException ex = assertThrows(InvocationTargetException.class, constructor::newInstance);
        assertInstanceOf(AssertionError.class, ex.getCause());
    }

    @ParameterizedTest(name = "tanhFloat lane size: {0}")
    @MethodSource("laneSizesF32")
    void tanhFloatLaneSizes(long size) {
        float[] data = new float[(int) size];
        for (int i = 0; i < size; i++) data[i] = (float) (i * 0.1 - 2.0);
        NDArray a = JNum.from(data, size);
        NDArray res = JNum.zeros(DType.f32, size);

        Tanh.tanhFloat(a, res);

        for (int i = 0; i < size; i++) {
            assertEquals((float) Math.tanh(data[i]), res.getFloat(i), 1e-4f);
        }
    }

    @Test
    @DisplayName("tanhDouble computes exact hyperbolic tangent values")
    void tanhDoubleKnown() {
        NDArray a = JNum.from(new double[]{0.0, 1.0, -1.0}, 3);
        NDArray res = JNum.zeros(DType.f64, 3);

        Tanh.tanhDouble(a, res);

        assertEquals(0.0, res.getDouble(0), 1e-9);
        assertEquals(Math.tanh(1.0), res.getDouble(1), 1e-9);
        assertEquals(Math.tanh(-1.0), res.getDouble(2), 1e-9);
    }
}
