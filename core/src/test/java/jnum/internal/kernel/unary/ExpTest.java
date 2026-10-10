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

@DisplayName("Exp Kernel - Natural Exponential Tests")
class ExpTest {

    static LongStream laneSizesF32() {
        return LongStream.of(LaneSizes.boundarySizesF32()).filter(s -> s > 0);
    }

    @Test
    @DisplayName("Private constructor throws AssertionError upon reflection")
    void constructorThrows() throws Exception {
        Constructor<Exp> constructor = Exp.class.getDeclaredConstructor();
        constructor.setAccessible(true);
        InvocationTargetException ex = assertThrows(InvocationTargetException.class, constructor::newInstance);
        assertInstanceOf(AssertionError.class, ex.getCause());
    }

    @ParameterizedTest(name = "expFloat lane size: {0}")
    @MethodSource("laneSizesF32")
    void expFloatLaneSizes(long size) {
        float[] data = new float[(int) size];
        for (int i = 0; i < size; i++) data[i] = (float) (i * 0.05);
        NDArray a = JNum.from(data, size);
        NDArray res = JNum.zeros(DType.f32, size);

        Exp.expFloat(a, res);

        for (int i = 0; i < size; i++) {
            assertEquals((float) Math.exp(data[i]), res.getFloat(i), 1e-4f);
        }
    }

    @Test
    @DisplayName("expDouble computes exact exponential values")
    void expDoubleKnown() {
        NDArray a = JNum.from(new double[]{0.0, 1.0, 2.0}, 3);
        NDArray res = JNum.zeros(DType.f64, 3);

        Exp.expDouble(a, res);

        assertEquals(1.0, res.getDouble(0), 1e-9);
        assertEquals(Math.E, res.getDouble(1), 1e-9);
        assertEquals(Math.exp(2.0), res.getDouble(2), 1e-9);
    }

    @Test
    @DisplayName("expFloat handles large numbers with overflow to +Infinity")
    void expFloatOverflow() {
        NDArray a = JNum.from(new float[]{1000f, -1000f}, 2);
        NDArray res = JNum.zeros(2);

        Exp.expFloat(a, res);

        assertEquals(Float.POSITIVE_INFINITY, res.getFloat(0));
        assertEquals(0.0f, res.getFloat(1));
    }
}
