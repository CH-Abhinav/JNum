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

@DisplayName("Sigmoid Kernel - Logistic Sigmoid Tests")
class SigmoidTest {

    static LongStream laneSizesF32() {
        return LongStream.of(LaneSizes.boundarySizesF32()).filter(s -> s > 0);
    }

    @Test
    @DisplayName("Private constructor throws AssertionError upon reflection")
    void constructorThrows() throws Exception {
        Constructor<Sigmoid> constructor = Sigmoid.class.getDeclaredConstructor();
        constructor.setAccessible(true);
        InvocationTargetException ex = assertThrows(InvocationTargetException.class, constructor::newInstance);
        assertInstanceOf(AssertionError.class, ex.getCause());
    }

    @ParameterizedTest(name = "sigmoidFloat lane size: {0}")
    @MethodSource("laneSizesF32")
    void sigmoidFloatLaneSizes(long size) {
        float[] data = new float[(int) size];
        for (int i = 0; i < size; i++) data[i] = (float) (i * 0.1 - 2.0);
        NDArray a = JNum.from(data, size);
        NDArray res = JNum.zeros(DType.f32, size);

        Sigmoid.sigmoidFloat(a, res);

        for (int i = 0; i < size; i++) {
            float expected = (float) (1.0 / (1.0 + Math.exp(-data[i])));
            assertEquals(expected, res.getFloat(i), 1e-4f);
        }
    }

    @Test
    @DisplayName("sigmoidFloat handles extreme values without NaN")
    void sigmoidExtremes() {
        NDArray a = JNum.from(new float[]{0.0f, 100.0f, -100.0f}, 3);
        NDArray res = JNum.zeros(3);

        Sigmoid.sigmoidFloat(a, res);

        assertEquals(0.5f, res.getFloat(0), 1e-6f);
        assertEquals(1.0f, res.getFloat(1), 1e-6f);
        assertEquals(0.0f, res.getFloat(2), 1e-6f);
    }
}
