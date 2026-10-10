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

@DisplayName("Sinh Kernel - Hyperbolic Sine Tests")
class SinhTest {

    static LongStream laneSizesF32() {
        return LongStream.of(LaneSizes.boundarySizesF32()).filter(s -> s > 0);
    }

    @Test
    @DisplayName("Private constructor throws AssertionError upon reflection")
    void constructorThrows() throws Exception {
        Constructor<Sinh> constructor = Sinh.class.getDeclaredConstructor();
        constructor.setAccessible(true);
        InvocationTargetException ex = assertThrows(InvocationTargetException.class, constructor::newInstance);
        assertInstanceOf(AssertionError.class, ex.getCause());
    }

    @ParameterizedTest(name = "sinhFloat lane size: {0}")
    @MethodSource("laneSizesF32")
    void sinhFloatLaneSizes(long size) {
        float[] data = new float[(int) size];
        for (int i = 0; i < size; i++) data[i] = (float) (i * 0.05);
        NDArray a = JNum.from(data, size);
        NDArray res = JNum.zeros(DType.f32, size);

        Sinh.sinhFloat(a, res);

        for (int i = 0; i < size; i++) {
            assertEquals((float) Math.sinh(data[i]), res.getFloat(i), 1e-4f);
        }
    }

    @Test
    @DisplayName("sinhDouble computes exact hyperbolic values")
    void sinhDoubleKnown() {
        NDArray a = JNum.from(new double[]{0.0, 1.0}, 2);
        NDArray res = JNum.zeros(DType.f64, 2);

        Sinh.sinhDouble(a, res);

        assertEquals(0.0, res.getDouble(0), 1e-9);
        assertEquals(Math.sinh(1.0), res.getDouble(1), 1e-9);
    }
}
