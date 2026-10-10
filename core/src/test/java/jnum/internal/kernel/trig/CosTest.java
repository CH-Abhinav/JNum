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

@DisplayName("Cos Kernel - SIMD & Mathematical Correctness Tests")
class CosTest {

    static LongStream laneSizesF32() {
        return LongStream.of(LaneSizes.boundarySizesF32()).filter(s -> s > 0);
    }

    @Test
    @DisplayName("Private constructor throws AssertionError upon reflection")
    void constructorThrows() throws Exception {
        Constructor<Cos> constructor = Cos.class.getDeclaredConstructor();
        constructor.setAccessible(true);
        InvocationTargetException ex = assertThrows(InvocationTargetException.class, constructor::newInstance);
        assertInstanceOf(AssertionError.class, ex.getCause());
    }

    @ParameterizedTest(name = "cosFloat lane size: {0}")
    @MethodSource("laneSizesF32")
    void cosFloatLaneSizes(long size) {
        float[] data = new float[(int) size];
        for (int i = 0; i < size; i++) data[i] = (float) (i * 0.1);
        NDArray a = JNum.from(data, size);
        NDArray res = JNum.zeros(DType.f32, size);

        Cos.cosFloat(a, res);

        for (int i = 0; i < size; i++) {
            assertEquals((float) Math.cos(data[i]), res.getFloat(i), 1e-5f);
        }
    }

    @Test
    @DisplayName("cosDouble computes accurate values for known angles")
    void cosDoubleKnownAngles() {
        NDArray a = JNum.from(new double[]{0.0, Math.PI / 3.0, Math.PI / 2.0, Math.PI}, 4);
        NDArray res = JNum.zeros(DType.f64, 4);

        Cos.cosDouble(a, res);

        assertEquals(1.0, res.getDouble(0), 1e-9);
        assertEquals(0.5, res.getDouble(1), 1e-9);
        assertEquals(0.0, res.getDouble(2), 1e-9);
        assertEquals(-1.0, res.getDouble(3), 1e-9);
    }
}
