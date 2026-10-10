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

@DisplayName("Minimum Kernel - Elementwise Minimum Tests")
class MinimumTest {

    static LongStream laneSizesF32() {
        return LongStream.of(LaneSizes.boundarySizesF32()).filter(s -> s > 0);
    }

    @Test
    @DisplayName("Private constructor throws AssertionError upon reflection")
    void constructorThrows() throws Exception {
        Constructor<Minimum> constructor = Minimum.class.getDeclaredConstructor();
        constructor.setAccessible(true);
        InvocationTargetException ex = assertThrows(InvocationTargetException.class, constructor::newInstance);
        assertInstanceOf(AssertionError.class, ex.getCause());
    }

    @ParameterizedTest(name = "minimumFloat contiguous lane size: {0}")
    @MethodSource("laneSizesF32")
    void minimumFloatLaneSizes(long size) {
        NDArray a = TestArrayFactory.random(2101L, DType.f32, size);
        NDArray b = TestArrayFactory.random(2102L, DType.f32, size);
        NDArray res = JNum.zeros(DType.f32, size);

        Minimum.minimumFloat(a, b, res);

        for (int i = 0; i < size; i++) {
            assertEquals(Math.min(a.getFloat(i), b.getFloat(i)), res.getFloat(i));
        }
    }

    @Test
    @DisplayName("minimumDouble and minimumInt scalar operations produce expected values")
    void minimumScalars() {
        NDArray aD = JNum.from(new double[]{1.0, 5.0, 10.0}, 3);
        NDArray resD = JNum.zeros(DType.f64, 3);
        Minimum.minimumDouble(aD, 6.0, resD);
        assertEquals(1.0, resD.getDouble(0));
        assertEquals(5.0, resD.getDouble(1));
        assertEquals(6.0, resD.getDouble(2));

        NDArray aI = JNum.from(new int[]{1, 5, 10}, 3);
        NDArray resI = JNum.zeros(DType.i32, 3);
        Minimum.minimumInt(aI, 6, resI);
        assertEquals(1, resI.getInt(0));
        assertEquals(5, resI.getInt(1));
        assertEquals(6, resI.getInt(2));
    }
}
