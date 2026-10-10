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

@DisplayName("Abs Kernel - Elementwise Absolute Value Tests")
class AbsTest {

    static LongStream laneSizesF32() {
        return LongStream.of(LaneSizes.boundarySizesF32()).filter(s -> s > 0);
    }

    @Test
    @DisplayName("Private constructor throws AssertionError upon reflection")
    void constructorThrows() throws Exception {
        Constructor<Abs> constructor = Abs.class.getDeclaredConstructor();
        constructor.setAccessible(true);
        InvocationTargetException ex = assertThrows(InvocationTargetException.class, constructor::newInstance);
        assertInstanceOf(AssertionError.class, ex.getCause());
    }

    @ParameterizedTest(name = "absFloat lane size: {0}")
    @MethodSource("laneSizesF32")
    void absFloatLaneSizes(long size) {
        float[] data = new float[(int) size];
        for (int i = 0; i < size; i++) data[i] = -((float) i + 1.0f);
        NDArray a = JNum.from(data, size);
        NDArray res = JNum.zeros(DType.f32, size);

        Abs.absFloat(a, res);

        for (int i = 0; i < size; i++) {
            assertEquals((float) (i + 1), res.getFloat(i));
        }
    }

    @Test
    @DisplayName("absDouble and absInt compute positive magnitudes")
    void absDoubleAndInt() {
        NDArray aD = JNum.from(new double[]{-1.5, 2.5, -3.5}, 3);
        NDArray resD = JNum.zeros(DType.f64, 3);
        Abs.absDouble(aD, resD);
        assertEquals(1.5, resD.getDouble(0));
        assertEquals(2.5, resD.getDouble(1));
        assertEquals(3.5, resD.getDouble(2));

        NDArray aI = JNum.from(new int[]{-10, 20, -30}, 3);
        NDArray resI = JNum.zeros(DType.i32, 3);
        Abs.absInt(aI, resI);
        assertEquals(10, resI.getInt(0));
        assertEquals(20, resI.getInt(1));
        assertEquals(30, resI.getInt(2));
    }
}
