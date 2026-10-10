package jnum.internal.ops;

import static org.junit.jupiter.api.Assertions.*;

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import jnum.DType;
import jnum.JNum;
import jnum.NDArray;
import jnum.testutil.TestArrayFactory;

public class GetOpsTest {

    @Test
    @DisplayName("Private constructor throws AssertionError")
    void testPrivateConstructor() throws Exception {
        Constructor<GetOps> constructor = GetOps.class.getDeclaredConstructor();
        constructor.setAccessible(true);
        InvocationTargetException ex = assertThrows(InvocationTargetException.class, constructor::newInstance);
        assertInstanceOf(AssertionError.class, ex.getCause());
    }

    @Test
    @DisplayName("Physical offset calculation: contiguous and non-contiguous")
    void testPhysicalOffset() {
        long[] shape = new long[]{2, 3};
        long[] stridesContig = new long[]{3, 1};
        assertEquals(0, GetOps.getPhysicalOffset(0, shape, stridesContig));
        assertEquals(4, GetOps.getPhysicalOffset(4, shape, stridesContig)); // row 1, col 1 -> 1*3 + 1 = 4

        long[] stridesTransposed = new long[]{1, 2};
        // logical 4 -> (1, 1) -> 1*1 + 1*2 = 3
        assertEquals(3, GetOps.getPhysicalOffset(4, shape, stridesTransposed));
    }

    @Test
    @DisplayName("Flat getters: float, double, int, bool")
    void testFlatGetters() {
        NDArray fArr = JNum.from(new float[]{1.5f, 2.5f}, 2);
        assertEquals(1.5f, GetOps.getFlatFloat(fArr, 0), 1e-6f);
        assertEquals(2.5, GetOps.getFlat(fArr, 1), 1e-6);

        NDArray dArr = JNum.from(new double[]{3.14, 2.71}, 2);
        assertEquals(3.14, GetOps.getFlatDouble(dArr, 0), 1e-12);

        NDArray iArr = JNum.from(new int[]{42, -99}, 2);
        assertEquals(42, GetOps.getFlatInt(iArr, 0));
        assertEquals(-99, GetOps.getFlatInt(iArr, 1));

        NDArray bArr = JNum.from(new boolean[]{true, false}, 2);
        assertTrue(GetOps.getFlatBoolean(bArr, 0));
        assertFalse(GetFlatBooleanWrapper(bArr, 1));
    }

    private boolean GetFlatBooleanWrapper(NDArray arr, long idx) {
        return GetOps.getFlatBoolean(arr, idx);
    }

    @Test
    @DisplayName("Out of bounds flat index throws IndexOutOfBoundsException")
    void testFlatIndexOutOfBounds() {
        NDArray arr = JNum.zeros(DType.f32, 5);
        assertThrows(IndexOutOfBoundsException.class, () -> GetOps.getFlat(arr, -1));
        assertThrows(IndexOutOfBoundsException.class, () -> GetOps.getFlat(arr, 5));
        assertThrows(IndexOutOfBoundsException.class, () -> GetOps.getFlatFloat(arr, 10));
    }

    @Test
    @DisplayName("Multi-dimensional coordinate getters with negative indices")
    void testCoordinateGetters() {
        NDArray m = TestArrayFactory.matrix(new float[][]{
            {10f, 20f, 30f},
            {40f, 50f, 60f}
        });

        assertEquals(10f, GetOps.getFloat(m, 0, 0), 1e-6f);
        assertEquals(60f, GetOps.getFloat(m, 1, 2), 1e-6f);
        // Negative indexing
        assertEquals(60f, GetOps.getFloat(m, -1, -1), 1e-6f);
        assertEquals(40f, GetOps.getFloat(m, -1, 0), 1e-6f);

        // General indices varargs
        assertEquals(50.0, GetOps.get(m, 1, 1), 1e-6);

        // Shape mismatch throws IllegalArgumentException
        assertThrows(IllegalArgumentException.class, () -> GetOps.get(m, 0, 1, 2));
        assertThrows(IndexOutOfBoundsException.class, () -> GetOps.getFloat(m, 5, 0));
    }
}
