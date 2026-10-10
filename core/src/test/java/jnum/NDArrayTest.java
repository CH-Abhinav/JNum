package jnum;

import static jnum.DType.*;
import static org.junit.jupiter.api.Assertions.*;

import java.lang.foreign.Arena;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import org.junit.jupiter.api.Test;

class NDArrayTest {

    @Test
    void viewSemanticsAndCopyPreserveLogicalElements() {
        NDArray dense = JNum.from(new float[]{1f, 2f, 3f, 4f}, 2, 2);
        NDArray transposed = dense.transpose();
        NDArray broadcast = JNum.from(new float[]{5f, 6f}, 1, 2).broadcastTo(3, 2);

        assertTrue(dense.isContiguous());
        assertFalse(transposed.isContiguous());
        assertFalse(broadcast.isContiguous());

        NDArray copied = transposed.copy();

        assertTrue(copied.isContiguous());
        assertEquals(f32, copied.getDType());
        assertEquals(1f, copied.getFloat(0, 0), 1e-6f);
        assertEquals(3f, copied.getFloat(0, 1), 1e-6f);
        assertEquals(2f, copied.getFloat(1, 0), 1e-6f);
        assertEquals(4f, copied.getFloat(1, 1), 1e-6f);
    }

    @Test
    void dotPromotesMixedDtypesCorrectly() {
        NDArray left = JNum.from(new float[]{1f, 2f, 3f}, 3);
        NDArray right = JNum.from(new double[]{0.5, 1.5, 2.0}, 3);

        double result = left.dot(right);

        assertEquals(9.5, result, 1e-9);
    }

    @Test
    void matmulPromotesMixedDtypesCorrectly() {
        NDArray left = JNum.from(new float[]{1f, 2f, 3f, 4f}, 2, 2);
        NDArray right = JNum.from(new double[]{5.0, 6.0, 7.0, 8.0}, 2, 2);

        NDArray result = left.matmul(right);

        assertEquals(DType.f64, result.getDType());
        assertArrayEquals(new long[]{2, 2}, result.getShape());
        assertEquals(19.0, result.get(0, 0), 1e-9);
        assertEquals(22.0, result.get(0, 1), 1e-9);
        assertEquals(43.0, result.get(1, 0), 1e-9);
        assertEquals(50.0, result.get(1, 1), 1e-9);
    }

    @Test
    void matmulRejectsNonContiguousOutputBuffer() {
        NDArray left = JNum.ones(f32, 2, 2);
        NDArray right = JNum.ones(f32, 2, 2);
        NDArray nonContiguousOutput = JNum.zeros(f32, 2, 2).transpose();

        IllegalArgumentException ex = assertThrows(
            IllegalArgumentException.class,
            () -> left.matmul(right, nonContiguousOutput)
        );

        assertTrue(ex.getMessage().contains("contiguous"));
    }

    @Test
    void addBroadcastsSmallerArrayAcrossLargerArray() {
        NDArray left = JNum.from(new float[]{1f, 2f, 3f, 4f}, 2, 2);
        NDArray right = JNum.from(new float[]{10f, 20f}, 2);

        NDArray result = left.add(right);

        assertArrayEquals(new long[]{2, 2}, result.getShape());
        assertEquals(11f, result.getFloat(0, 0), 1e-6f);
        assertEquals(22f, result.getFloat(0, 1), 1e-6f);
        assertEquals(13f, result.getFloat(1, 0), 1e-6f);
        assertEquals(24f, result.getFloat(1, 1), 1e-6f);
    }

    @Test
    void axisReductionsWorkForContiguousAndTransposedViews() {
        NDArray dense = JNum.from(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, 2, 3);
        NDArray transposed = dense.transpose();

        NDArray denseSumAxisOne = dense.sum(1);
        NDArray transposedSumAxisOne = transposed.sum(1);
        NDArray transposedMaxAxisZero = transposed.max(0);

        assertArrayEquals(new long[]{2}, denseSumAxisOne.getShape());
        assertEquals(6f, denseSumAxisOne.getFloat(0), 1e-6f);
        assertEquals(15f, denseSumAxisOne.getFloat(1), 1e-6f);

        assertArrayEquals(new long[]{3}, transposedSumAxisOne.getShape());
        assertEquals(5f, transposedSumAxisOne.getFloat(0), 1e-6f);
        assertEquals(7f, transposedSumAxisOne.getFloat(1), 1e-6f);
        assertEquals(9f, transposedSumAxisOne.getFloat(2), 1e-6f);

        assertArrayEquals(new long[]{2}, transposedMaxAxisZero.getShape());
        assertEquals(3f, transposedMaxAxisZero.getFloat(0), 1e-6f);
        assertEquals(6f, transposedMaxAxisZero.getFloat(1), 1e-6f);
    }

    @Test
    void vectorizedExpTrigAndSqrtOpsProduceCorrectResults() {
        NDArray sqrtInput = JNum.from(new float[]{1f, 4f, 9f, 16f}, 4);
        NDArray sinInput = JNum.from(new double[]{0.0, Math.PI / 2.0}, 2);
        NDArray expInput = JNum.from(new double[]{0.0, 1.0}, 2);

        NDArray sqrtResult = sqrtInput.sqrt();
        NDArray sinResult = sinInput.sin();
        NDArray expResult = expInput.exp();

        assertEquals(1f, sqrtResult.getFloat(0), 1e-6f);
        assertEquals(2f, sqrtResult.getFloat(1), 1e-6f);
        assertEquals(3f, sqrtResult.getFloat(2), 1e-6f);
        assertEquals(4f, sqrtResult.getFloat(3), 1e-6f);

        assertEquals(0.0, sinResult.get(0), 1e-9);
        assertEquals(1.0, sinResult.get(1), 1e-9);

        assertEquals(1.0, expResult.get(0), 1e-9);
        assertEquals(Math.E, expResult.get(1), 1e-9);
    }

    @Test
    void booleanLifecycleAndOperationsWorkCorrectly() {
        NDArray boolArr = JNum.from(new boolean[]{true, false, true, false}, 2, 2);

        assertEquals(DType.bool, boolArr.getDType());
        assertTrue(boolArr.getBoolean(0, 0));
        assertFalse(boolArr.getBoolean(0, 1));
        assertTrue(boolArr.getFlatBoolean(2));

        // toString() check
        String str = boolArr.toString();
        assertTrue(str.contains("true") && str.contains("false"));

        // copy() & contiguous() check
        NDArray transposedBool = boolArr.transpose();
        assertFalse(transposedBool.isContiguous());
        NDArray contiguousBool = transposedBool.contiguous();
        assertTrue(contiguousBool.isContiguous());
        assertEquals(DType.bool, contiguousBool.getDType());
        assertTrue(contiguousBool.getBoolean(0, 1)); // row 0 col 1 in transposed is row 1 col 0 in orig (true)

        NDArray copyBool = boolArr.copy();
        assertEquals(boolArr, copyBool);
        assertEquals(boolArr.hashCode(), copyBool.hashCode());

        // cast() check
        NDArray castInt = boolArr.cast(DType.i32);
        assertEquals(DType.i32, castInt.getDType());
        assertEquals(1, castInt.getInt(0, 0));
        assertEquals(0, castInt.getInt(0, 1));

        // Boolean operations check (and, or, xor)
        NDArray b2 = JNum.from(new boolean[]{true, true, false, false}, 2, 2);
        NDArray andRes = boolArr.and(b2);
        assertTrue(andRes.getBoolean(0, 0));
        assertFalse(andRes.getBoolean(0, 1));

        NDArray orRes = boolArr.or(b2);
        assertTrue(orRes.getBoolean(0, 0));
        assertTrue(orRes.getBoolean(0, 1));

        NDArray xorRes = boolArr.xor(b2);
        assertFalse(xorRes.getBoolean(0, 0));
        assertTrue(xorRes.getBoolean(0, 1));
    }

    @Test
    void privateConstructorThrowsAssertionError() throws Exception {
        Constructor<NDArray> constructor = NDArray.class.getDeclaredConstructor();
        constructor.setAccessible(true);
        InvocationTargetException ex = assertThrows(InvocationTargetException.class, constructor::newInstance);
        assertInstanceOf(AssertionError.class, ex.getCause());
    }

    @Test
    void subviewSlicingAndBoundsChecks() {
        NDArray arr = JNum.from(new float[]{
            1f, 2f, 3f,
            4f, 5f, 6f
        }, 2, 3);

        NDArray row0 = arr.subview(0);
        assertArrayEquals(new long[]{3}, row0.getShape());
        assertEquals(1f, row0.getFloat(0));
        assertEquals(2f, row0.getFloat(1));
        assertEquals(3f, row0.getFloat(2));

        NDArray row1 = arr.subview(-1); // negative index
        assertArrayEquals(new long[]{3}, row1.getShape());
        assertEquals(4f, row1.getFloat(0));

        assertThrows(IndexOutOfBoundsException.class, () -> arr.subview(2));
        assertThrows(IndexOutOfBoundsException.class, () -> arr.subview(-3));
    }

    @Test
    void multiAxisSliceWorksCorrectly() {
        NDArray arr = JNum.from(new float[]{
            10f, 20f, 30f, 40f,
            50f, 60f, 70f, 80f,
            90f, 100f, 110f, 120f
        }, 3, 4);

        NDArray sliced = arr.slice(Slice.range(0, 2), Slice.range(1, 3));
        assertArrayEquals(new long[]{2, 2}, sliced.getShape());
        assertEquals(20f, sliced.getFloat(0, 0));
        assertEquals(30f, sliced.getFloat(0, 1));
        assertEquals(60f, sliced.getFloat(1, 0));
        assertEquals(70f, sliced.getFloat(1, 1));

        // Too many slices throws
        assertThrows(IllegalArgumentException.class,
            () -> arr.slice(Slice.all(), Slice.all(), Slice.all()));
    }

    @Test
    void reshapeValidatesTotalElementCount() {
        NDArray arr = JNum.zeros(2, 3);
        NDArray reshaped = arr.reshape(3, 2);
        assertArrayEquals(new long[]{3, 2}, reshaped.getShape());

        NDArray flat = arr.reshape(6);
        assertArrayEquals(new long[]{6}, flat.getShape());

        assertThrows(IllegalArgumentException.class, () -> arr.reshape(5));
        assertThrows(IllegalArgumentException.class, () -> arr.reshape(2, 4));
    }

    @Test
    void inPlaceArithmeticMutatesUnderlyingBuffer() {
        NDArray a = JNum.from(new float[]{1f, 2f, 3f}, 3);
        NDArray b = JNum.from(new float[]{10f, 20f, 30f}, 3);

        a.addi(b);
        assertEquals(11f, a.getFloat(0));
        assertEquals(22f, a.getFloat(1));
        assertEquals(33f, a.getFloat(2));

        a.subi(1.0f);
        assertEquals(10f, a.getFloat(0));
        assertEquals(21f, a.getFloat(1));
        assertEquals(32f, a.getFloat(2));

        a.muli(2.0f);
        assertEquals(20f, a.getFloat(0));
        assertEquals(42f, a.getFloat(1));
        assertEquals(64f, a.getFloat(2));

        a.divi(2.0f);
        assertEquals(10f, a.getFloat(0));
        assertEquals(21f, a.getFloat(1));
        assertEquals(32f, a.getFloat(2));
    }

    @Test
    void closedArenaThrowsIllegalStateExceptionOnAccess() {
        NDArray arr;
        try (Arena confined = Arena.ofConfined()) {
            arr = JNum.zeros(confined, DType.f32, 2, 2);
            assertEquals(0.0f, arr.getFloat(0, 0));
        }
        // Scope is now closed -> segment access must throw IllegalStateException
        assertThrows(IllegalStateException.class, () -> arr.getFloat(0, 0));
        assertThrows(IllegalStateException.class, () -> arr.setFloat(5.0f, 0, 0));
    }
}
