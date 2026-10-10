package jnum.io.internal.common;

import static org.junit.jupiter.api.Assertions.*;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import jnum.DType;
import org.junit.jupiter.api.Test;

class ByteSwapUtilTest {

    @Test
    void testPrivateConstructor() throws Exception {
        Constructor<ByteSwapUtil> constructor = ByteSwapUtil.class.getDeclaredConstructor();
        constructor.setAccessible(true);
        InvocationTargetException ex = assertThrows(InvocationTargetException.class, constructor::newInstance);
        assertInstanceOf(AssertionError.class, ex.getCause());
    }

    @Test
    void testSwapInPlaceInt32() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment seg = arena.allocate(ValueLayout.JAVA_INT, 4);
            int[] vals = new int[]{0x12345678, 0, -1, 0x01020304};
            for (int i = 0; i < vals.length; i++) {
                seg.setAtIndex(ValueLayout.JAVA_INT, i, vals[i]);
            }

            ByteSwapUtil.swapInPlace(seg, DType.i32, 4);
            assertEquals(0x78563412, seg.getAtIndex(ValueLayout.JAVA_INT, 0));
            assertEquals(0, seg.getAtIndex(ValueLayout.JAVA_INT, 1));
            assertEquals(-1, seg.getAtIndex(ValueLayout.JAVA_INT, 2));
            assertEquals(0x04030201, seg.getAtIndex(ValueLayout.JAVA_INT, 3));

            // Idempotence: swap again recovers original
            ByteSwapUtil.swapInPlace(seg, DType.i32, 4);
            for (int i = 0; i < vals.length; i++) {
                assertEquals(vals[i], seg.getAtIndex(ValueLayout.JAVA_INT, i));
            }
        }
    }

    @Test
    void testSwapInPlaceFloat32() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment seg = arena.allocate(ValueLayout.JAVA_FLOAT, 5);
            float[] vals = new float[]{1.0f, -2.5f, 0.0f, Float.NaN, Float.POSITIVE_INFINITY};
            for (int i = 0; i < vals.length; i++) {
                seg.setAtIndex(ValueLayout.JAVA_FLOAT, i, vals[i]);
            }

            // Swap once
            ByteSwapUtil.swapInPlace(seg, DType.f32, 5);

            // Swap twice to verify exact bit-level reversibility (including NaNs)
            ByteSwapUtil.swapInPlace(seg, DType.f32, 5);
            for (int i = 0; i < vals.length; i++) {
                if (Float.isNaN(vals[i])) {
                    assertTrue(Float.isNaN(seg.getAtIndex(ValueLayout.JAVA_FLOAT, i)));
                } else {
                    assertEquals(vals[i], seg.getAtIndex(ValueLayout.JAVA_FLOAT, i));
                }
            }
        }
    }

    @Test
    void testSwapInPlaceFloat64() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment seg = arena.allocate(ValueLayout.JAVA_DOUBLE, 4);
            double[] vals = new double[]{1.23456789, -987.654, Double.NaN, Double.NEGATIVE_INFINITY};
            for (int i = 0; i < vals.length; i++) {
                seg.setAtIndex(ValueLayout.JAVA_DOUBLE, i, vals[i]);
            }

            // Swap twice to verify reversibility
            ByteSwapUtil.swapInPlace(seg, DType.f64, 4);
            ByteSwapUtil.swapInPlace(seg, DType.f64, 4);
            for (int i = 0; i < vals.length; i++) {
                if (Double.isNaN(vals[i])) {
                    assertTrue(Double.isNaN(seg.getAtIndex(ValueLayout.JAVA_DOUBLE, i)));
                } else {
                    assertEquals(vals[i], seg.getAtIndex(ValueLayout.JAVA_DOUBLE, i));
                }
            }
        }
    }

    @Test
    void testSwapInPlaceBoolIsNoOp() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment seg = arena.allocate(ValueLayout.JAVA_BYTE, 3);
            seg.setAtIndex(ValueLayout.JAVA_BYTE, 0, (byte) 1);
            seg.setAtIndex(ValueLayout.JAVA_BYTE, 1, (byte) 0);
            seg.setAtIndex(ValueLayout.JAVA_BYTE, 2, (byte) 1);

            ByteSwapUtil.swapInPlace(seg, DType.bool, 3);

            assertEquals((byte) 1, seg.getAtIndex(ValueLayout.JAVA_BYTE, 0));
            assertEquals((byte) 0, seg.getAtIndex(ValueLayout.JAVA_BYTE, 1));
            assertEquals((byte) 1, seg.getAtIndex(ValueLayout.JAVA_BYTE, 2));
        }
    }
}
