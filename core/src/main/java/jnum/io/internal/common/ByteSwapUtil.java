package jnum.io.internal.common;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import jnum.DType;

/**
 * Endianness translation kernels directly operating on Panama MemorySegments.
 */
public final class ByteSwapUtil {

    private ByteSwapUtil() {
        throw new AssertionError("ByteSwapUtil cannot be instantiated.");
    }

    public static void swapInPlace(MemorySegment segment, DType dtype, long elementCount) {
        switch (dtype) {
            case i32 -> {
                for (long i = 0; i < elementCount; i++) {
                    int val = segment.getAtIndex(ValueLayout.JAVA_INT, i);
                    segment.setAtIndex(ValueLayout.JAVA_INT, i, Integer.reverseBytes(val));
                }
            }
            case f32 -> {
                // An IEEE 754 32-bit float consists of 4 bytes (same size as JAVA_INT).
                // Reversing the raw 32-bit integer word via Integer.reverseBytes() swaps the 4 bytes
                // at the bit level without triggering FPU NaN-canonicalization.
                for (long i = 0; i < elementCount; i++) {
                    int bits = segment.getAtIndex(ValueLayout.JAVA_INT, i);
                    segment.setAtIndex(ValueLayout.JAVA_INT, i, Integer.reverseBytes(bits));
                }
            }
            case f64 -> {
                // Similarly, a 64-bit double consists of 8 bytes (same size as JAVA_LONG).
                // Reversing the raw 64-bit integer word via Long.reverseBytes() swaps the 8 bytes.
                for (long i = 0; i < elementCount; i++) {
                    long bits = segment.getAtIndex(ValueLayout.JAVA_LONG, i);
                    segment.setAtIndex(ValueLayout.JAVA_LONG, i, Long.reverseBytes(bits));
                }
            }
            case bool -> {
                // 1-byte booleans do not have an endianness distinction
            }
        }
    }
}