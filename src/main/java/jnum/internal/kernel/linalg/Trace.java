package jnum.internal.kernel.linalg;

import static jnum.DType.*;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import jnum.NDArray;

public final class Trace {

    private Trace() {
        throw new AssertionError("Trace kernel cannot be instantiated.");
    }

    public static double compute(NDArray a, int offset) {
        if (a.dim() != 2) {
            throw new IllegalArgumentException("Trace requires a 2D matrix, got shape: " + a.shapeString());
        }

        long[] shape = a.internalShapeUnsafe();
        long rows = shape[0];
        long cols = shape[1];

        long startRow = offset < 0 ? -offset : 0;
        long startCol = offset > 0 ? offset : 0;

        if (startRow >= rows || startCol >= cols) {
            return 0.0;
        }

        long diagLen = Math.min(rows - startRow, cols - startCol);
        long[] strides = a.internalStridesUnsafe();
        long s0 = strides[0];
        long s1 = strides[1];
        MemorySegment seg = a.getData();

        double sum = 0.0;
        long baseOffset = startRow * s0 + startCol * s1;
        long step = s0 + s1;

        switch (a.getDType()) {
            case f32 -> {
                for (long i = 0; i < diagLen; i++) {
                    sum += seg.getAtIndex(ValueLayout.JAVA_FLOAT, baseOffset + i * step);
                }
            }
            case f64 -> {
                for (long i = 0; i < diagLen; i++) {
                    sum += seg.getAtIndex(ValueLayout.JAVA_DOUBLE, baseOffset + i * step);
                }
            }
            case i32 -> {
                for (long i = 0; i < diagLen; i++) {
                    sum += seg.getAtIndex(ValueLayout.JAVA_INT, baseOffset + i * step);
                }
            }
            case bool -> {
                // In boolean matrices, trace is the count of true values along the diagonal
                long count = 0;
                for (long i = 0; i < diagLen; i++) {
                    if (seg.getAtIndex(ValueLayout.JAVA_BYTE, baseOffset + i * step) != 0) {
                        count++;
                    }
                }
                return (double) count;
            }
        }
        return sum;
    }
}