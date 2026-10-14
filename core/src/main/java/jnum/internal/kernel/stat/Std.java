package jnum.internal.kernel.stat;

import static jnum.internal.Constants.*;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jnum.NDArray;

/**
 * Standard deviation kernels: Math.sqrt(var).
 */
public final class Std {

    private Std() {
        throw new AssertionError("Std kernel cannot be instantiated.");
    }

    public static double stdFloat(NDArray a, int ddof) {
        return Math.sqrt(Var.varFloat(a, ddof));
    }

    public static double stdDouble(NDArray a, int ddof) {
        return Math.sqrt(Var.varDouble(a, ddof));
    }

    public static double stdInt(NDArray a, int ddof) {
        return Math.sqrt(Var.varInt(a, ddof));
    }

    public static NDArray stdFloatAxis(NDArray a, int axis, int ddof, NDArray resArray) {
        Var.varFloatAxis(a, axis, ddof, resArray);
        applySqrtFloat(resArray);
        return resArray;
    }

    public static NDArray stdDoubleAxis(NDArray a, int axis, int ddof, NDArray resArray) {
        Var.varDoubleAxis(a, axis, ddof, resArray);
        applySqrtDouble(resArray);
        return resArray;
    }

    public static NDArray stdIntAxis(NDArray a, int axis, int ddof, NDArray resArray) {
        Var.varIntAxis(a, axis, ddof, resArray);
        applySqrtDouble(resArray);
        return resArray;
    }

    private static void applySqrtFloat(NDArray arr) {
        MemorySegment seg = arr.getData();
        long size = arr.getSize();
        long loopbound = SPECIES_F32.loopBound(size);
        long i = 0;
        for (; i < loopbound; i += VL_F32) {
            var v = FloatVector.fromMemorySegment(SPECIES_F32, seg, i * BYTES_F32, NATIVE_ORDER);
            v.sqrt().intoMemorySegment(seg, i * BYTES_F32, NATIVE_ORDER);
        }
        for (; i < size; i++) {
            float val = seg.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            seg.setAtIndex(ValueLayout.JAVA_FLOAT, i, (float) Math.sqrt(val));
        }
    }

    private static void applySqrtDouble(NDArray arr) {
        MemorySegment seg = arr.getData();
        long size = arr.getSize();
        long loopbound = SPECIES_F64.loopBound(size);
        long i = 0;
        for (; i < loopbound; i += VL_F64) {
            var v = DoubleVector.fromMemorySegment(SPECIES_F64, seg, i * BYTES_F64, NATIVE_ORDER);
            v.sqrt().intoMemorySegment(seg, i * BYTES_F64, NATIVE_ORDER);
        }
        for (; i < size; i++) {
            double val = seg.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
            seg.setAtIndex(ValueLayout.JAVA_DOUBLE, i, Math.sqrt(val));
        }
    }
}
