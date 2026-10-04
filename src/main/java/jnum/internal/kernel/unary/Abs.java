package jnum.internal.kernel.unary;

import static jnum.internal.Constants.*;

import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorSpecies;
import jdk.incubator.vector.VectorOperators;
import jnum.NDArray;
import jnum.internal.layout.NDIter;

public final class Abs {

    private Abs() {
        throw new AssertionError();
    }

    public static NDArray absFloat(NDArray a, NDArray resArray) {
        if (a.isContiguous() && resArray.isContiguous()) {
            long i = 0;
            long loopbound = a.getSize() - (a.getSize() % (VL_F32 * 2));
                         
            for (; i < loopbound; i += VL_F32 * 2) {
                var v1 = FloatVector.fromMemorySegment(SPECIES_F32, a.getData(), i * BYTES_F32, NATIVE_ORDER);
                var v2 = FloatVector.fromMemorySegment(SPECIES_F32, a.getData(), (i + VL_F32) * BYTES_F32, NATIVE_ORDER);
                var VRes1 = v1.lanewise(VectorOperators.ABS);
                var VRes2 = v2.lanewise(VectorOperators.ABS);
                VRes1.intoMemorySegment(resArray.getData(), i * BYTES_F32, NATIVE_ORDER);
                VRes2.intoMemorySegment(resArray.getData(), (i + VL_F32) * BYTES_F32, NATIVE_ORDER);
            }
            loopbound = SPECIES_F32.loopBound(a.getSize());
            for (; i < loopbound; i += VL_F32) {
                var v = FloatVector.fromMemorySegment(SPECIES_F32, a.getData(), i * BYTES_F32, NATIVE_ORDER);
                var VRes = v.lanewise(VectorOperators.ABS);
                VRes.intoMemorySegment(resArray.getData(), i * BYTES_F32, NATIVE_ORDER);
            }
            for (; i < a.getSize(); i++) {
                float val = a.getData().getAtIndex(ValueLayout.JAVA_FLOAT, i);
                resArray.getData().setAtIndex(ValueLayout.JAVA_FLOAT, i, (float) Math.abs(val));
            }
                     
        } else {
            NDIter iterA = new NDIter(resArray.internalShapeUnsafe(), a.internalStridesUnsafe());
            NDIter iterRes = new NDIter(resArray.internalShapeUnsafe(), resArray.internalStridesUnsafe());
                         
            while (iterA.hasNext) {
                float val = a.getData().getAtIndex(ValueLayout.JAVA_FLOAT, iterA.offset);
                resArray.getData().setAtIndex(ValueLayout.JAVA_FLOAT, iterRes.offset, (float) Math.abs(val));
                iterA.next();
                iterRes.next();
            }
        }
        return resArray;
    }


    public static NDArray absDouble(NDArray a, NDArray resArray) {
        if (a.isContiguous() && resArray.isContiguous()) {
            long i = 0;
            long loopbound = a.getSize() - (a.getSize() % (VL_F64 * 2));
                         
            for (; i < loopbound; i += VL_F64 * 2) {
                var v1 = DoubleVector.fromMemorySegment(SPECIES_F64, a.getData(), i * BYTES_F64, NATIVE_ORDER);
                var v2 = DoubleVector.fromMemorySegment(SPECIES_F64, a.getData(), (i + VL_F64) * BYTES_F64, NATIVE_ORDER);
                var VRes1 = v1.lanewise(VectorOperators.ABS);
                var VRes2 = v2.lanewise(VectorOperators.ABS);
                VRes1.intoMemorySegment(resArray.getData(), i * BYTES_F64, NATIVE_ORDER);
                VRes2.intoMemorySegment(resArray.getData(), (i + VL_F64) * BYTES_F64, NATIVE_ORDER);
            }
            loopbound = SPECIES_F64.loopBound(a.getSize());
            for (; i < loopbound; i += VL_F64) {
                var v = DoubleVector.fromMemorySegment(SPECIES_F64, a.getData(), i * BYTES_F64, NATIVE_ORDER);
                var VRes = v.lanewise(VectorOperators.ABS);
                VRes.intoMemorySegment(resArray.getData(), i * BYTES_F64, NATIVE_ORDER);
            }
            for (; i < a.getSize(); i++) {
                double val = a.getData().getAtIndex(ValueLayout.JAVA_DOUBLE, i);
                resArray.getData().setAtIndex(ValueLayout.JAVA_DOUBLE, i, Math.abs(val));
            }
                     
        } else {
            NDIter iterA = new NDIter(resArray.internalShapeUnsafe(), a.internalStridesUnsafe());
            NDIter iterRes = new NDIter(resArray.internalShapeUnsafe(), resArray.internalStridesUnsafe());
                         
            while (iterA.hasNext) {
                double val = a.getData().getAtIndex(ValueLayout.JAVA_DOUBLE, iterA.offset);
                resArray.getData().setAtIndex(ValueLayout.JAVA_DOUBLE, iterRes.offset, Math.abs(val));
                iterA.next();
                iterRes.next();
            }
        }
        return resArray;
    }


    public static NDArray absInt(NDArray a, NDArray resArray) {
        if (a.isContiguous() && resArray.isContiguous()) {
            long i = 0;
            long loopbound = a.getSize() - (a.getSize() % (VL_I32 * 2));
                         
            for (; i < loopbound; i += VL_I32 * 2) {
                var v1 = IntVector.fromMemorySegment(SPECIES_I32, a.getData(), i * BYTES_I32, NATIVE_ORDER);
                var v2 = IntVector.fromMemorySegment(SPECIES_I32, a.getData(), (i + VL_I32) * BYTES_I32, NATIVE_ORDER);
                var VRes1 = v1.lanewise(VectorOperators.ABS);
                var VRes2 = v2.lanewise(VectorOperators.ABS);
                VRes1.intoMemorySegment(resArray.getData(), i * BYTES_I32, NATIVE_ORDER);
                VRes2.intoMemorySegment(resArray.getData(), (i + VL_I32) * BYTES_I32, NATIVE_ORDER);
            }
            loopbound = SPECIES_I32.loopBound(a.getSize());
            for (; i < loopbound; i += VL_I32) {
                var v = IntVector.fromMemorySegment(SPECIES_I32, a.getData(), i * BYTES_I32, NATIVE_ORDER);
                var VRes = v.lanewise(VectorOperators.ABS);
                VRes.intoMemorySegment(resArray.getData(), i * BYTES_I32, NATIVE_ORDER);
            }
            for (; i < a.getSize(); i++) {
                int val = a.getData().getAtIndex(ValueLayout.JAVA_INT, i);
                resArray.getData().setAtIndex(ValueLayout.JAVA_INT, i, Math.abs(val));
            }
                     
        } else {
            NDIter iterA = new NDIter(resArray.internalShapeUnsafe(), a.internalStridesUnsafe());
            NDIter iterRes = new NDIter(resArray.internalShapeUnsafe(), resArray.internalStridesUnsafe());
                         
            while (iterA.hasNext) {
                int val = a.getData().getAtIndex(ValueLayout.JAVA_INT, iterA.offset);
                resArray.getData().setAtIndex(ValueLayout.JAVA_INT, iterRes.offset, Math.abs(val));
                iterA.next();
                iterRes.next();
            }
        }
        return resArray;
    }
}
