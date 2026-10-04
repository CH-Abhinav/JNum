package jnum.internal.kernel.arithmetic;

import static jnum.internal.Constants.*;

import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;
import jnum.NDArray;
import jnum.internal.layout.NDIter;

public final class Sub {

    private Sub() {
        throw new AssertionError();
    }

    public static NDArray subFloat(NDArray a, NDArray b, NDArray resArray) {
        if (a.isContiguous() && b.isContiguous() && resArray.isContiguous()) {
            long i = 0;
            long loopbound = a.getSize() - (a.getSize() % (VL_F32 * 2));
                         
            for (; i < loopbound; i += VL_F32 * 2) {
                var vA1 = FloatVector.fromMemorySegment(SPECIES_F32, a.getData(), i * BYTES_F32, NATIVE_ORDER);
                var vA2 = FloatVector.fromMemorySegment(SPECIES_F32, a.getData(), (i + VL_F32) * BYTES_F32, NATIVE_ORDER);
                var vB1 = FloatVector.fromMemorySegment(SPECIES_F32, b.getData(), i * BYTES_F32, NATIVE_ORDER);
                var vB2 = FloatVector.fromMemorySegment(SPECIES_F32, b.getData(), (i + VL_F32) * BYTES_F32, NATIVE_ORDER);
                                 
                var VRes1 = vA1.sub(vB1);
                var VRes2 = vA2.sub(vB2);
                                 
                VRes1.intoMemorySegment(resArray.getData(), i * BYTES_F32, NATIVE_ORDER);
                VRes2.intoMemorySegment(resArray.getData(), (i + VL_F32) * BYTES_F32, NATIVE_ORDER);
            }
            loopbound = SPECIES_F32.loopBound(a.getSize());
            for (; i < loopbound; i += VL_F32) {
                var vA = FloatVector.fromMemorySegment(SPECIES_F32, a.getData(), i * BYTES_F32, NATIVE_ORDER);
                var vB = FloatVector.fromMemorySegment(SPECIES_F32, b.getData(), i * BYTES_F32, NATIVE_ORDER);
                var VRes = vA.sub(vB);
                VRes.intoMemorySegment(resArray.getData(), i * BYTES_F32, NATIVE_ORDER);
            }
            for (; i < a.getSize(); i++) {
                float valA = a.getData().getAtIndex(ValueLayout.JAVA_FLOAT, i);
                float valB = b.getData().getAtIndex(ValueLayout.JAVA_FLOAT, i);
                resArray.getData().setAtIndex(ValueLayout.JAVA_FLOAT, i, (float)(valA - valB));
            }
        } else {
            NDIter iterA = new NDIter(resArray.internalShapeUnsafe(), a.internalStridesUnsafe());
            NDIter iterB = new NDIter(resArray.internalShapeUnsafe(), b.internalStridesUnsafe());
            NDIter iterRes = new NDIter(resArray.internalShapeUnsafe(), resArray.internalStridesUnsafe());
            while (iterA.hasNext) {
                float valA = a.getData().getAtIndex(ValueLayout.JAVA_FLOAT, iterA.offset);
                float valB = b.getData().getAtIndex(ValueLayout.JAVA_FLOAT, iterB.offset);
                resArray.getData().setAtIndex(ValueLayout.JAVA_FLOAT, iterRes.offset, (float)(valA - valB));
                iterA.next();
                iterB.next();
                iterRes.next();
            }
        }
        return resArray;
    }

    public static NDArray subFloat(NDArray a, float b, NDArray resArray) {
        var vB = FloatVector.broadcast(SPECIES_F32, b);
                 
        if (a.isContiguous() && resArray.isContiguous()) {
            long i = 0;
            long loopbound = a.getSize() - (a.getSize() % (VL_F32 * 2));
                         
            for (; i < loopbound; i += VL_F32 * 2) {
                var vA1 = FloatVector.fromMemorySegment(SPECIES_F32, a.getData(), i * BYTES_F32, NATIVE_ORDER);
                var vA2 = FloatVector.fromMemorySegment(SPECIES_F32, a.getData(), (i + VL_F32) * BYTES_F32, NATIVE_ORDER);
                var VRes1 = vA1.sub(vB);
                var VRes2 = vA2.sub(vB);
                                 
                VRes1.intoMemorySegment(resArray.getData(), i * BYTES_F32, NATIVE_ORDER);
                VRes2.intoMemorySegment(resArray.getData(), (i + VL_F32) * BYTES_F32, NATIVE_ORDER);
            }
            loopbound = SPECIES_F32.loopBound(a.getSize());
            for (; i < loopbound; i += VL_F32) {
                var vA = FloatVector.fromMemorySegment(SPECIES_F32, a.getData(), i * BYTES_F32, NATIVE_ORDER);
                var VRes = vA.sub(vB);
                VRes.intoMemorySegment(resArray.getData(), i * BYTES_F32, NATIVE_ORDER);
            }
            for (; i < a.getSize(); i++) {
                float valA = a.getData().getAtIndex(ValueLayout.JAVA_FLOAT, i);
                resArray.getData().setAtIndex(ValueLayout.JAVA_FLOAT, i, (float)(valA - b));
            }
        } else {
            NDIter iterA = new NDIter(resArray.internalShapeUnsafe(), a.internalStridesUnsafe());
            NDIter iterRes = new NDIter(resArray.internalShapeUnsafe(), resArray.internalStridesUnsafe());
            while (iterA.hasNext) {
                float valA = a.getData().getAtIndex(ValueLayout.JAVA_FLOAT, iterA.offset);
                resArray.getData().setAtIndex(ValueLayout.JAVA_FLOAT, iterRes.offset, (float)(valA - b));
                iterA.next();
                iterRes.next();
            }
        }
        return resArray;
    }

    public static NDArray subDouble(NDArray a, NDArray b, NDArray resArray) {
        if (a.isContiguous() && b.isContiguous() && resArray.isContiguous()) {
            long i = 0;
            long loopbound = a.getSize() - (a.getSize() % (VL_F64 * 2));
                         
            for (; i < loopbound; i += VL_F64 * 2) {
                var vA1 = DoubleVector.fromMemorySegment(SPECIES_F64, a.getData(), i * BYTES_F64, NATIVE_ORDER);
                var vA2 = DoubleVector.fromMemorySegment(SPECIES_F64, a.getData(), (i + VL_F64) * BYTES_F64, NATIVE_ORDER);
                var vB1 = DoubleVector.fromMemorySegment(SPECIES_F64, b.getData(), i * BYTES_F64, NATIVE_ORDER);
                var vB2 = DoubleVector.fromMemorySegment(SPECIES_F64, b.getData(), (i + VL_F64) * BYTES_F64, NATIVE_ORDER);
                                 
                var VRes1 = vA1.sub(vB1);
                var VRes2 = vA2.sub(vB2);
                                 
                VRes1.intoMemorySegment(resArray.getData(), i * BYTES_F64, NATIVE_ORDER);
                VRes2.intoMemorySegment(resArray.getData(), (i + VL_F64) * BYTES_F64, NATIVE_ORDER);
            }
            loopbound = SPECIES_F64.loopBound(a.getSize());
            for (; i < loopbound; i += VL_F64) {
                var vA = DoubleVector.fromMemorySegment(SPECIES_F64, a.getData(), i * BYTES_F64, NATIVE_ORDER);
                var vB = DoubleVector.fromMemorySegment(SPECIES_F64, b.getData(), i * BYTES_F64, NATIVE_ORDER);
                var VRes = vA.sub(vB);
                VRes.intoMemorySegment(resArray.getData(), i * BYTES_F64, NATIVE_ORDER);
            }
            for (; i < a.getSize(); i++) {
                double valA = a.getData().getAtIndex(ValueLayout.JAVA_DOUBLE, i);
                double valB = b.getData().getAtIndex(ValueLayout.JAVA_DOUBLE, i);
                resArray.getData().setAtIndex(ValueLayout.JAVA_DOUBLE, i, (double)(valA - valB));
            }
        } else {
            NDIter iterA = new NDIter(resArray.internalShapeUnsafe(), a.internalStridesUnsafe());
            NDIter iterB = new NDIter(resArray.internalShapeUnsafe(), b.internalStridesUnsafe());
            NDIter iterRes = new NDIter(resArray.internalShapeUnsafe(), resArray.internalStridesUnsafe());
            while (iterA.hasNext) {
                double valA = a.getData().getAtIndex(ValueLayout.JAVA_DOUBLE, iterA.offset);
                double valB = b.getData().getAtIndex(ValueLayout.JAVA_DOUBLE, iterB.offset);
                resArray.getData().setAtIndex(ValueLayout.JAVA_DOUBLE, iterRes.offset, (double)(valA - valB));
                iterA.next();
                iterB.next();
                iterRes.next();
            }
        }
        return resArray;
    }

    public static NDArray subDouble(NDArray a, double b, NDArray resArray) {
        var vB = DoubleVector.broadcast(SPECIES_F64, b);
                 
        if (a.isContiguous() && resArray.isContiguous()) {
            long i = 0;
            long loopbound = a.getSize() - (a.getSize() % (VL_F64 * 2));
                         
            for (; i < loopbound; i += VL_F64 * 2) {
                var vA1 = DoubleVector.fromMemorySegment(SPECIES_F64, a.getData(), i * BYTES_F64, NATIVE_ORDER);
                var vA2 = DoubleVector.fromMemorySegment(SPECIES_F64, a.getData(), (i + VL_F64) * BYTES_F64, NATIVE_ORDER);
                var VRes1 = vA1.sub(vB);
                var VRes2 = vA2.sub(vB);
                                 
                VRes1.intoMemorySegment(resArray.getData(), i * BYTES_F64, NATIVE_ORDER);
                VRes2.intoMemorySegment(resArray.getData(), (i + VL_F64) * BYTES_F64, NATIVE_ORDER);
            }
            loopbound = SPECIES_F64.loopBound(a.getSize());
            for (; i < loopbound; i += VL_F64) {
                var vA = DoubleVector.fromMemorySegment(SPECIES_F64, a.getData(), i * BYTES_F64, NATIVE_ORDER);
                var VRes = vA.sub(vB);
                VRes.intoMemorySegment(resArray.getData(), i * BYTES_F64, NATIVE_ORDER);
            }
            for (; i < a.getSize(); i++) {
                double valA = a.getData().getAtIndex(ValueLayout.JAVA_DOUBLE, i);
                resArray.getData().setAtIndex(ValueLayout.JAVA_DOUBLE, i, (double)(valA - b));
            }
        } else {
            NDIter iterA = new NDIter(resArray.internalShapeUnsafe(), a.internalStridesUnsafe());
            NDIter iterRes = new NDIter(resArray.internalShapeUnsafe(), resArray.internalStridesUnsafe());
            while (iterA.hasNext) {
                double valA = a.getData().getAtIndex(ValueLayout.JAVA_DOUBLE, iterA.offset);
                resArray.getData().setAtIndex(ValueLayout.JAVA_DOUBLE, iterRes.offset, (double)(valA - b));
                iterA.next();
                iterRes.next();
            }
        }
        return resArray;
    }

    public static NDArray subInt(NDArray a, NDArray b, NDArray resArray) {
        if (a.isContiguous() && b.isContiguous() && resArray.isContiguous()) {
            long i = 0;
            long loopbound = a.getSize() - (a.getSize() % (VL_I32 * 2));
                         
            for (; i < loopbound; i += VL_I32 * 2) {
                var vA1 = IntVector.fromMemorySegment(SPECIES_I32, a.getData(), i * BYTES_I32, NATIVE_ORDER);
                var vA2 = IntVector.fromMemorySegment(SPECIES_I32, a.getData(), (i + VL_I32) * BYTES_I32, NATIVE_ORDER);
                var vB1 = IntVector.fromMemorySegment(SPECIES_I32, b.getData(), i * BYTES_I32, NATIVE_ORDER);
                var vB2 = IntVector.fromMemorySegment(SPECIES_I32, b.getData(), (i + VL_I32) * BYTES_I32, NATIVE_ORDER);
                                 
                var VRes1 = vA1.sub(vB1);
                var VRes2 = vA2.sub(vB2);
                                 
                VRes1.intoMemorySegment(resArray.getData(), i * BYTES_I32, NATIVE_ORDER);
                VRes2.intoMemorySegment(resArray.getData(), (i + VL_I32) * BYTES_I32, NATIVE_ORDER);
            }
            loopbound = SPECIES_I32.loopBound(a.getSize());
            for (; i < loopbound; i += VL_I32) {
                var vA = IntVector.fromMemorySegment(SPECIES_I32, a.getData(), i * BYTES_I32, NATIVE_ORDER);
                var vB = IntVector.fromMemorySegment(SPECIES_I32, b.getData(), i * BYTES_I32, NATIVE_ORDER);
                var VRes = vA.sub(vB);
                VRes.intoMemorySegment(resArray.getData(), i * BYTES_I32, NATIVE_ORDER);
            }
            for (; i < a.getSize(); i++) {
                int valA = a.getData().getAtIndex(ValueLayout.JAVA_INT, i);
                int valB = b.getData().getAtIndex(ValueLayout.JAVA_INT, i);
                resArray.getData().setAtIndex(ValueLayout.JAVA_INT, i, (int)(valA - valB));
            }
        } else {
            NDIter iterA = new NDIter(resArray.internalShapeUnsafe(), a.internalStridesUnsafe());
            NDIter iterB = new NDIter(resArray.internalShapeUnsafe(), b.internalStridesUnsafe());
            NDIter iterRes = new NDIter(resArray.internalShapeUnsafe(), resArray.internalStridesUnsafe());
            while (iterA.hasNext) {
                int valA = a.getData().getAtIndex(ValueLayout.JAVA_INT, iterA.offset);
                int valB = b.getData().getAtIndex(ValueLayout.JAVA_INT, iterB.offset);
                resArray.getData().setAtIndex(ValueLayout.JAVA_INT, iterRes.offset, (int)(valA - valB));
                iterA.next();
                iterB.next();
                iterRes.next();
            }
        }
        return resArray;
    }

    public static NDArray subInt(NDArray a, int b, NDArray resArray) {
        var vB = IntVector.broadcast(SPECIES_I32, b);
                 
        if (a.isContiguous() && resArray.isContiguous()) {
            long i = 0;
            long loopbound = a.getSize() - (a.getSize() % (VL_I32 * 2));
                         
            for (; i < loopbound; i += VL_I32 * 2) {
                var vA1 = IntVector.fromMemorySegment(SPECIES_I32, a.getData(), i * BYTES_I32, NATIVE_ORDER);
                var vA2 = IntVector.fromMemorySegment(SPECIES_I32, a.getData(), (i + VL_I32) * BYTES_I32, NATIVE_ORDER);
                var VRes1 = vA1.sub(vB);
                var VRes2 = vA2.sub(vB);
                                 
                VRes1.intoMemorySegment(resArray.getData(), i * BYTES_I32, NATIVE_ORDER);
                VRes2.intoMemorySegment(resArray.getData(), (i + VL_I32) * BYTES_I32, NATIVE_ORDER);
            }
            loopbound = SPECIES_I32.loopBound(a.getSize());
            for (; i < loopbound; i += VL_I32) {
                var vA = IntVector.fromMemorySegment(SPECIES_I32, a.getData(), i * BYTES_I32, NATIVE_ORDER);
                var VRes = vA.sub(vB);
                VRes.intoMemorySegment(resArray.getData(), i * BYTES_I32, NATIVE_ORDER);
            }
            for (; i < a.getSize(); i++) {
                int valA = a.getData().getAtIndex(ValueLayout.JAVA_INT, i);
                resArray.getData().setAtIndex(ValueLayout.JAVA_INT, i, (int)(valA - b));
            }
        } else {
            NDIter iterA = new NDIter(resArray.internalShapeUnsafe(), a.internalStridesUnsafe());
            NDIter iterRes = new NDIter(resArray.internalShapeUnsafe(), resArray.internalStridesUnsafe());
            while (iterA.hasNext) {
                int valA = a.getData().getAtIndex(ValueLayout.JAVA_INT, iterA.offset);
                resArray.getData().setAtIndex(ValueLayout.JAVA_INT, iterRes.offset, (int)(valA - b));
                iterA.next();
                iterRes.next();
            }
        }
        return resArray;
    }


}
