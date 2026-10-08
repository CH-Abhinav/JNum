package jnum.internal.kernel.logical;

import java.lang.foreign.ValueLayout;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.VectorOperators;
import jnum.NDArray;
import jnum.internal.layout.NDIter;

import static jnum.internal.Constants.*;

public final class And {


    private And() {
        throw new AssertionError();
    }

    public static NDArray and(NDArray a, NDArray b, NDArray resArray) {
        if (a.isContiguous() && b.isContiguous() && resArray.isContiguous()) {
            long i = 0;
            long loopbound = a.getSize() - (a.getSize() % (VL_BOOL * 2L));

            for (; i < loopbound; i += VL_BOOL * 2L) {
                var va1 = ByteVector.fromMemorySegment(SPECIES_BOOL, a.getData(), i * BYTES_BOOL, NATIVE_ORDER);
                var va2 = ByteVector.fromMemorySegment(SPECIES_BOOL, a.getData(), (i + VL_BOOL) * BYTES_BOOL, NATIVE_ORDER);
                var vb1 = ByteVector.fromMemorySegment(SPECIES_BOOL, b.getData(), i * BYTES_BOOL, NATIVE_ORDER);
                var vb2 = ByteVector.fromMemorySegment(SPECIES_BOOL, b.getData(), (i + VL_BOOL) * BYTES_BOOL, NATIVE_ORDER);
                var VRes1 = va1.lanewise(VectorOperators.AND, vb1);
                var VRes2 = va2.lanewise(VectorOperators.AND, vb2);
                VRes1.intoMemorySegment(resArray.getData(), i * BYTES_BOOL, NATIVE_ORDER);
                VRes2.intoMemorySegment(resArray.getData(), (i + VL_BOOL) * BYTES_BOOL, NATIVE_ORDER);
            }
            loopbound = SPECIES_BOOL.loopBound(a.getSize());
            for (; i < loopbound; i += VL_BOOL) {
                var va = ByteVector.fromMemorySegment(SPECIES_BOOL, a.getData(), i * BYTES_BOOL, NATIVE_ORDER);
                var vb = ByteVector.fromMemorySegment(SPECIES_BOOL, b.getData(), i * BYTES_BOOL, NATIVE_ORDER);
                var VRes = va.lanewise(VectorOperators.AND, vb);
                VRes.intoMemorySegment(resArray.getData(), i * BYTES_BOOL, NATIVE_ORDER);
            }
            for (; i < a.getSize(); i++) {
                byte valA = a.getData().getAtIndex(ValueLayout.JAVA_BYTE, i);
                byte valB = b.getData().getAtIndex(ValueLayout.JAVA_BYTE, i);
                resArray.getData().setAtIndex(ValueLayout.JAVA_BYTE, i, (byte) (valA & valB));
            }
        } else {
            NDIter iterA = new NDIter(resArray.internalShapeUnsafe(), a.internalStridesUnsafe());
            NDIter iterB = new NDIter(resArray.internalShapeUnsafe(), b.internalStridesUnsafe());
            NDIter iterRes = new NDIter(resArray.internalShapeUnsafe(), resArray.internalStridesUnsafe());

            while (iterA.hasNext) {
                byte valA = a.getData().getAtIndex(ValueLayout.JAVA_BYTE, iterA.offset);
                byte valB = b.getData().getAtIndex(ValueLayout.JAVA_BYTE, iterB.offset);
                resArray.getData().setAtIndex(ValueLayout.JAVA_BYTE, iterRes.offset, (byte) (valA & valB));
                iterA.next();
                iterB.next();
                iterRes.next();
            }
        }
        return resArray;
    }
}
