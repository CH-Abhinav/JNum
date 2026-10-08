package jnum.internal.kernel.logical;

import java.lang.foreign.ValueLayout;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.VectorOperators;
import jnum.NDArray;
import jnum.internal.layout.NDIter;
import static jnum.internal.Constants.*;


public final class Not {

    private Not() {
        throw new AssertionError();
    }

    public static NDArray not(NDArray a, NDArray resArray) {
        if (a.isContiguous() && resArray.isContiguous()) {
            long i = 0;
            long loopbound = a.getSize() - (a.getSize() % (VL_BOOL * 2L));

            for (; i < loopbound; i += VL_BOOL * 2L) {
                var va1 = ByteVector.fromMemorySegment(SPECIES_BOOL, a.getData(), i * BYTES_BOOL, NATIVE_ORDER);
                var va2 = ByteVector.fromMemorySegment(SPECIES_BOOL, a.getData(), (i + VL_BOOL) * BYTES_BOOL, NATIVE_ORDER);
                var VRes1 = va1.lanewise(VectorOperators.NOT);
                var VRes2 = va2.lanewise(VectorOperators.NOT);
                VRes1.intoMemorySegment(resArray.getData(), i * BYTES_BOOL, NATIVE_ORDER);
                VRes2.intoMemorySegment(resArray.getData(), (i + VL_BOOL) * BYTES_BOOL, NATIVE_ORDER);
            }
            loopbound = SPECIES_BOOL.loopBound(a.getSize());
            for (; i < loopbound; i += VL_BOOL) {
                var va = ByteVector.fromMemorySegment(SPECIES_BOOL, a.getData(), i * BYTES_BOOL, NATIVE_ORDER);
                var VRes = va.lanewise(VectorOperators.NOT);
                VRes.intoMemorySegment(resArray.getData(), i * BYTES_BOOL, NATIVE_ORDER);
            }
            for (; i < a.getSize(); i++) {
                byte valA = a.getData().getAtIndex(ValueLayout.JAVA_BYTE, i);
                resArray.getData().setAtIndex(ValueLayout.JAVA_BYTE, i, (byte) (~valA));
            }
        } else {
            NDIter iterA = new NDIter(resArray.internalShapeUnsafe(), a.internalStridesUnsafe());
            NDIter iterRes = new NDIter(resArray.internalShapeUnsafe(), resArray.internalStridesUnsafe());

            while (iterA.hasNext) {
                byte valA = a.getData().getAtIndex(ValueLayout.JAVA_BYTE, iterA.offset);
                resArray.getData().setAtIndex(ValueLayout.JAVA_BYTE, iterRes.offset, (byte) (~valA));
                iterA.next();
                iterRes.next();
            }
        }
        return resArray;
    }
}
