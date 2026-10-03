package jnum.internal.kernel.logical;

import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import jnum.NDArray;
import jnum.internal.layout.NDIter;

public final class Not {
    private static final VectorSpecies<Byte> SPECIES_BOOL = ByteVector.SPECIES_PREFERRED;
    private static final long BOOL_BYTES = ValueLayout.JAVA_BYTE.byteSize();
    private static final int BOOL_VL = SPECIES_BOOL.length();
    private static final ByteOrder ORDER = ByteOrder.nativeOrder();

    private Not() {
        throw new AssertionError();
    }

    public static NDArray not(NDArray a, NDArray resArray) {
        if (a.isContiguous() && resArray.isContiguous()) {
            long i = 0;
            long loopbound = a.getSize() - (a.getSize() % (BOOL_VL * 2L));

            for (; i < loopbound; i += BOOL_VL * 2L) {
                var va1 = ByteVector.fromMemorySegment(SPECIES_BOOL, a.getData(), i * BOOL_BYTES, ORDER);
                var va2 = ByteVector.fromMemorySegment(SPECIES_BOOL, a.getData(), (i + BOOL_VL) * BOOL_BYTES, ORDER);
                var VRes1 = va1.lanewise(VectorOperators.NOT);
                var VRes2 = va2.lanewise(VectorOperators.NOT);
                VRes1.intoMemorySegment(resArray.getData(), i * BOOL_BYTES, ORDER);
                VRes2.intoMemorySegment(resArray.getData(), (i + BOOL_VL) * BOOL_BYTES, ORDER);
            }
            loopbound = SPECIES_BOOL.loopBound(a.getSize());
            for (; i < loopbound; i += BOOL_VL) {
                var va = ByteVector.fromMemorySegment(SPECIES_BOOL, a.getData(), i * BOOL_BYTES, ORDER);
                var VRes = va.lanewise(VectorOperators.NOT);
                VRes.intoMemorySegment(resArray.getData(), i * BOOL_BYTES, ORDER);
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
