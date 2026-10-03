package jnum.internal.kernel.logical;

import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import jnum.NDArray;
import jnum.internal.layout.NDIter;

public final class Or {
    private static final VectorSpecies<Byte> SPECIES_BOOL = ByteVector.SPECIES_PREFERRED;
    private static final long BOOL_BYTES = ValueLayout.JAVA_BYTE.byteSize();
    private static final int BOOL_VL = SPECIES_BOOL.length();
    private static final ByteOrder ORDER = ByteOrder.nativeOrder();

    private Or() {
        throw new AssertionError();
    }

    public static NDArray or(NDArray a, NDArray b, NDArray resArray) {
        if (a.isContiguous() && b.isContiguous() && resArray.isContiguous()) {
            long i = 0;
            long loopbound = a.getSize() - (a.getSize() % (BOOL_VL * 2L));

            for (; i < loopbound; i += BOOL_VL * 2L) {
                var va1 = ByteVector.fromMemorySegment(SPECIES_BOOL, a.getData(), i * BOOL_BYTES, ORDER);
                var va2 = ByteVector.fromMemorySegment(SPECIES_BOOL, a.getData(), (i + BOOL_VL) * BOOL_BYTES, ORDER);
                var vb1 = ByteVector.fromMemorySegment(SPECIES_BOOL, b.getData(), i * BOOL_BYTES, ORDER);
                var vb2 = ByteVector.fromMemorySegment(SPECIES_BOOL, b.getData(), (i + BOOL_VL) * BOOL_BYTES, ORDER);
                var VRes1 = va1.lanewise(VectorOperators.OR, vb1);
                var VRes2 = va2.lanewise(VectorOperators.OR, vb2);
                VRes1.intoMemorySegment(resArray.getData(), i * BOOL_BYTES, ORDER);
                VRes2.intoMemorySegment(resArray.getData(), (i + BOOL_VL) * BOOL_BYTES, ORDER);
            }
            loopbound = SPECIES_BOOL.loopBound(a.getSize());
            for (; i < loopbound; i += BOOL_VL) {
                var va = ByteVector.fromMemorySegment(SPECIES_BOOL, a.getData(), i * BOOL_BYTES, ORDER);
                var vb = ByteVector.fromMemorySegment(SPECIES_BOOL, b.getData(), i * BOOL_BYTES, ORDER);
                var VRes = va.lanewise(VectorOperators.OR, vb);
                VRes.intoMemorySegment(resArray.getData(), i * BOOL_BYTES, ORDER);
            }
            for (; i < a.getSize(); i++) {
                byte valA = a.getData().getAtIndex(ValueLayout.JAVA_BYTE, i);
                byte valB = b.getData().getAtIndex(ValueLayout.JAVA_BYTE, i);
                resArray.getData().setAtIndex(ValueLayout.JAVA_BYTE, i, (byte) (valA | valB));
            }
        } else {
            NDIter iterA = new NDIter(resArray.internalShapeUnsafe(), a.internalStridesUnsafe());
            NDIter iterB = new NDIter(resArray.internalShapeUnsafe(), b.internalStridesUnsafe());
            NDIter iterRes = new NDIter(resArray.internalShapeUnsafe(), resArray.internalStridesUnsafe());

            while (iterA.hasNext) {
                byte valA = a.getData().getAtIndex(ValueLayout.JAVA_BYTE, iterA.offset);
                byte valB = b.getData().getAtIndex(ValueLayout.JAVA_BYTE, iterB.offset);
                resArray.getData().setAtIndex(ValueLayout.JAVA_BYTE, iterRes.offset, (byte) (valA | valB));
                iterA.next();
                iterB.next();
                iterRes.next();
            }
        }
        return resArray;
    }
}
