package jnum.internal.kernel.logical;

import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import jnum.NDArray;
import jnum.internal.layout.NDIter;

public final class All {
    private static final VectorSpecies<Byte> SPECIES_BOOL = ByteVector.SPECIES_PREFERRED;
    private static final long BOOL_BYTES = ValueLayout.JAVA_BYTE.byteSize();
    private static final int BOOL_VL = SPECIES_BOOL.length();
    private static final ByteOrder ORDER = ByteOrder.nativeOrder();

    private All() {
        throw new AssertionError();
    }

    public static boolean all(NDArray a) {
        if (a.isContiguous()) {
            long i = 0;
            long loopbound = a.getSize() - (a.getSize() % BOOL_VL);

            for (; i < loopbound; i += BOOL_VL) {
                var va = ByteVector.fromMemorySegment(SPECIES_BOOL, a.getData(), i * BOOL_BYTES, ORDER);
                if (va.compare(VectorOperators.EQ, 0).anyTrue()) return false;
            }
            for (; i < a.getSize(); i++) {
                if (a.getData().getAtIndex(ValueLayout.JAVA_BYTE, i) == 0) return false;
            }
        } else {
            NDIter iterA = new NDIter(a.internalShapeUnsafe(), a.internalStridesUnsafe());
            while (iterA.hasNext) {
                if (a.getData().getAtIndex(ValueLayout.JAVA_BYTE, iterA.offset) == 0) return false;
                iterA.next();
            }
        }
        return true;
    }
}
