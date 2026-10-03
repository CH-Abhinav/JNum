package jnum.internal.kernel.logical;

import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import jnum.NDArray;
import jnum.internal.layout.NDIter;

public final class Any {
    private static final VectorSpecies<Byte> SPECIES_BOOL = ByteVector.SPECIES_PREFERRED;
    private static final long BOOL_BYTES = ValueLayout.JAVA_BYTE.byteSize();
    private static final int BOOL_VL = SPECIES_BOOL.length();
    private static final ByteOrder ORDER = ByteOrder.nativeOrder();

    private Any() {
        throw new AssertionError();
    }

    public static boolean any(NDArray a) {
        if (a.isContiguous()) {
            long i = 0;
            long loopbound = a.getSize() - (a.getSize() % BOOL_VL);

            for (; i < loopbound; i += BOOL_VL) {
                var va = ByteVector.fromMemorySegment(SPECIES_BOOL, a.getData(), i * BOOL_BYTES, ORDER);
                if (va.compare(VectorOperators.NE, 0).anyTrue()) return true;
            }
            for (; i < a.getSize(); i++) {
                if (a.getData().getAtIndex(ValueLayout.JAVA_BYTE, i) != 0) return true;
            }
        } else {
            NDIter iterA = new NDIter(a.internalShapeUnsafe(), a.internalStridesUnsafe());
            while (iterA.hasNext) {
                if (a.getData().getAtIndex(ValueLayout.JAVA_BYTE, iterA.offset) != 0) return true;
                iterA.next();
            }
        }
        return false;
    }
}
