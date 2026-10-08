package jnum.internal.kernel.logical;

import java.lang.foreign.ValueLayout;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.VectorOperators;
import jnum.NDArray;
import jnum.internal.layout.NDIter;

import static jnum.internal.Constants.*;

public final class All {


    private All() {
        throw new AssertionError();
    }

    public static boolean all(NDArray a) {
        if (a.isContiguous()) {
            long i = 0;
            long loopbound = a.getSize() - (a.getSize() % VL_BOOL);

            for (; i < loopbound; i += VL_BOOL) {
                var va = ByteVector.fromMemorySegment(SPECIES_BOOL, a.getData(), i * BYTES_BOOL, NATIVE_ORDER);
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
