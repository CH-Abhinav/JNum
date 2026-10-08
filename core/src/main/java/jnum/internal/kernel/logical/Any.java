package jnum.internal.kernel.logical;

import java.lang.foreign.ValueLayout;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.VectorOperators;
import jnum.NDArray;
import jnum.internal.layout.NDIter;

import static jnum.internal.Constants.*;

public final class Any {

    private Any() {
        throw new AssertionError();
    }

    public static boolean any(NDArray a) {
        if (a.isContiguous()) {
            long i = 0;
            long loopbound = a.getSize() - (a.getSize() % VL_BOOL);

            for (; i < loopbound; i += VL_BOOL) {
                var va = ByteVector.fromMemorySegment(SPECIES_BOOL, a.getData(), i * BYTES_BOOL, NATIVE_ORDER);
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
