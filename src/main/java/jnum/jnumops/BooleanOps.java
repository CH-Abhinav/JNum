package jnum.jnumops;

import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import jnum.NDArray;

public class BooleanOps {
    private static final VectorSpecies<Byte> SPECIES_BOOL = ByteVector.SPECIES_PREFERRED;
    private static final long BOOL_BYTES = ValueLayout.JAVA_BYTE.byteSize();
    private static final int BOOL_VL = SPECIES_BOOL.length();
    private static final ByteOrder ORDER = ByteOrder.nativeOrder();

    private BooleanOps() {
        throw new AssertionError();
    }


    public static NDArray and(NDArray a, NDArray b, NDArray resArray) {
        if (a.isContiguous() && b.isContiguous() && resArray.isContiguous()) {
            long i = 0;
            long loopbound = a.getSize() - (a.getSize() % (BOOL_VL * 2L));

            for (; i < loopbound; i += BOOL_VL * 2L) {
                var va1 = ByteVector.fromMemorySegment(SPECIES_BOOL, a.getData(), i * BOOL_BYTES, ORDER);
                var va2 = ByteVector.fromMemorySegment(SPECIES_BOOL, a.getData(), (i + BOOL_VL) * BOOL_BYTES, ORDER);
                var vb1 = ByteVector.fromMemorySegment(SPECIES_BOOL, b.getData(), i * BOOL_BYTES, ORDER);
                var vb2 = ByteVector.fromMemorySegment(SPECIES_BOOL, b.getData(), (i + BOOL_VL) * BOOL_BYTES, ORDER);
                var VRes1 = va1.lanewise(VectorOperators.AND, vb1);
                var VRes2 = va2.lanewise(VectorOperators.AND, vb2);
                VRes1.intoMemorySegment(resArray.getData(), i * BOOL_BYTES, ORDER);
                VRes2.intoMemorySegment(resArray.getData(), (i + BOOL_VL) * BOOL_BYTES, ORDER);
            }
            loopbound = SPECIES_BOOL.loopBound(a.getSize());
            for (; i < loopbound; i += BOOL_VL) {
                var va = ByteVector.fromMemorySegment(SPECIES_BOOL, a.getData(), i * BOOL_BYTES, ORDER);
                var vb = ByteVector.fromMemorySegment(SPECIES_BOOL, b.getData(), i * BOOL_BYTES, ORDER);
                var VRes = va.lanewise(VectorOperators.AND, vb);
                VRes.intoMemorySegment(resArray.getData(), i * BOOL_BYTES, ORDER);
            }
            for (; i < a.getSize(); i++) {
                byte valA = a.getData().getAtIndex(ValueLayout.JAVA_BYTE, i);
                byte valB = b.getData().getAtIndex(ValueLayout.JAVA_BYTE, i);
                resArray.getData().setAtIndex(ValueLayout.JAVA_BYTE, i, (byte) (valA & valB));
            }
        } else {
            jnum.jnumops.NDIter iterA = new jnum.jnumops.NDIter(resArray.internalShapeUnsafe(), a.internalStridesUnsafe());
            jnum.jnumops.NDIter iterB = new jnum.jnumops.NDIter(resArray.internalShapeUnsafe(), b.internalStridesUnsafe());
            jnum.jnumops.NDIter iterRes = new jnum.jnumops.NDIter(resArray.internalShapeUnsafe(), resArray.internalStridesUnsafe());

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
            jnum.jnumops.NDIter iterA = new jnum.jnumops.NDIter(resArray.internalShapeUnsafe(), a.internalStridesUnsafe());
            jnum.jnumops.NDIter iterB = new jnum.jnumops.NDIter(resArray.internalShapeUnsafe(), b.internalStridesUnsafe());
            jnum.jnumops.NDIter iterRes = new jnum.jnumops.NDIter(resArray.internalShapeUnsafe(), resArray.internalStridesUnsafe());

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


    public static NDArray xor(NDArray a, NDArray b, NDArray resArray) {
        if (a.isContiguous() && b.isContiguous() && resArray.isContiguous()) {
            long i = 0;
            long loopbound = a.getSize() - (a.getSize() % (BOOL_VL * 2L));

            for (; i < loopbound; i += BOOL_VL * 2L) {
                var va1 = ByteVector.fromMemorySegment(SPECIES_BOOL, a.getData(), i * BOOL_BYTES, ORDER);
                var va2 = ByteVector.fromMemorySegment(SPECIES_BOOL, a.getData(), (i + BOOL_VL) * BOOL_BYTES, ORDER);
                var vb1 = ByteVector.fromMemorySegment(SPECIES_BOOL, b.getData(), i * BOOL_BYTES, ORDER);
                var vb2 = ByteVector.fromMemorySegment(SPECIES_BOOL, b.getData(), (i + BOOL_VL) * BOOL_BYTES, ORDER);
                var VRes1 = va1.lanewise(VectorOperators.XOR, vb1);
                var VRes2 = va2.lanewise(VectorOperators.XOR, vb2);
                VRes1.intoMemorySegment(resArray.getData(), i * BOOL_BYTES, ORDER);
                VRes2.intoMemorySegment(resArray.getData(), (i + BOOL_VL) * BOOL_BYTES, ORDER);
            }
            loopbound = SPECIES_BOOL.loopBound(a.getSize());
            for (; i < loopbound; i += BOOL_VL) {
                var va = ByteVector.fromMemorySegment(SPECIES_BOOL, a.getData(), i * BOOL_BYTES, ORDER);
                var vb = ByteVector.fromMemorySegment(SPECIES_BOOL, b.getData(), i * BOOL_BYTES, ORDER);
                var VRes = va.lanewise(VectorOperators.XOR, vb);
                VRes.intoMemorySegment(resArray.getData(), i * BOOL_BYTES, ORDER);
            }
            for (; i < a.getSize(); i++) {
                byte valA = a.getData().getAtIndex(ValueLayout.JAVA_BYTE, i);
                byte valB = b.getData().getAtIndex(ValueLayout.JAVA_BYTE, i);
                resArray.getData().setAtIndex(ValueLayout.JAVA_BYTE, i, (byte) (valA ^ valB));
            }
        } else {
            jnum.jnumops.NDIter iterA = new jnum.jnumops.NDIter(resArray.internalShapeUnsafe(), a.internalStridesUnsafe());
            jnum.jnumops.NDIter iterB = new jnum.jnumops.NDIter(resArray.internalShapeUnsafe(), b.internalStridesUnsafe());
            jnum.jnumops.NDIter iterRes = new jnum.jnumops.NDIter(resArray.internalShapeUnsafe(), resArray.internalStridesUnsafe());

            while (iterA.hasNext) {
                byte valA = a.getData().getAtIndex(ValueLayout.JAVA_BYTE, iterA.offset);
                byte valB = b.getData().getAtIndex(ValueLayout.JAVA_BYTE, iterB.offset);
                resArray.getData().setAtIndex(ValueLayout.JAVA_BYTE, iterRes.offset, (byte) (valA ^ valB));
                iterA.next();
                iterB.next();
                iterRes.next();
            }
        }
        return resArray;
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
            jnum.jnumops.NDIter iterA = new jnum.jnumops.NDIter(resArray.internalShapeUnsafe(), a.internalStridesUnsafe());
            jnum.jnumops.NDIter iterRes = new jnum.jnumops.NDIter(resArray.internalShapeUnsafe(), resArray.internalStridesUnsafe());

            while (iterA.hasNext) {
                byte valA = a.getData().getAtIndex(ValueLayout.JAVA_BYTE, iterA.offset);
                resArray.getData().setAtIndex(ValueLayout.JAVA_BYTE, iterRes.offset, (byte) (~valA));
                iterA.next();
                iterRes.next();
            }
        }
        return resArray;
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
            jnum.jnumops.NDIter iterA = new jnum.jnumops.NDIter(a.internalShapeUnsafe(), a.internalStridesUnsafe());
            while (iterA.hasNext) {
                if (a.getData().getAtIndex(ValueLayout.JAVA_BYTE, iterA.offset) != 0) return true;
                iterA.next();
            }
        }
        return false;
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
            jnum.jnumops.NDIter iterA = new jnum.jnumops.NDIter(a.internalShapeUnsafe(), a.internalStridesUnsafe());
            while (iterA.hasNext) {
                if (a.getData().getAtIndex(ValueLayout.JAVA_BYTE, iterA.offset) == 0) return false;
                iterA.next();
            }
        }
        return true;
    }


}
