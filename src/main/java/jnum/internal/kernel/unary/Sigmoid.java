package jnum.internal.kernel.unary;

import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorSpecies;
import jdk.incubator.vector.VectorOperators;
import jnum.NDArray;
import jnum.internal.layout.NDIter;

public final class Sigmoid {
    private static final VectorSpecies<Float> SPECIES= FloatVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Integer> SPECIESINT= IntVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Double> SPECIESDB= DoubleVector.SPECIES_PREFERRED;
    private static final long FLOAT_BYTES = ValueLayout.JAVA_FLOAT.byteSize();
    private static final long INT_BYTES = ValueLayout.JAVA_INT.byteSize();
    private static final long DB_BYTES = ValueLayout.JAVA_DOUBLE.byteSize();
    private static final ByteOrder ORDER = ByteOrder.nativeOrder();
    private static final int VL = SPECIES.length();
    private static final int INT_VL = SPECIESINT.length();
    private static final int DB_VL = SPECIESDB.length();

    private Sigmoid() {
        throw new AssertionError();
    }

    public static NDArray sigmoidFloat(NDArray a, NDArray resArray) {
        if (a.isContiguous() && resArray.isContiguous()) {
            long i = 0;
            long loopbound = a.getSize() - (a.getSize() % (VL * 2));
            var one = FloatVector.broadcast(SPECIES, 1.0f);
                         
            for (; i < loopbound; i += VL * 2) {
                var v1 = FloatVector.fromMemorySegment(SPECIES, a.getData(), i * FLOAT_BYTES, ORDER);
                var v2 = FloatVector.fromMemorySegment(SPECIES, a.getData(), (i + VL) * FLOAT_BYTES, ORDER);
                var VRes1 = one.div(one.add(v1.neg().lanewise(VectorOperators.EXP)));
                var VRes2 = one.div(one.add(v2.neg().lanewise(VectorOperators.EXP)));
                VRes1.intoMemorySegment(resArray.getData(), i * FLOAT_BYTES, ORDER);
                VRes2.intoMemorySegment(resArray.getData(), (i + VL) * FLOAT_BYTES, ORDER);
            }
            loopbound = SPECIES.loopBound(a.getSize());
            for (; i < loopbound; i += VL) {
                var v = FloatVector.fromMemorySegment(SPECIES, a.getData(), i * FLOAT_BYTES, ORDER);
                var VRes = one.div(one.add(v.neg().lanewise(VectorOperators.EXP)));
                VRes.intoMemorySegment(resArray.getData(), i * FLOAT_BYTES, ORDER);
            }
            for (; i < a.getSize(); i++) {
                float val = a.getData().getAtIndex(ValueLayout.JAVA_FLOAT, i);
                resArray.getData().setAtIndex(ValueLayout.JAVA_FLOAT, i, (float) (1.0 / (1.0 + Math.exp(-val))));
            }
                     
        } else {
            NDIter iterA = new NDIter(resArray.internalShapeUnsafe(), a.internalStridesUnsafe());
            NDIter iterRes = new NDIter(resArray.internalShapeUnsafe(), resArray.internalStridesUnsafe());
                         
            while (iterA.hasNext) {
                float val = a.getData().getAtIndex(ValueLayout.JAVA_FLOAT, iterA.offset);
                resArray.getData().setAtIndex(ValueLayout.JAVA_FLOAT, iterRes.offset, (float) (1.0 / (1.0 + Math.exp(-val))));
                iterA.next();
                iterRes.next();
            }
        }
        return resArray;
    }


    public static NDArray sigmoidDouble(NDArray a, NDArray resArray) {
        if (a.isContiguous() && resArray.isContiguous()) {
            long i = 0;
            long loopbound = a.getSize() - (a.getSize() % (DB_VL * 2));
            var one = DoubleVector.broadcast(SPECIESDB, 1.0);
                         
            for (; i < loopbound; i += DB_VL * 2) {
                var v1 = DoubleVector.fromMemorySegment(SPECIESDB, a.getData(), i * DB_BYTES, ORDER);
                var v2 = DoubleVector.fromMemorySegment(SPECIESDB, a.getData(), (i + DB_VL) * DB_BYTES, ORDER);
                var VRes1 = one.div(one.add(v1.neg().lanewise(VectorOperators.EXP)));
                var VRes2 = one.div(one.add(v2.neg().lanewise(VectorOperators.EXP)));
                VRes1.intoMemorySegment(resArray.getData(), i * DB_BYTES, ORDER);
                VRes2.intoMemorySegment(resArray.getData(), (i + DB_VL) * DB_BYTES, ORDER);
            }
            loopbound = SPECIESDB.loopBound(a.getSize());
            for (; i < loopbound; i += DB_VL) {
                var v = DoubleVector.fromMemorySegment(SPECIESDB, a.getData(), i * DB_BYTES, ORDER);
                var VRes = one.div(one.add(v.neg().lanewise(VectorOperators.EXP)));
                VRes.intoMemorySegment(resArray.getData(), i * DB_BYTES, ORDER);
            }
            for (; i < a.getSize(); i++) {
                double val = a.getData().getAtIndex(ValueLayout.JAVA_DOUBLE, i);
                resArray.getData().setAtIndex(ValueLayout.JAVA_DOUBLE, i, 1.0 / (1.0 + Math.exp(-val)));
            }
                     
        } else {
            NDIter iterA = new NDIter(resArray.internalShapeUnsafe(), a.internalStridesUnsafe());
            NDIter iterRes = new NDIter(resArray.internalShapeUnsafe(), resArray.internalStridesUnsafe());
                         
            while (iterA.hasNext) {
                double val = a.getData().getAtIndex(ValueLayout.JAVA_DOUBLE, iterA.offset);
                resArray.getData().setAtIndex(ValueLayout.JAVA_DOUBLE, iterRes.offset, 1.0 / (1.0 + Math.exp(-val)));
                iterA.next();
                iterRes.next();
            }
        }
        return resArray;
    }


    public static NDArray sigmoidInt(NDArray a, NDArray resArray) {
        if (a.isContiguous() && resArray.isContiguous()) {
            long i = 0;
            long loopbound = a.getSize() - (a.getSize() % (INT_VL * 2));
            var one = FloatVector.broadcast(SPECIES, 1.0f);
                         
            for (; i < loopbound; i += INT_VL * 2) {
                var vInt1 = IntVector.fromMemorySegment(SPECIESINT, a.getData(), i * INT_BYTES, ORDER);
                var vInt2 = IntVector.fromMemorySegment(SPECIESINT, a.getData(), (i + INT_VL) * INT_BYTES, ORDER);
                var vFloat1 = vInt1.convert(VectorOperators.I2F, 0);
                var vFloat2 = vInt2.convert(VectorOperators.I2F, 0);
                var VRes1 = one.div(one.add((FloatVector) vFloat1.neg().lanewise(VectorOperators.EXP)));
                var VRes2 = one.div(one.add((FloatVector) vFloat2.neg().lanewise(VectorOperators.EXP)));
                VRes1.intoMemorySegment(resArray.getData(), i * FLOAT_BYTES, ORDER);
                VRes2.intoMemorySegment(resArray.getData(), (i + INT_VL) * FLOAT_BYTES, ORDER);
            }
            loopbound = SPECIESINT.loopBound(a.getSize());
            for (; i < loopbound; i += INT_VL) {
                var vInt = IntVector.fromMemorySegment(SPECIESINT, a.getData(), i * INT_BYTES, ORDER);
                var vFloat = vInt.convert(VectorOperators.I2F, 0);
                var VRes = one.div(one.add((FloatVector) vFloat.neg().lanewise(VectorOperators.EXP)));
                VRes.intoMemorySegment(resArray.getData(), i * FLOAT_BYTES, ORDER);
            }
            for (; i < a.getSize(); i++) {
                int val = a.getData().getAtIndex(ValueLayout.JAVA_INT, i);
                resArray.getData().setAtIndex(ValueLayout.JAVA_FLOAT, i, (float) (1.0 / (1.0 + Math.exp(-val))));
            }
                     
        } else {
            NDIter iterA = new NDIter(resArray.internalShapeUnsafe(), a.internalStridesUnsafe());
            NDIter iterRes = new NDIter(resArray.internalShapeUnsafe(), resArray.internalStridesUnsafe());
                         
            while (iterA.hasNext) {
                int val = a.getData().getAtIndex(ValueLayout.JAVA_INT, iterA.offset);
                resArray.getData().setAtIndex(ValueLayout.JAVA_FLOAT, iterRes.offset, (float) (1.0 / (1.0 + Math.exp(-val))));
                iterA.next();
                iterRes.next();
            }
        }
        return resArray;
    }
}
