package jnum.jnumops;

import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorSpecies;
import jdk.incubator.vector.VectorOperators;
import jnum.NDArray;

public class ExpOps {
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

    private ExpOps() {
        throw new AssertionError();
    }

    
    public static NDArray sqrtFloat(NDArray a, NDArray resArray) {
        if (a.isContiguous() && resArray.isContiguous()) {
            long i = 0;
            long loopbound = a.size - (a.size % (VL * 2));
            
            for (; i < loopbound; i += VL * 2) {
                var v1 = FloatVector.fromMemorySegment(SPECIES, a.data, i * FLOAT_BYTES, ORDER);
                var v2 = FloatVector.fromMemorySegment(SPECIES, a.data, (i + VL) * FLOAT_BYTES, ORDER);
                var VRes1 = v1.lanewise(VectorOperators.SQRT);
                var VRes2 = v2.lanewise(VectorOperators.SQRT);
                VRes1.intoMemorySegment(resArray.data, i * FLOAT_BYTES, ORDER);
                VRes2.intoMemorySegment(resArray.data, (i + VL) * FLOAT_BYTES, ORDER);
            }

            loopbound = SPECIES.loopBound(a.size);
            for (; i < loopbound; i += VL) {
                var v = FloatVector.fromMemorySegment(SPECIES, a.data, i * FLOAT_BYTES, ORDER);
                var VRes = v.lanewise(VectorOperators.SQRT);
                VRes.intoMemorySegment(resArray.data, i * FLOAT_BYTES, ORDER);
            }

            for (; i < a.size; i++) {
                float val = a.data.getAtIndex(ValueLayout.JAVA_FLOAT, i);
                resArray.data.setAtIndex(ValueLayout.JAVA_FLOAT, i, (float) Math.sqrt(val));
            }
            
        } else {
            jnum.jnumops.NDIter iterA = new jnum.jnumops.NDIter(resArray.shape, a.strides);
            jnum.jnumops.NDIter iterRes = new jnum.jnumops.NDIter(resArray.shape, resArray.strides);
            
            while (iterA.hasNext) {
                float val = a.data.getAtIndex(ValueLayout.JAVA_FLOAT, iterA.offset);
                resArray.data.setAtIndex(ValueLayout.JAVA_FLOAT, iterRes.offset, (float) Math.sqrt(val));
                iterA.next();
                iterRes.next();
            }
        }
        return resArray;
    }


    public static NDArray sqrtDouble(NDArray a, NDArray resArray) {
        if (a.isContiguous() && resArray.isContiguous()) {
            long i = 0;
            long loopbound = a.size - (a.size % (DB_VL * 2));
            
            for (; i < loopbound; i += DB_VL * 2) {
                var v1 = DoubleVector.fromMemorySegment(SPECIESDB, a.data, i * DB_BYTES, ORDER);
                var v2 = DoubleVector.fromMemorySegment(SPECIESDB, a.data, (i + DB_VL) * DB_BYTES, ORDER);
                var VRes1 = v1.lanewise(VectorOperators.SQRT);
                var VRes2 = v2.lanewise(VectorOperators.SQRT);
                VRes1.intoMemorySegment(resArray.data, i * DB_BYTES, ORDER);
                VRes2.intoMemorySegment(resArray.data, (i + DB_VL) * DB_BYTES, ORDER);
            }

            loopbound = SPECIESDB.loopBound(a.size);
            for (; i < loopbound; i += DB_VL) {
                var v = DoubleVector.fromMemorySegment(SPECIESDB, a.data, i * DB_BYTES, ORDER);
                var VRes = v.lanewise(VectorOperators.SQRT);
                VRes.intoMemorySegment(resArray.data, i * DB_BYTES, ORDER);
            }

            for (; i < a.size; i++) {
                double val = a.data.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
                resArray.data.setAtIndex(ValueLayout.JAVA_DOUBLE, i, Math.sqrt(val));
            }
            
        } else {
            jnum.jnumops.NDIter iterA = new jnum.jnumops.NDIter(resArray.shape, a.strides);
            jnum.jnumops.NDIter iterRes = new jnum.jnumops.NDIter(resArray.shape, resArray.strides);
            
            while (iterA.hasNext) {
                double val = a.data.getAtIndex(ValueLayout.JAVA_DOUBLE, iterA.offset);
                resArray.data.setAtIndex(ValueLayout.JAVA_DOUBLE, iterRes.offset, Math.sqrt(val));
                iterA.next();
                iterRes.next();
            }
        }
        return resArray;
    }


    public static NDArray sqrtInt(NDArray a, NDArray resArray) {
        if (a.isContiguous() && resArray.isContiguous()) {
            long i = 0;
            long loopbound = a.size - (a.size % (INT_VL * 2));
            
            for (; i < loopbound; i += INT_VL * 2) {
                var vInt1 = IntVector.fromMemorySegment(SPECIESINT, a.data, i * INT_BYTES, ORDER);
                var vInt2 = IntVector.fromMemorySegment(SPECIESINT, a.data, (i + INT_VL) * INT_BYTES, ORDER);
                var vFloat1 = vInt1.convert(VectorOperators.I2F, 0);
                var vFloat2 = vInt2.convert(VectorOperators.I2F, 0);
                var VRes1 = vFloat1.lanewise(VectorOperators.SQRT);
                var VRes2 = vFloat2.lanewise(VectorOperators.SQRT);
                VRes1.intoMemorySegment(resArray.data, i * FLOAT_BYTES, ORDER);
                VRes2.intoMemorySegment(resArray.data, (i + INT_VL) * FLOAT_BYTES, ORDER);
            }

            loopbound = SPECIESINT.loopBound(a.size);
            for (; i < loopbound; i += INT_VL) {
                var vInt = IntVector.fromMemorySegment(SPECIESINT, a.data, i * INT_BYTES, ORDER);
                var vFloat = vInt.convert(VectorOperators.I2F, 0);
                var VRes = vFloat.lanewise(VectorOperators.SQRT);
                VRes.intoMemorySegment(resArray.data, i * FLOAT_BYTES, ORDER);
            }

            for (; i < a.size; i++) {
                int val = a.data.getAtIndex(ValueLayout.JAVA_INT, i);
                resArray.data.setAtIndex(ValueLayout.JAVA_FLOAT, i, (float) Math.sqrt(val));
            }
            
        } else {
            jnum.jnumops.NDIter iterA = new jnum.jnumops.NDIter(resArray.shape, a.strides);
            jnum.jnumops.NDIter iterRes = new jnum.jnumops.NDIter(resArray.shape, resArray.strides);
            
            while (iterA.hasNext) {
                int val = a.data.getAtIndex(ValueLayout.JAVA_INT, iterA.offset);
                resArray.data.setAtIndex(ValueLayout.JAVA_FLOAT, iterRes.offset, (float) Math.sqrt(val));
                iterA.next();
                iterRes.next();
            }
        }
        return resArray;
    }


    public static NDArray absFloat(NDArray a, NDArray resArray) {
        if (a.isContiguous() && resArray.isContiguous()) {
            long i = 0;
            long loopbound = a.size - (a.size % (VL * 2));
            
            for (; i < loopbound; i += VL * 2) {
                var v1 = FloatVector.fromMemorySegment(SPECIES, a.data, i * FLOAT_BYTES, ORDER);
                var v2 = FloatVector.fromMemorySegment(SPECIES, a.data, (i + VL) * FLOAT_BYTES, ORDER);
                var VRes1 = v1.lanewise(VectorOperators.ABS);
                var VRes2 = v2.lanewise(VectorOperators.ABS);
                VRes1.intoMemorySegment(resArray.data, i * FLOAT_BYTES, ORDER);
                VRes2.intoMemorySegment(resArray.data, (i + VL) * FLOAT_BYTES, ORDER);
            }

            loopbound = SPECIES.loopBound(a.size);
            for (; i < loopbound; i += VL) {
                var v = FloatVector.fromMemorySegment(SPECIES, a.data, i * FLOAT_BYTES, ORDER);
                var VRes = v.lanewise(VectorOperators.ABS);
                VRes.intoMemorySegment(resArray.data, i * FLOAT_BYTES, ORDER);
            }

            for (; i < a.size; i++) {
                float val = a.data.getAtIndex(ValueLayout.JAVA_FLOAT, i);
                resArray.data.setAtIndex(ValueLayout.JAVA_FLOAT, i, (float) Math.abs(val));
            }
            
        } else {
            jnum.jnumops.NDIter iterA = new jnum.jnumops.NDIter(resArray.shape, a.strides);
            jnum.jnumops.NDIter iterRes = new jnum.jnumops.NDIter(resArray.shape, resArray.strides);
            
            while (iterA.hasNext) {
                float val = a.data.getAtIndex(ValueLayout.JAVA_FLOAT, iterA.offset);
                resArray.data.setAtIndex(ValueLayout.JAVA_FLOAT, iterRes.offset, (float) Math.abs(val));
                iterA.next();
                iterRes.next();
            }
        }
        return resArray;
    }


    public static NDArray absDouble(NDArray a, NDArray resArray) {
        if (a.isContiguous() && resArray.isContiguous()) {
            long i = 0;
            long loopbound = a.size - (a.size % (DB_VL * 2));
            
            for (; i < loopbound; i += DB_VL * 2) {
                var v1 = DoubleVector.fromMemorySegment(SPECIESDB, a.data, i * DB_BYTES, ORDER);
                var v2 = DoubleVector.fromMemorySegment(SPECIESDB, a.data, (i + DB_VL) * DB_BYTES, ORDER);
                var VRes1 = v1.lanewise(VectorOperators.ABS);
                var VRes2 = v2.lanewise(VectorOperators.ABS);
                VRes1.intoMemorySegment(resArray.data, i * DB_BYTES, ORDER);
                VRes2.intoMemorySegment(resArray.data, (i + DB_VL) * DB_BYTES, ORDER);
            }

            loopbound = SPECIESDB.loopBound(a.size);
            for (; i < loopbound; i += DB_VL) {
                var v = DoubleVector.fromMemorySegment(SPECIESDB, a.data, i * DB_BYTES, ORDER);
                var VRes = v.lanewise(VectorOperators.ABS);
                VRes.intoMemorySegment(resArray.data, i * DB_BYTES, ORDER);
            }

            for (; i < a.size; i++) {
                double val = a.data.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
                resArray.data.setAtIndex(ValueLayout.JAVA_DOUBLE, i, Math.abs(val));
            }
            
        } else {
            jnum.jnumops.NDIter iterA = new jnum.jnumops.NDIter(resArray.shape, a.strides);
            jnum.jnumops.NDIter iterRes = new jnum.jnumops.NDIter(resArray.shape, resArray.strides);
            
            while (iterA.hasNext) {
                double val = a.data.getAtIndex(ValueLayout.JAVA_DOUBLE, iterA.offset);
                resArray.data.setAtIndex(ValueLayout.JAVA_DOUBLE, iterRes.offset, Math.abs(val));
                iterA.next();
                iterRes.next();
            }
        }
        return resArray;
    }


    public static NDArray absInt(NDArray a, NDArray resArray) {
        if (a.isContiguous() && resArray.isContiguous()) {
            long i = 0;
            long loopbound = a.size - (a.size % (INT_VL * 2));
            
            for (; i < loopbound; i += INT_VL * 2) {
                var v1 = IntVector.fromMemorySegment(SPECIESINT, a.data, i * INT_BYTES, ORDER);
                var v2 = IntVector.fromMemorySegment(SPECIESINT, a.data, (i + INT_VL) * INT_BYTES, ORDER);
                var VRes1 = v1.lanewise(VectorOperators.ABS);
                var VRes2 = v2.lanewise(VectorOperators.ABS);
                VRes1.intoMemorySegment(resArray.data, i * INT_BYTES, ORDER);
                VRes2.intoMemorySegment(resArray.data, (i + INT_VL) * INT_BYTES, ORDER);
            }

            loopbound = SPECIESINT.loopBound(a.size);
            for (; i < loopbound; i += INT_VL) {
                var v = IntVector.fromMemorySegment(SPECIESINT, a.data, i * INT_BYTES, ORDER);
                var VRes = v.lanewise(VectorOperators.ABS);
                VRes.intoMemorySegment(resArray.data, i * INT_BYTES, ORDER);
            }

            for (; i < a.size; i++) {
                int val = a.data.getAtIndex(ValueLayout.JAVA_INT, i);
                resArray.data.setAtIndex(ValueLayout.JAVA_INT, i, (int) Math.abs(val));
            }
            
        } else {
            jnum.jnumops.NDIter iterA = new jnum.jnumops.NDIter(resArray.shape, a.strides);
            jnum.jnumops.NDIter iterRes = new jnum.jnumops.NDIter(resArray.shape, resArray.strides);
            
            while (iterA.hasNext) {
                int val = a.data.getAtIndex(ValueLayout.JAVA_INT, iterA.offset);
                resArray.data.setAtIndex(ValueLayout.JAVA_INT, iterRes.offset, (int) Math.abs(val));
                iterA.next();
                iterRes.next();
            }
        }
        return resArray;
    }


    public static NDArray expFloat(NDArray a, NDArray resArray) {
        if (a.isContiguous() && resArray.isContiguous()) {
            long i = 0;
            long loopbound = a.size - (a.size % (VL * 2));
            
            for (; i < loopbound; i += VL * 2) {
                var v1 = FloatVector.fromMemorySegment(SPECIES, a.data, i * FLOAT_BYTES, ORDER);
                var v2 = FloatVector.fromMemorySegment(SPECIES, a.data, (i + VL) * FLOAT_BYTES, ORDER);
                var VRes1 = v1.lanewise(VectorOperators.EXP);
                var VRes2 = v2.lanewise(VectorOperators.EXP);
                VRes1.intoMemorySegment(resArray.data, i * FLOAT_BYTES, ORDER);
                VRes2.intoMemorySegment(resArray.data, (i + VL) * FLOAT_BYTES, ORDER);
            }

            loopbound = SPECIES.loopBound(a.size);
            for (; i < loopbound; i += VL) {
                var v = FloatVector.fromMemorySegment(SPECIES, a.data, i * FLOAT_BYTES, ORDER);
                var VRes = v.lanewise(VectorOperators.EXP);
                VRes.intoMemorySegment(resArray.data, i * FLOAT_BYTES, ORDER);
            }

            for (; i < a.size; i++) {
                float val = a.data.getAtIndex(ValueLayout.JAVA_FLOAT, i);
                resArray.data.setAtIndex(ValueLayout.JAVA_FLOAT, i, (float) Math.exp(val));
            }
            
        } else {
            jnum.jnumops.NDIter iterA = new jnum.jnumops.NDIter(resArray.shape, a.strides);
            jnum.jnumops.NDIter iterRes = new jnum.jnumops.NDIter(resArray.shape, resArray.strides);
            
            while (iterA.hasNext) {
                float val = a.data.getAtIndex(ValueLayout.JAVA_FLOAT, iterA.offset);
                resArray.data.setAtIndex(ValueLayout.JAVA_FLOAT, iterRes.offset, (float) Math.exp(val));
                iterA.next();
                iterRes.next();
            }
        }
        return resArray;
    }


    public static NDArray expDouble(NDArray a, NDArray resArray) {
        if (a.isContiguous() && resArray.isContiguous()) {
            long i = 0;
            long loopbound = a.size - (a.size % (DB_VL * 2));
            
            for (; i < loopbound; i += DB_VL * 2) {
                var v1 = DoubleVector.fromMemorySegment(SPECIESDB, a.data, i * DB_BYTES, ORDER);
                var v2 = DoubleVector.fromMemorySegment(SPECIESDB, a.data, (i + DB_VL) * DB_BYTES, ORDER);
                var VRes1 = v1.lanewise(VectorOperators.EXP);
                var VRes2 = v2.lanewise(VectorOperators.EXP);
                VRes1.intoMemorySegment(resArray.data, i * DB_BYTES, ORDER);
                VRes2.intoMemorySegment(resArray.data, (i + DB_VL) * DB_BYTES, ORDER);
            }

            loopbound = SPECIESDB.loopBound(a.size);
            for (; i < loopbound; i += DB_VL) {
                var v = DoubleVector.fromMemorySegment(SPECIESDB, a.data, i * DB_BYTES, ORDER);
                var VRes = v.lanewise(VectorOperators.EXP);
                VRes.intoMemorySegment(resArray.data, i * DB_BYTES, ORDER);
            }

            for (; i < a.size; i++) {
                double val = a.data.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
                resArray.data.setAtIndex(ValueLayout.JAVA_DOUBLE, i, Math.exp(val));
            }
            
        } else {
            jnum.jnumops.NDIter iterA = new jnum.jnumops.NDIter(resArray.shape, a.strides);
            jnum.jnumops.NDIter iterRes = new jnum.jnumops.NDIter(resArray.shape, resArray.strides);
            
            while (iterA.hasNext) {
                double val = a.data.getAtIndex(ValueLayout.JAVA_DOUBLE, iterA.offset);
                resArray.data.setAtIndex(ValueLayout.JAVA_DOUBLE, iterRes.offset, Math.exp(val));
                iterA.next();
                iterRes.next();
            }
        }
        return resArray;
    }


    public static NDArray expInt(NDArray a, NDArray resArray) {
        if (a.isContiguous() && resArray.isContiguous()) {
            long i = 0;
            long loopbound = a.size - (a.size % (INT_VL * 2));
            
            for (; i < loopbound; i += INT_VL * 2) {
                var vInt1 = IntVector.fromMemorySegment(SPECIESINT, a.data, i * INT_BYTES, ORDER);
                var vInt2 = IntVector.fromMemorySegment(SPECIESINT, a.data, (i + INT_VL) * INT_BYTES, ORDER);
                var vFloat1 = vInt1.convert(VectorOperators.I2F, 0);
                var vFloat2 = vInt2.convert(VectorOperators.I2F, 0);
                var VRes1 = vFloat1.lanewise(VectorOperators.EXP);
                var VRes2 = vFloat2.lanewise(VectorOperators.EXP);
                VRes1.intoMemorySegment(resArray.data, i * FLOAT_BYTES, ORDER);
                VRes2.intoMemorySegment(resArray.data, (i + INT_VL) * FLOAT_BYTES, ORDER);
            }

            loopbound = SPECIESINT.loopBound(a.size);
            for (; i < loopbound; i += INT_VL) {
                var vInt = IntVector.fromMemorySegment(SPECIESINT, a.data, i * INT_BYTES, ORDER);
                var vFloat = vInt.convert(VectorOperators.I2F, 0);
                var VRes = vFloat.lanewise(VectorOperators.EXP);
                VRes.intoMemorySegment(resArray.data, i * FLOAT_BYTES, ORDER);
            }

            for (; i < a.size; i++) {
                int val = a.data.getAtIndex(ValueLayout.JAVA_INT, i);
                resArray.data.setAtIndex(ValueLayout.JAVA_FLOAT, i, (float) Math.exp(val));
            }
            
        } else {
            jnum.jnumops.NDIter iterA = new jnum.jnumops.NDIter(resArray.shape, a.strides);
            jnum.jnumops.NDIter iterRes = new jnum.jnumops.NDIter(resArray.shape, resArray.strides);
            
            while (iterA.hasNext) {
                int val = a.data.getAtIndex(ValueLayout.JAVA_INT, iterA.offset);
                resArray.data.setAtIndex(ValueLayout.JAVA_FLOAT, iterRes.offset, (float) Math.exp(val));
                iterA.next();
                iterRes.next();
            }
        }
        return resArray;
    }


    public static NDArray logFloat(NDArray a, NDArray resArray) {
        if (a.isContiguous() && resArray.isContiguous()) {
            long i = 0;
            long loopbound = a.size - (a.size % (VL * 2));
            
            for (; i < loopbound; i += VL * 2) {
                var v1 = FloatVector.fromMemorySegment(SPECIES, a.data, i * FLOAT_BYTES, ORDER);
                var v2 = FloatVector.fromMemorySegment(SPECIES, a.data, (i + VL) * FLOAT_BYTES, ORDER);
                var VRes1 = v1.lanewise(VectorOperators.LOG);
                var VRes2 = v2.lanewise(VectorOperators.LOG);
                VRes1.intoMemorySegment(resArray.data, i * FLOAT_BYTES, ORDER);
                VRes2.intoMemorySegment(resArray.data, (i + VL) * FLOAT_BYTES, ORDER);
            }

            loopbound = SPECIES.loopBound(a.size);
            for (; i < loopbound; i += VL) {
                var v = FloatVector.fromMemorySegment(SPECIES, a.data, i * FLOAT_BYTES, ORDER);
                var VRes = v.lanewise(VectorOperators.LOG);
                VRes.intoMemorySegment(resArray.data, i * FLOAT_BYTES, ORDER);
            }

            for (; i < a.size; i++) {
                float val = a.data.getAtIndex(ValueLayout.JAVA_FLOAT, i);
                resArray.data.setAtIndex(ValueLayout.JAVA_FLOAT, i, (float) Math.log(val));
            }
            
        } else {
            jnum.jnumops.NDIter iterA = new jnum.jnumops.NDIter(resArray.shape, a.strides);
            jnum.jnumops.NDIter iterRes = new jnum.jnumops.NDIter(resArray.shape, resArray.strides);
            
            while (iterA.hasNext) {
                float val = a.data.getAtIndex(ValueLayout.JAVA_FLOAT, iterA.offset);
                resArray.data.setAtIndex(ValueLayout.JAVA_FLOAT, iterRes.offset, (float) Math.log(val));
                iterA.next();
                iterRes.next();
            }
        }
        return resArray;
    }


    public static NDArray logDouble(NDArray a, NDArray resArray) {
        if (a.isContiguous() && resArray.isContiguous()) {
            long i = 0;
            long loopbound = a.size - (a.size % (DB_VL * 2));
            
            for (; i < loopbound; i += DB_VL * 2) {
                var v1 = DoubleVector.fromMemorySegment(SPECIESDB, a.data, i * DB_BYTES, ORDER);
                var v2 = DoubleVector.fromMemorySegment(SPECIESDB, a.data, (i + DB_VL) * DB_BYTES, ORDER);
                var VRes1 = v1.lanewise(VectorOperators.LOG);
                var VRes2 = v2.lanewise(VectorOperators.LOG);
                VRes1.intoMemorySegment(resArray.data, i * DB_BYTES, ORDER);
                VRes2.intoMemorySegment(resArray.data, (i + DB_VL) * DB_BYTES, ORDER);
            }

            loopbound = SPECIESDB.loopBound(a.size);
            for (; i < loopbound; i += DB_VL) {
                var v = DoubleVector.fromMemorySegment(SPECIESDB, a.data, i * DB_BYTES, ORDER);
                var VRes = v.lanewise(VectorOperators.LOG);
                VRes.intoMemorySegment(resArray.data, i * DB_BYTES, ORDER);
            }

            for (; i < a.size; i++) {
                double val = a.data.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
                resArray.data.setAtIndex(ValueLayout.JAVA_DOUBLE, i, Math.log(val));
            }
            
        } else {
            jnum.jnumops.NDIter iterA = new jnum.jnumops.NDIter(resArray.shape, a.strides);
            jnum.jnumops.NDIter iterRes = new jnum.jnumops.NDIter(resArray.shape, resArray.strides);
            
            while (iterA.hasNext) {
                double val = a.data.getAtIndex(ValueLayout.JAVA_DOUBLE, iterA.offset);
                resArray.data.setAtIndex(ValueLayout.JAVA_DOUBLE, iterRes.offset, Math.log(val));
                iterA.next();
                iterRes.next();
            }
        }
        return resArray;
    }


    public static NDArray logInt(NDArray a, NDArray resArray) {
        if (a.isContiguous() && resArray.isContiguous()) {
            long i = 0;
            long loopbound = a.size - (a.size % (INT_VL * 2));
            
            for (; i < loopbound; i += INT_VL * 2) {
                var vInt1 = IntVector.fromMemorySegment(SPECIESINT, a.data, i * INT_BYTES, ORDER);
                var vInt2 = IntVector.fromMemorySegment(SPECIESINT, a.data, (i + INT_VL) * INT_BYTES, ORDER);
                var vFloat1 = vInt1.convert(VectorOperators.I2F, 0);
                var vFloat2 = vInt2.convert(VectorOperators.I2F, 0);
                var VRes1 = vFloat1.lanewise(VectorOperators.LOG);
                var VRes2 = vFloat2.lanewise(VectorOperators.LOG);
                VRes1.intoMemorySegment(resArray.data, i * FLOAT_BYTES, ORDER);
                VRes2.intoMemorySegment(resArray.data, (i + INT_VL) * FLOAT_BYTES, ORDER);
            }

            loopbound = SPECIESINT.loopBound(a.size);
            for (; i < loopbound; i += INT_VL) {
                var vInt = IntVector.fromMemorySegment(SPECIESINT, a.data, i * INT_BYTES, ORDER);
                var vFloat = vInt.convert(VectorOperators.I2F, 0);
                var VRes = vFloat.lanewise(VectorOperators.LOG);
                VRes.intoMemorySegment(resArray.data, i * FLOAT_BYTES, ORDER);
            }

            for (; i < a.size; i++) {
                int val = a.data.getAtIndex(ValueLayout.JAVA_INT, i);
                resArray.data.setAtIndex(ValueLayout.JAVA_FLOAT, i, (float) Math.log(val));
            }
            
        } else {
            jnum.jnumops.NDIter iterA = new jnum.jnumops.NDIter(resArray.shape, a.strides);
            jnum.jnumops.NDIter iterRes = new jnum.jnumops.NDIter(resArray.shape, resArray.strides);
            
            while (iterA.hasNext) {
                int val = a.data.getAtIndex(ValueLayout.JAVA_INT, iterA.offset);
                resArray.data.setAtIndex(ValueLayout.JAVA_FLOAT, iterRes.offset, (float) Math.log(val));
                iterA.next();
                iterRes.next();
            }
        }
        return resArray;
    }


    public static NDArray log10Float(NDArray a, NDArray resArray) {
        if (a.isContiguous() && resArray.isContiguous()) {
            long i = 0;
            long loopbound = a.size - (a.size % (VL * 2));
            
            for (; i < loopbound; i += VL * 2) {
                var v1 = FloatVector.fromMemorySegment(SPECIES, a.data, i * FLOAT_BYTES, ORDER);
                var v2 = FloatVector.fromMemorySegment(SPECIES, a.data, (i + VL) * FLOAT_BYTES, ORDER);
                var VRes1 = v1.lanewise(VectorOperators.LOG10);
                var VRes2 = v2.lanewise(VectorOperators.LOG10);
                VRes1.intoMemorySegment(resArray.data, i * FLOAT_BYTES, ORDER);
                VRes2.intoMemorySegment(resArray.data, (i + VL) * FLOAT_BYTES, ORDER);
            }

            loopbound = SPECIES.loopBound(a.size);
            for (; i < loopbound; i += VL) {
                var v = FloatVector.fromMemorySegment(SPECIES, a.data, i * FLOAT_BYTES, ORDER);
                var VRes = v.lanewise(VectorOperators.LOG10);
                VRes.intoMemorySegment(resArray.data, i * FLOAT_BYTES, ORDER);
            }

            for (; i < a.size; i++) {
                float val = a.data.getAtIndex(ValueLayout.JAVA_FLOAT, i);
                resArray.data.setAtIndex(ValueLayout.JAVA_FLOAT, i, (float) Math.log10(val));
            }
            
        } else {
            jnum.jnumops.NDIter iterA = new jnum.jnumops.NDIter(resArray.shape, a.strides);
            jnum.jnumops.NDIter iterRes = new jnum.jnumops.NDIter(resArray.shape, resArray.strides);
            
            while (iterA.hasNext) {
                float val = a.data.getAtIndex(ValueLayout.JAVA_FLOAT, iterA.offset);
                resArray.data.setAtIndex(ValueLayout.JAVA_FLOAT, iterRes.offset, (float) Math.log10(val));
                iterA.next();
                iterRes.next();
            }
        }
        return resArray;
    }


    public static NDArray log10Double(NDArray a, NDArray resArray) {
        if (a.isContiguous() && resArray.isContiguous()) {
            long i = 0;
            long loopbound = a.size - (a.size % (DB_VL * 2));
            
            for (; i < loopbound; i += DB_VL * 2) {
                var v1 = DoubleVector.fromMemorySegment(SPECIESDB, a.data, i * DB_BYTES, ORDER);
                var v2 = DoubleVector.fromMemorySegment(SPECIESDB, a.data, (i + DB_VL) * DB_BYTES, ORDER);
                var VRes1 = v1.lanewise(VectorOperators.LOG10);
                var VRes2 = v2.lanewise(VectorOperators.LOG10);
                VRes1.intoMemorySegment(resArray.data, i * DB_BYTES, ORDER);
                VRes2.intoMemorySegment(resArray.data, (i + DB_VL) * DB_BYTES, ORDER);
            }

            loopbound = SPECIESDB.loopBound(a.size);
            for (; i < loopbound; i += DB_VL) {
                var v = DoubleVector.fromMemorySegment(SPECIESDB, a.data, i * DB_BYTES, ORDER);
                var VRes = v.lanewise(VectorOperators.LOG10);
                VRes.intoMemorySegment(resArray.data, i * DB_BYTES, ORDER);
            }

            for (; i < a.size; i++) {
                double val = a.data.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
                resArray.data.setAtIndex(ValueLayout.JAVA_DOUBLE, i, Math.log10(val));
            }
            
        } else {
            jnum.jnumops.NDIter iterA = new jnum.jnumops.NDIter(resArray.shape, a.strides);
            jnum.jnumops.NDIter iterRes = new jnum.jnumops.NDIter(resArray.shape, resArray.strides);
            
            while (iterA.hasNext) {
                double val = a.data.getAtIndex(ValueLayout.JAVA_DOUBLE, iterA.offset);
                resArray.data.setAtIndex(ValueLayout.JAVA_DOUBLE, iterRes.offset, Math.log10(val));
                iterA.next();
                iterRes.next();
            }
        }
        return resArray;
    }


    public static NDArray log10Int(NDArray a, NDArray resArray) {
        if (a.isContiguous() && resArray.isContiguous()) {
            long i = 0;
            long loopbound = a.size - (a.size % (INT_VL * 2));
            
            for (; i < loopbound; i += INT_VL * 2) {
                var vInt1 = IntVector.fromMemorySegment(SPECIESINT, a.data, i * INT_BYTES, ORDER);
                var vInt2 = IntVector.fromMemorySegment(SPECIESINT, a.data, (i + INT_VL) * INT_BYTES, ORDER);
                var vFloat1 = vInt1.convert(VectorOperators.I2F, 0);
                var vFloat2 = vInt2.convert(VectorOperators.I2F, 0);
                var VRes1 = vFloat1.lanewise(VectorOperators.LOG10);
                var VRes2 = vFloat2.lanewise(VectorOperators.LOG10);
                VRes1.intoMemorySegment(resArray.data, i * FLOAT_BYTES, ORDER);
                VRes2.intoMemorySegment(resArray.data, (i + INT_VL) * FLOAT_BYTES, ORDER);
            }

            loopbound = SPECIESINT.loopBound(a.size);
            for (; i < loopbound; i += INT_VL) {
                var vInt = IntVector.fromMemorySegment(SPECIESINT, a.data, i * INT_BYTES, ORDER);
                var vFloat = vInt.convert(VectorOperators.I2F, 0);
                var VRes = vFloat.lanewise(VectorOperators.LOG10);
                VRes.intoMemorySegment(resArray.data, i * FLOAT_BYTES, ORDER);
            }

            for (; i < a.size; i++) {
                int val = a.data.getAtIndex(ValueLayout.JAVA_INT, i);
                resArray.data.setAtIndex(ValueLayout.JAVA_FLOAT, i, (float) Math.log10(val));
            }
            
        } else {
            jnum.jnumops.NDIter iterA = new jnum.jnumops.NDIter(resArray.shape, a.strides);
            jnum.jnumops.NDIter iterRes = new jnum.jnumops.NDIter(resArray.shape, resArray.strides);
            
            while (iterA.hasNext) {
                int val = a.data.getAtIndex(ValueLayout.JAVA_INT, iterA.offset);
                resArray.data.setAtIndex(ValueLayout.JAVA_FLOAT, iterRes.offset, (float) Math.log10(val));
                iterA.next();
                iterRes.next();
            }
        }
        return resArray;
    }

}