package jnum.jnumops;

import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;
import jdk.incubator.vector.VectorMask;
import jnum.NDArray;

public class ArithmeticOps {
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Integer> SPECIESINT = IntVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Double> SPECIESDB = DoubleVector.SPECIES_PREFERRED;
    private static final long FLOAT_BYTES = ValueLayout.JAVA_FLOAT.byteSize();
    private static final long INT_BYTES = ValueLayout.JAVA_INT.byteSize();
    private static final long DB_BYTES = ValueLayout.JAVA_DOUBLE.byteSize();
    private static final ByteOrder ORDER = ByteOrder.nativeOrder();
    private static final int VL = SPECIES.length();
    private static final int INT_VL = SPECIESINT.length();
    private static final int DB_VL = SPECIESDB.length();

    // Non-instantiable utility class    
    private ArithmeticOps() {
        throw new AssertionError();
    }


    public static NDArray addFloat(NDArray a, NDArray b, NDArray resArray) {
        if (a.isContiguous() && b.isContiguous() && resArray.isContiguous()) {
            long i = 0;
            long loopbound = a.getSize() - (a.getSize() % (VL * 2));
                         
            for (; i < loopbound; i += VL * 2) {
                var vA1 = FloatVector.fromMemorySegment(SPECIES, a.getData(), i * FLOAT_BYTES, ORDER);
                var vA2 = FloatVector.fromMemorySegment(SPECIES, a.getData(), (i + VL) * FLOAT_BYTES, ORDER);
                var vB1 = FloatVector.fromMemorySegment(SPECIES, b.getData(), i * FLOAT_BYTES, ORDER);
                var vB2 = FloatVector.fromMemorySegment(SPECIES, b.getData(), (i + VL) * FLOAT_BYTES, ORDER);
                                 
                var VRes1 = vA1.add(vB1);
                var VRes2 = vA2.add(vB2);
                                 
                VRes1.intoMemorySegment(resArray.getData(), i * FLOAT_BYTES, ORDER);
                VRes2.intoMemorySegment(resArray.getData(), (i + VL) * FLOAT_BYTES, ORDER);
            }
            loopbound = SPECIES.loopBound(a.getSize());
            for (; i < loopbound; i += VL) {
                var vA = FloatVector.fromMemorySegment(SPECIES, a.getData(), i * FLOAT_BYTES, ORDER);
                var vB = FloatVector.fromMemorySegment(SPECIES, b.getData(), i * FLOAT_BYTES, ORDER);
                var VRes = vA.add(vB);
                VRes.intoMemorySegment(resArray.getData(), i * FLOAT_BYTES, ORDER);
            }
            for (; i < a.getSize(); i++) {
                float valA = a.getData().getAtIndex(ValueLayout.JAVA_FLOAT, i);
                float valB = b.getData().getAtIndex(ValueLayout.JAVA_FLOAT, i);
                resArray.getData().setAtIndex(ValueLayout.JAVA_FLOAT, i, (float)(valA + valB));
            }
        } else {
            int vl = SPECIES.length();
            int[] mapA = new int[vl];
            int[] mapB = new int[vl];
            int[] mapRes = new int[vl];
                         
            float[] bufA = new float[vl];
            float[] bufB = new float[vl];
            float[] bufRes = new float[vl];
            var iterA = new NDIter(resArray.internalShapeUnsafe(), a.internalStridesUnsafe());
            var iterB = new NDIter(resArray.internalShapeUnsafe(), b.internalStridesUnsafe());
            var iterRes = new NDIter(resArray.internalShapeUnsafe(), resArray.internalStridesUnsafe());
            while (iterA.hasNext) {
                int validLanes = iterA.nextVector(mapA, vl);
                iterB.nextVector(mapB, vl);
                iterRes.nextVector(mapRes, vl);
                for(int k=0; k < validLanes; k++) {
                    bufA[k] = a.getData().getAtIndex(ValueLayout.JAVA_FLOAT, mapA[k]);
                    bufB[k] = b.getData().getAtIndex(ValueLayout.JAVA_FLOAT, mapB[k]);
                }
                var mask = SPECIES.indexInRange(0, validLanes);
                var vA = FloatVector.fromArray(SPECIES, bufA, 0, mask);
                var vB = FloatVector.fromArray(SPECIES, bufB, 0, mask);
                                 
                var vRes = vA.add(vB);
                                 
                vRes.intoArray(bufRes, 0, mask);
                for(int k=0; k < validLanes; k++) {
                    resArray.getData().setAtIndex(ValueLayout.JAVA_FLOAT, mapRes[k], bufRes[k]);
                }
            }
        }
        return resArray;
    }


    public static NDArray addFloat(NDArray a, float b, NDArray resArray) {
        var vB = FloatVector.broadcast(SPECIES, b);
                 
        if (a.isContiguous() && resArray.isContiguous()) {
            long i = 0;
            long loopbound = a.getSize() - (a.getSize() % (VL * 2));
                         
            for (; i < loopbound; i += VL * 2) {
                var vA1 = FloatVector.fromMemorySegment(SPECIES, a.getData(), i * FLOAT_BYTES, ORDER);
                var vA2 = FloatVector.fromMemorySegment(SPECIES, a.getData(), (i + VL) * FLOAT_BYTES, ORDER);
                var VRes1 = vA1.add(vB);
                var VRes2 = vA2.add(vB);
                                 
                VRes1.intoMemorySegment(resArray.getData(), i * FLOAT_BYTES, ORDER);
                VRes2.intoMemorySegment(resArray.getData(), (i + VL) * FLOAT_BYTES, ORDER);
            }
            loopbound = SPECIES.loopBound(a.getSize());
            for (; i < loopbound; i += VL) {
                var vA = FloatVector.fromMemorySegment(SPECIES, a.getData(), i * FLOAT_BYTES, ORDER);
                var VRes = vA.add(vB);
                VRes.intoMemorySegment(resArray.getData(), i * FLOAT_BYTES, ORDER);
            }
            for (; i < a.getSize(); i++) {
                float valA = a.getData().getAtIndex(ValueLayout.JAVA_FLOAT, i);
                resArray.getData().setAtIndex(ValueLayout.JAVA_FLOAT, i, (float)(valA + b));
            }
        } else {
            int vl = SPECIES.length();
            int[] mapA = new int[vl];
            int[] mapRes = new int[vl];
                         
            float[] bufA = new float[vl];
            float[] bufRes = new float[vl];
            var iterA = new NDIter(resArray.internalShapeUnsafe(), a.internalStridesUnsafe());
            var iterRes = new NDIter(resArray.internalShapeUnsafe(), resArray.internalStridesUnsafe());
            while (iterA.hasNext) {
                int validLanes = iterA.nextVector(mapA, vl);
                iterRes.nextVector(mapRes, vl);
                for(int k=0; k < validLanes; k++) {
                    bufA[k] = a.getData().getAtIndex(ValueLayout.JAVA_FLOAT, mapA[k]);
                }
                var mask = SPECIES.indexInRange(0, validLanes);
                var vA = FloatVector.fromArray(SPECIES, bufA, 0, mask);
                                 
                var vRes = vA.add(vB);
                                 
                vRes.intoArray(bufRes, 0, mask);
                for(int k=0; k < validLanes; k++) {
                    resArray.getData().setAtIndex(ValueLayout.JAVA_FLOAT, mapRes[k], bufRes[k]);
                }
            }
        }
        return resArray;
    }


    public static NDArray addDouble(NDArray a, NDArray b, NDArray resArray) {
        if (a.isContiguous() && b.isContiguous() && resArray.isContiguous()) {
            long i = 0;
            long loopbound = a.getSize() - (a.getSize() % (DB_VL * 2));
                         
            for (; i < loopbound; i += DB_VL * 2) {
                var vA1 = DoubleVector.fromMemorySegment(SPECIESDB, a.getData(), i * DB_BYTES, ORDER);
                var vA2 = DoubleVector.fromMemorySegment(SPECIESDB, a.getData(), (i + DB_VL) * DB_BYTES, ORDER);
                var vB1 = DoubleVector.fromMemorySegment(SPECIESDB, b.getData(), i * DB_BYTES, ORDER);
                var vB2 = DoubleVector.fromMemorySegment(SPECIESDB, b.getData(), (i + DB_VL) * DB_BYTES, ORDER);
                                 
                var VRes1 = vA1.add(vB1);
                var VRes2 = vA2.add(vB2);
                                 
                VRes1.intoMemorySegment(resArray.getData(), i * DB_BYTES, ORDER);
                VRes2.intoMemorySegment(resArray.getData(), (i + DB_VL) * DB_BYTES, ORDER);
            }
            loopbound = SPECIESDB.loopBound(a.getSize());
            for (; i < loopbound; i += DB_VL) {
                var vA = DoubleVector.fromMemorySegment(SPECIESDB, a.getData(), i * DB_BYTES, ORDER);
                var vB = DoubleVector.fromMemorySegment(SPECIESDB, b.getData(), i * DB_BYTES, ORDER);
                var VRes = vA.add(vB);
                VRes.intoMemorySegment(resArray.getData(), i * DB_BYTES, ORDER);
            }
            for (; i < a.getSize(); i++) {
                double valA = a.getData().getAtIndex(ValueLayout.JAVA_DOUBLE, i);
                double valB = b.getData().getAtIndex(ValueLayout.JAVA_DOUBLE, i);
                resArray.getData().setAtIndex(ValueLayout.JAVA_DOUBLE, i, (double)(valA + valB));
            }
        } else {
            int vl = SPECIESDB.length();
            int[] mapA = new int[vl];
            int[] mapB = new int[vl];
            int[] mapRes = new int[vl];
                         
            double[] bufA = new double[vl];
            double[] bufB = new double[vl];
            double[] bufRes = new double[vl];
            var iterA = new NDIter(resArray.internalShapeUnsafe(), a.internalStridesUnsafe());
            var iterB = new NDIter(resArray.internalShapeUnsafe(), b.internalStridesUnsafe());
            var iterRes = new NDIter(resArray.internalShapeUnsafe(), resArray.internalStridesUnsafe());
            while (iterA.hasNext) {
                int validLanes = iterA.nextVector(mapA, vl);
                iterB.nextVector(mapB, vl);
                iterRes.nextVector(mapRes, vl);
                for(int k=0; k < validLanes; k++) {
                    bufA[k] = a.getData().getAtIndex(ValueLayout.JAVA_DOUBLE, mapA[k]);
                    bufB[k] = b.getData().getAtIndex(ValueLayout.JAVA_DOUBLE, mapB[k]);
                }
                var mask = SPECIESDB.indexInRange(0, validLanes);
                var vA = DoubleVector.fromArray(SPECIESDB, bufA, 0, mask);
                var vB = DoubleVector.fromArray(SPECIESDB, bufB, 0, mask);
                                 
                var vRes = vA.add(vB);
                                 
                vRes.intoArray(bufRes, 0, mask);
                for(int k=0; k < validLanes; k++) {
                    resArray.getData().setAtIndex(ValueLayout.JAVA_DOUBLE, mapRes[k], bufRes[k]);
                }
            }
        }
        return resArray;
    }


    public static NDArray addDouble(NDArray a, double b, NDArray resArray) {
        var vB = DoubleVector.broadcast(SPECIESDB, b);
                 
        if (a.isContiguous() && resArray.isContiguous()) {
            long i = 0;
            long loopbound = a.getSize() - (a.getSize() % (DB_VL * 2));
                         
            for (; i < loopbound; i += DB_VL * 2) {
                var vA1 = DoubleVector.fromMemorySegment(SPECIESDB, a.getData(), i * DB_BYTES, ORDER);
                var vA2 = DoubleVector.fromMemorySegment(SPECIESDB, a.getData(), (i + DB_VL) * DB_BYTES, ORDER);
                var VRes1 = vA1.add(vB);
                var VRes2 = vA2.add(vB);
                                 
                VRes1.intoMemorySegment(resArray.getData(), i * DB_BYTES, ORDER);
                VRes2.intoMemorySegment(resArray.getData(), (i + DB_VL) * DB_BYTES, ORDER);
            }
            loopbound = SPECIESDB.loopBound(a.getSize());
            for (; i < loopbound; i += DB_VL) {
                var vA = DoubleVector.fromMemorySegment(SPECIESDB, a.getData(), i * DB_BYTES, ORDER);
                var VRes = vA.add(vB);
                VRes.intoMemorySegment(resArray.getData(), i * DB_BYTES, ORDER);
            }
            for (; i < a.getSize(); i++) {
                double valA = a.getData().getAtIndex(ValueLayout.JAVA_DOUBLE, i);
                resArray.getData().setAtIndex(ValueLayout.JAVA_DOUBLE, i, (double)(valA + b));
            }
        } else {
            int vl = SPECIESDB.length();
            int[] mapA = new int[vl];
            int[] mapRes = new int[vl];
                         
            double[] bufA = new double[vl];
            double[] bufRes = new double[vl];
            var iterA = new NDIter(resArray.internalShapeUnsafe(), a.internalStridesUnsafe());
            var iterRes = new NDIter(resArray.internalShapeUnsafe(), resArray.internalStridesUnsafe());
            while (iterA.hasNext) {
                int validLanes = iterA.nextVector(mapA, vl);
                iterRes.nextVector(mapRes, vl);
                for(int k=0; k < validLanes; k++) {
                    bufA[k] = a.getData().getAtIndex(ValueLayout.JAVA_DOUBLE, mapA[k]);
                }
                var mask = SPECIESDB.indexInRange(0, validLanes);
                var vA = DoubleVector.fromArray(SPECIESDB, bufA, 0, mask);
                                 
                var vRes = vA.add(vB);
                                 
                vRes.intoArray(bufRes, 0, mask);
                for(int k=0; k < validLanes; k++) {
                    resArray.getData().setAtIndex(ValueLayout.JAVA_DOUBLE, mapRes[k], bufRes[k]);
                }
            }
        }
        return resArray;
    }


    public static NDArray addInt(NDArray a, NDArray b, NDArray resArray) {
        if (a.isContiguous() && b.isContiguous() && resArray.isContiguous()) {
            long i = 0;
            long loopbound = a.getSize() - (a.getSize() % (INT_VL * 2));
                         
            for (; i < loopbound; i += INT_VL * 2) {
                var vA1 = IntVector.fromMemorySegment(SPECIESINT, a.getData(), i * INT_BYTES, ORDER);
                var vA2 = IntVector.fromMemorySegment(SPECIESINT, a.getData(), (i + INT_VL) * INT_BYTES, ORDER);
                var vB1 = IntVector.fromMemorySegment(SPECIESINT, b.getData(), i * INT_BYTES, ORDER);
                var vB2 = IntVector.fromMemorySegment(SPECIESINT, b.getData(), (i + INT_VL) * INT_BYTES, ORDER);
                                 
                var VRes1 = vA1.add(vB1);
                var VRes2 = vA2.add(vB2);
                                 
                VRes1.intoMemorySegment(resArray.getData(), i * INT_BYTES, ORDER);
                VRes2.intoMemorySegment(resArray.getData(), (i + INT_VL) * INT_BYTES, ORDER);
            }
            loopbound = SPECIESINT.loopBound(a.getSize());
            for (; i < loopbound; i += INT_VL) {
                var vA = IntVector.fromMemorySegment(SPECIESINT, a.getData(), i * INT_BYTES, ORDER);
                var vB = IntVector.fromMemorySegment(SPECIESINT, b.getData(), i * INT_BYTES, ORDER);
                var VRes = vA.add(vB);
                VRes.intoMemorySegment(resArray.getData(), i * INT_BYTES, ORDER);
            }
            for (; i < a.getSize(); i++) {
                int valA = a.getData().getAtIndex(ValueLayout.JAVA_INT, i);
                int valB = b.getData().getAtIndex(ValueLayout.JAVA_INT, i);
                resArray.getData().setAtIndex(ValueLayout.JAVA_INT, i, (int)(valA + valB));
            }
        } else {
            int vl = SPECIESINT.length();
            int[] mapA = new int[vl];
            int[] mapB = new int[vl];
            int[] mapRes = new int[vl];
                         
            int[] bufA = new int[vl];
            int[] bufB = new int[vl];
            int[] bufRes = new int[vl];
            var iterA = new NDIter(resArray.internalShapeUnsafe(), a.internalStridesUnsafe());
            var iterB = new NDIter(resArray.internalShapeUnsafe(), b.internalStridesUnsafe());
            var iterRes = new NDIter(resArray.internalShapeUnsafe(), resArray.internalStridesUnsafe());
            while (iterA.hasNext) {
                int validLanes = iterA.nextVector(mapA, vl);
                iterB.nextVector(mapB, vl);
                iterRes.nextVector(mapRes, vl);
                for(int k=0; k < validLanes; k++) {
                    bufA[k] = a.getData().getAtIndex(ValueLayout.JAVA_INT, mapA[k]);
                    bufB[k] = b.getData().getAtIndex(ValueLayout.JAVA_INT, mapB[k]);
                }
                var mask = SPECIESINT.indexInRange(0, validLanes);
                var vA = IntVector.fromArray(SPECIESINT, bufA, 0, mask);
                var vB = IntVector.fromArray(SPECIESINT, bufB, 0, mask);
                                 
                var vRes = vA.add(vB);
                                 
                vRes.intoArray(bufRes, 0, mask);
                for(int k=0; k < validLanes; k++) {
                    resArray.getData().setAtIndex(ValueLayout.JAVA_INT, mapRes[k], bufRes[k]);
                }
            }
        }
        return resArray;
    }


    public static NDArray addInt(NDArray a, int b, NDArray resArray) {
        var vB = IntVector.broadcast(SPECIESINT, b);
                 
        if (a.isContiguous() && resArray.isContiguous()) {
            long i = 0;
            long loopbound = a.getSize() - (a.getSize() % (INT_VL * 2));
                         
            for (; i < loopbound; i += INT_VL * 2) {
                var vA1 = IntVector.fromMemorySegment(SPECIESINT, a.getData(), i * INT_BYTES, ORDER);
                var vA2 = IntVector.fromMemorySegment(SPECIESINT, a.getData(), (i + INT_VL) * INT_BYTES, ORDER);
                var VRes1 = vA1.add(vB);
                var VRes2 = vA2.add(vB);
                                 
                VRes1.intoMemorySegment(resArray.getData(), i * INT_BYTES, ORDER);
                VRes2.intoMemorySegment(resArray.getData(), (i + INT_VL) * INT_BYTES, ORDER);
            }
            loopbound = SPECIESINT.loopBound(a.getSize());
            for (; i < loopbound; i += INT_VL) {
                var vA = IntVector.fromMemorySegment(SPECIESINT, a.getData(), i * INT_BYTES, ORDER);
                var VRes = vA.add(vB);
                VRes.intoMemorySegment(resArray.getData(), i * INT_BYTES, ORDER);
            }
            for (; i < a.getSize(); i++) {
                int valA = a.getData().getAtIndex(ValueLayout.JAVA_INT, i);
                resArray.getData().setAtIndex(ValueLayout.JAVA_INT, i, (int)(valA + b));
            }
        } else {
            int vl = SPECIESINT.length();
            int[] mapA = new int[vl];
            int[] mapRes = new int[vl];
                         
            int[] bufA = new int[vl];
            int[] bufRes = new int[vl];
            var iterA = new NDIter(resArray.internalShapeUnsafe(), a.internalStridesUnsafe());
            var iterRes = new NDIter(resArray.internalShapeUnsafe(), resArray.internalStridesUnsafe());
            while (iterA.hasNext) {
                int validLanes = iterA.nextVector(mapA, vl);
                iterRes.nextVector(mapRes, vl);
                for(int k=0; k < validLanes; k++) {
                    bufA[k] = a.getData().getAtIndex(ValueLayout.JAVA_INT, mapA[k]);
                }
                var mask = SPECIESINT.indexInRange(0, validLanes);
                var vA = IntVector.fromArray(SPECIESINT, bufA, 0, mask);
                                 
                var vRes = vA.add(vB);
                                 
                vRes.intoArray(bufRes, 0, mask);
                for(int k=0; k < validLanes; k++) {
                    resArray.getData().setAtIndex(ValueLayout.JAVA_INT, mapRes[k], bufRes[k]);
                }
            }
        }
        return resArray;
    }


    public static NDArray subFloat(NDArray a, NDArray b, NDArray resArray) {
        if (a.isContiguous() && b.isContiguous() && resArray.isContiguous()) {
            long i = 0;
            long loopbound = a.getSize() - (a.getSize() % (VL * 2));
                         
            for (; i < loopbound; i += VL * 2) {
                var vA1 = FloatVector.fromMemorySegment(SPECIES, a.getData(), i * FLOAT_BYTES, ORDER);
                var vA2 = FloatVector.fromMemorySegment(SPECIES, a.getData(), (i + VL) * FLOAT_BYTES, ORDER);
                var vB1 = FloatVector.fromMemorySegment(SPECIES, b.getData(), i * FLOAT_BYTES, ORDER);
                var vB2 = FloatVector.fromMemorySegment(SPECIES, b.getData(), (i + VL) * FLOAT_BYTES, ORDER);
                                 
                var VRes1 = vA1.sub(vB1);
                var VRes2 = vA2.sub(vB2);
                                 
                VRes1.intoMemorySegment(resArray.getData(), i * FLOAT_BYTES, ORDER);
                VRes2.intoMemorySegment(resArray.getData(), (i + VL) * FLOAT_BYTES, ORDER);
            }
            loopbound = SPECIES.loopBound(a.getSize());
            for (; i < loopbound; i += VL) {
                var vA = FloatVector.fromMemorySegment(SPECIES, a.getData(), i * FLOAT_BYTES, ORDER);
                var vB = FloatVector.fromMemorySegment(SPECIES, b.getData(), i * FLOAT_BYTES, ORDER);
                var VRes = vA.sub(vB);
                VRes.intoMemorySegment(resArray.getData(), i * FLOAT_BYTES, ORDER);
            }
            for (; i < a.getSize(); i++) {
                float valA = a.getData().getAtIndex(ValueLayout.JAVA_FLOAT, i);
                float valB = b.getData().getAtIndex(ValueLayout.JAVA_FLOAT, i);
                resArray.getData().setAtIndex(ValueLayout.JAVA_FLOAT, i, (float)(valA - valB));
            }
        } else {
            int vl = SPECIES.length();
            int[] mapA = new int[vl];
            int[] mapB = new int[vl];
            int[] mapRes = new int[vl];
                         
            float[] bufA = new float[vl];
            float[] bufB = new float[vl];
            float[] bufRes = new float[vl];
            var iterA = new NDIter(resArray.internalShapeUnsafe(), a.internalStridesUnsafe());
            var iterB = new NDIter(resArray.internalShapeUnsafe(), b.internalStridesUnsafe());
            var iterRes = new NDIter(resArray.internalShapeUnsafe(), resArray.internalStridesUnsafe());
            while (iterA.hasNext) {
                int validLanes = iterA.nextVector(mapA, vl);
                iterB.nextVector(mapB, vl);
                iterRes.nextVector(mapRes, vl);
                for(int k=0; k < validLanes; k++) {
                    bufA[k] = a.getData().getAtIndex(ValueLayout.JAVA_FLOAT, mapA[k]);
                    bufB[k] = b.getData().getAtIndex(ValueLayout.JAVA_FLOAT, mapB[k]);
                }
                var mask = SPECIES.indexInRange(0, validLanes);
                var vA = FloatVector.fromArray(SPECIES, bufA, 0, mask);
                var vB = FloatVector.fromArray(SPECIES, bufB, 0, mask);
                                 
                var vRes = vA.sub(vB);
                                 
                vRes.intoArray(bufRes, 0, mask);
                for(int k=0; k < validLanes; k++) {
                    resArray.getData().setAtIndex(ValueLayout.JAVA_FLOAT, mapRes[k], bufRes[k]);
                }
            }
        }
        return resArray;
    }


    public static NDArray subFloat(NDArray a, float b, NDArray resArray) {
        var vB = FloatVector.broadcast(SPECIES, b);
                 
        if (a.isContiguous() && resArray.isContiguous()) {
            long i = 0;
            long loopbound = a.getSize() - (a.getSize() % (VL * 2));
                         
            for (; i < loopbound; i += VL * 2) {
                var vA1 = FloatVector.fromMemorySegment(SPECIES, a.getData(), i * FLOAT_BYTES, ORDER);
                var vA2 = FloatVector.fromMemorySegment(SPECIES, a.getData(), (i + VL) * FLOAT_BYTES, ORDER);
                var VRes1 = vA1.sub(vB);
                var VRes2 = vA2.sub(vB);
                                 
                VRes1.intoMemorySegment(resArray.getData(), i * FLOAT_BYTES, ORDER);
                VRes2.intoMemorySegment(resArray.getData(), (i + VL) * FLOAT_BYTES, ORDER);
            }
            loopbound = SPECIES.loopBound(a.getSize());
            for (; i < loopbound; i += VL) {
                var vA = FloatVector.fromMemorySegment(SPECIES, a.getData(), i * FLOAT_BYTES, ORDER);
                var VRes = vA.sub(vB);
                VRes.intoMemorySegment(resArray.getData(), i * FLOAT_BYTES, ORDER);
            }
            for (; i < a.getSize(); i++) {
                float valA = a.getData().getAtIndex(ValueLayout.JAVA_FLOAT, i);
                resArray.getData().setAtIndex(ValueLayout.JAVA_FLOAT, i, (float)(valA - b));
            }
        } else {
            int vl = SPECIES.length();
            int[] mapA = new int[vl];
            int[] mapRes = new int[vl];
                         
            float[] bufA = new float[vl];
            float[] bufRes = new float[vl];
            var iterA = new NDIter(resArray.internalShapeUnsafe(), a.internalStridesUnsafe());
            var iterRes = new NDIter(resArray.internalShapeUnsafe(), resArray.internalStridesUnsafe());
            while (iterA.hasNext) {
                int validLanes = iterA.nextVector(mapA, vl);
                iterRes.nextVector(mapRes, vl);
                for(int k=0; k < validLanes; k++) {
                    bufA[k] = a.getData().getAtIndex(ValueLayout.JAVA_FLOAT, mapA[k]);
                }
                var mask = SPECIES.indexInRange(0, validLanes);
                var vA = FloatVector.fromArray(SPECIES, bufA, 0, mask);
                                 
                var vRes = vA.sub(vB);
                                 
                vRes.intoArray(bufRes, 0, mask);
                for(int k=0; k < validLanes; k++) {
                    resArray.getData().setAtIndex(ValueLayout.JAVA_FLOAT, mapRes[k], bufRes[k]);
                }
            }
        }
        return resArray;
    }


    public static NDArray subDouble(NDArray a, NDArray b, NDArray resArray) {
        if (a.isContiguous() && b.isContiguous() && resArray.isContiguous()) {
            long i = 0;
            long loopbound = a.getSize() - (a.getSize() % (DB_VL * 2));
                         
            for (; i < loopbound; i += DB_VL * 2) {
                var vA1 = DoubleVector.fromMemorySegment(SPECIESDB, a.getData(), i * DB_BYTES, ORDER);
                var vA2 = DoubleVector.fromMemorySegment(SPECIESDB, a.getData(), (i + DB_VL) * DB_BYTES, ORDER);
                var vB1 = DoubleVector.fromMemorySegment(SPECIESDB, b.getData(), i * DB_BYTES, ORDER);
                var vB2 = DoubleVector.fromMemorySegment(SPECIESDB, b.getData(), (i + DB_VL) * DB_BYTES, ORDER);
                                 
                var VRes1 = vA1.sub(vB1);
                var VRes2 = vA2.sub(vB2);
                                 
                VRes1.intoMemorySegment(resArray.getData(), i * DB_BYTES, ORDER);
                VRes2.intoMemorySegment(resArray.getData(), (i + DB_VL) * DB_BYTES, ORDER);
            }
            loopbound = SPECIESDB.loopBound(a.getSize());
            for (; i < loopbound; i += DB_VL) {
                var vA = DoubleVector.fromMemorySegment(SPECIESDB, a.getData(), i * DB_BYTES, ORDER);
                var vB = DoubleVector.fromMemorySegment(SPECIESDB, b.getData(), i * DB_BYTES, ORDER);
                var VRes = vA.sub(vB);
                VRes.intoMemorySegment(resArray.getData(), i * DB_BYTES, ORDER);
            }
            for (; i < a.getSize(); i++) {
                double valA = a.getData().getAtIndex(ValueLayout.JAVA_DOUBLE, i);
                double valB = b.getData().getAtIndex(ValueLayout.JAVA_DOUBLE, i);
                resArray.getData().setAtIndex(ValueLayout.JAVA_DOUBLE, i, (double)(valA - valB));
            }
        } else {
            int vl = SPECIESDB.length();
            int[] mapA = new int[vl];
            int[] mapB = new int[vl];
            int[] mapRes = new int[vl];
                         
            double[] bufA = new double[vl];
            double[] bufB = new double[vl];
            double[] bufRes = new double[vl];
            var iterA = new NDIter(resArray.internalShapeUnsafe(), a.internalStridesUnsafe());
            var iterB = new NDIter(resArray.internalShapeUnsafe(), b.internalStridesUnsafe());
            var iterRes = new NDIter(resArray.internalShapeUnsafe(), resArray.internalStridesUnsafe());
            while (iterA.hasNext) {
                int validLanes = iterA.nextVector(mapA, vl);
                iterB.nextVector(mapB, vl);
                iterRes.nextVector(mapRes, vl);
                for(int k=0; k < validLanes; k++) {
                    bufA[k] = a.getData().getAtIndex(ValueLayout.JAVA_DOUBLE, mapA[k]);
                    bufB[k] = b.getData().getAtIndex(ValueLayout.JAVA_DOUBLE, mapB[k]);
                }
                var mask = SPECIESDB.indexInRange(0, validLanes);
                var vA = DoubleVector.fromArray(SPECIESDB, bufA, 0, mask);
                var vB = DoubleVector.fromArray(SPECIESDB, bufB, 0, mask);
                                 
                var vRes = vA.sub(vB);
                                 
                vRes.intoArray(bufRes, 0, mask);
                for(int k=0; k < validLanes; k++) {
                    resArray.getData().setAtIndex(ValueLayout.JAVA_DOUBLE, mapRes[k], bufRes[k]);
                }
            }
        }
        return resArray;
    }


    public static NDArray subDouble(NDArray a, double b, NDArray resArray) {
        var vB = DoubleVector.broadcast(SPECIESDB, b);
                 
        if (a.isContiguous() && resArray.isContiguous()) {
            long i = 0;
            long loopbound = a.getSize() - (a.getSize() % (DB_VL * 2));
                         
            for (; i < loopbound; i += DB_VL * 2) {
                var vA1 = DoubleVector.fromMemorySegment(SPECIESDB, a.getData(), i * DB_BYTES, ORDER);
                var vA2 = DoubleVector.fromMemorySegment(SPECIESDB, a.getData(), (i + DB_VL) * DB_BYTES, ORDER);
                var VRes1 = vA1.sub(vB);
                var VRes2 = vA2.sub(vB);
                                 
                VRes1.intoMemorySegment(resArray.getData(), i * DB_BYTES, ORDER);
                VRes2.intoMemorySegment(resArray.getData(), (i + DB_VL) * DB_BYTES, ORDER);
            }
            loopbound = SPECIESDB.loopBound(a.getSize());
            for (; i < loopbound; i += DB_VL) {
                var vA = DoubleVector.fromMemorySegment(SPECIESDB, a.getData(), i * DB_BYTES, ORDER);
                var VRes = vA.sub(vB);
                VRes.intoMemorySegment(resArray.getData(), i * DB_BYTES, ORDER);
            }
            for (; i < a.getSize(); i++) {
                double valA = a.getData().getAtIndex(ValueLayout.JAVA_DOUBLE, i);
                resArray.getData().setAtIndex(ValueLayout.JAVA_DOUBLE, i, (double)(valA - b));
            }
        } else {
            int vl = SPECIESDB.length();
            int[] mapA = new int[vl];
            int[] mapRes = new int[vl];
                         
            double[] bufA = new double[vl];
            double[] bufRes = new double[vl];
            var iterA = new NDIter(resArray.internalShapeUnsafe(), a.internalStridesUnsafe());
            var iterRes = new NDIter(resArray.internalShapeUnsafe(), resArray.internalStridesUnsafe());
            while (iterA.hasNext) {
                int validLanes = iterA.nextVector(mapA, vl);
                iterRes.nextVector(mapRes, vl);
                for(int k=0; k < validLanes; k++) {
                    bufA[k] = a.getData().getAtIndex(ValueLayout.JAVA_DOUBLE, mapA[k]);
                }
                var mask = SPECIESDB.indexInRange(0, validLanes);
                var vA = DoubleVector.fromArray(SPECIESDB, bufA, 0, mask);
                                 
                var vRes = vA.sub(vB);
                                 
                vRes.intoArray(bufRes, 0, mask);
                for(int k=0; k < validLanes; k++) {
                    resArray.getData().setAtIndex(ValueLayout.JAVA_DOUBLE, mapRes[k], bufRes[k]);
                }
            }
        }
        return resArray;
    }


    public static NDArray subInt(NDArray a, NDArray b, NDArray resArray) {
        if (a.isContiguous() && b.isContiguous() && resArray.isContiguous()) {
            long i = 0;
            long loopbound = a.getSize() - (a.getSize() % (INT_VL * 2));
                         
            for (; i < loopbound; i += INT_VL * 2) {
                var vA1 = IntVector.fromMemorySegment(SPECIESINT, a.getData(), i * INT_BYTES, ORDER);
                var vA2 = IntVector.fromMemorySegment(SPECIESINT, a.getData(), (i + INT_VL) * INT_BYTES, ORDER);
                var vB1 = IntVector.fromMemorySegment(SPECIESINT, b.getData(), i * INT_BYTES, ORDER);
                var vB2 = IntVector.fromMemorySegment(SPECIESINT, b.getData(), (i + INT_VL) * INT_BYTES, ORDER);
                                 
                var VRes1 = vA1.sub(vB1);
                var VRes2 = vA2.sub(vB2);
                                 
                VRes1.intoMemorySegment(resArray.getData(), i * INT_BYTES, ORDER);
                VRes2.intoMemorySegment(resArray.getData(), (i + INT_VL) * INT_BYTES, ORDER);
            }
            loopbound = SPECIESINT.loopBound(a.getSize());
            for (; i < loopbound; i += INT_VL) {
                var vA = IntVector.fromMemorySegment(SPECIESINT, a.getData(), i * INT_BYTES, ORDER);
                var vB = IntVector.fromMemorySegment(SPECIESINT, b.getData(), i * INT_BYTES, ORDER);
                var VRes = vA.sub(vB);
                VRes.intoMemorySegment(resArray.getData(), i * INT_BYTES, ORDER);
            }
            for (; i < a.getSize(); i++) {
                int valA = a.getData().getAtIndex(ValueLayout.JAVA_INT, i);
                int valB = b.getData().getAtIndex(ValueLayout.JAVA_INT, i);
                resArray.getData().setAtIndex(ValueLayout.JAVA_INT, i, (int)(valA - valB));
            }
        } else {
            int vl = SPECIESINT.length();
            int[] mapA = new int[vl];
            int[] mapB = new int[vl];
            int[] mapRes = new int[vl];
                         
            int[] bufA = new int[vl];
            int[] bufB = new int[vl];
            int[] bufRes = new int[vl];
            var iterA = new NDIter(resArray.internalShapeUnsafe(), a.internalStridesUnsafe());
            var iterB = new NDIter(resArray.internalShapeUnsafe(), b.internalStridesUnsafe());
            var iterRes = new NDIter(resArray.internalShapeUnsafe(), resArray.internalStridesUnsafe());
            while (iterA.hasNext) {
                int validLanes = iterA.nextVector(mapA, vl);
                iterB.nextVector(mapB, vl);
                iterRes.nextVector(mapRes, vl);
                for(int k=0; k < validLanes; k++) {
                    bufA[k] = a.getData().getAtIndex(ValueLayout.JAVA_INT, mapA[k]);
                    bufB[k] = b.getData().getAtIndex(ValueLayout.JAVA_INT, mapB[k]);
                }
                var mask = SPECIESINT.indexInRange(0, validLanes);
                var vA = IntVector.fromArray(SPECIESINT, bufA, 0, mask);
                var vB = IntVector.fromArray(SPECIESINT, bufB, 0, mask);
                                 
                var vRes = vA.sub(vB);
                                 
                vRes.intoArray(bufRes, 0, mask);
                for(int k=0; k < validLanes; k++) {
                    resArray.getData().setAtIndex(ValueLayout.JAVA_INT, mapRes[k], bufRes[k]);
                }
            }
        }
        return resArray;
    }


    public static NDArray subInt(NDArray a, int b, NDArray resArray) {
        var vB = IntVector.broadcast(SPECIESINT, b);
                 
        if (a.isContiguous() && resArray.isContiguous()) {
            long i = 0;
            long loopbound = a.getSize() - (a.getSize() % (INT_VL * 2));
                         
            for (; i < loopbound; i += INT_VL * 2) {
                var vA1 = IntVector.fromMemorySegment(SPECIESINT, a.getData(), i * INT_BYTES, ORDER);
                var vA2 = IntVector.fromMemorySegment(SPECIESINT, a.getData(), (i + INT_VL) * INT_BYTES, ORDER);
                var VRes1 = vA1.sub(vB);
                var VRes2 = vA2.sub(vB);
                                 
                VRes1.intoMemorySegment(resArray.getData(), i * INT_BYTES, ORDER);
                VRes2.intoMemorySegment(resArray.getData(), (i + INT_VL) * INT_BYTES, ORDER);
            }
            loopbound = SPECIESINT.loopBound(a.getSize());
            for (; i < loopbound; i += INT_VL) {
                var vA = IntVector.fromMemorySegment(SPECIESINT, a.getData(), i * INT_BYTES, ORDER);
                var VRes = vA.sub(vB);
                VRes.intoMemorySegment(resArray.getData(), i * INT_BYTES, ORDER);
            }
            for (; i < a.getSize(); i++) {
                int valA = a.getData().getAtIndex(ValueLayout.JAVA_INT, i);
                resArray.getData().setAtIndex(ValueLayout.JAVA_INT, i, (int)(valA - b));
            }
        } else {
            int vl = SPECIESINT.length();
            int[] mapA = new int[vl];
            int[] mapRes = new int[vl];
                         
            int[] bufA = new int[vl];
            int[] bufRes = new int[vl];
            var iterA = new NDIter(resArray.internalShapeUnsafe(), a.internalStridesUnsafe());
            var iterRes = new NDIter(resArray.internalShapeUnsafe(), resArray.internalStridesUnsafe());
            while (iterA.hasNext) {
                int validLanes = iterA.nextVector(mapA, vl);
                iterRes.nextVector(mapRes, vl);
                for(int k=0; k < validLanes; k++) {
                    bufA[k] = a.getData().getAtIndex(ValueLayout.JAVA_INT, mapA[k]);
                }
                var mask = SPECIESINT.indexInRange(0, validLanes);
                var vA = IntVector.fromArray(SPECIESINT, bufA, 0, mask);
                                 
                var vRes = vA.sub(vB);
                                 
                vRes.intoArray(bufRes, 0, mask);
                for(int k=0; k < validLanes; k++) {
                    resArray.getData().setAtIndex(ValueLayout.JAVA_INT, mapRes[k], bufRes[k]);
                }
            }
        }
        return resArray;
    }


    public static NDArray mulFloat(NDArray a, NDArray b, NDArray resArray) {
        if (a.isContiguous() && b.isContiguous() && resArray.isContiguous()) {
            long i = 0;
            long loopbound = a.getSize() - (a.getSize() % (VL * 2));
                         
            for (; i < loopbound; i += VL * 2) {
                var vA1 = FloatVector.fromMemorySegment(SPECIES, a.getData(), i * FLOAT_BYTES, ORDER);
                var vA2 = FloatVector.fromMemorySegment(SPECIES, a.getData(), (i + VL) * FLOAT_BYTES, ORDER);
                var vB1 = FloatVector.fromMemorySegment(SPECIES, b.getData(), i * FLOAT_BYTES, ORDER);
                var vB2 = FloatVector.fromMemorySegment(SPECIES, b.getData(), (i + VL) * FLOAT_BYTES, ORDER);
                                 
                var VRes1 = vA1.mul(vB1);
                var VRes2 = vA2.mul(vB2);
                                 
                VRes1.intoMemorySegment(resArray.getData(), i * FLOAT_BYTES, ORDER);
                VRes2.intoMemorySegment(resArray.getData(), (i + VL) * FLOAT_BYTES, ORDER);
            }
            loopbound = SPECIES.loopBound(a.getSize());
            for (; i < loopbound; i += VL) {
                var vA = FloatVector.fromMemorySegment(SPECIES, a.getData(), i * FLOAT_BYTES, ORDER);
                var vB = FloatVector.fromMemorySegment(SPECIES, b.getData(), i * FLOAT_BYTES, ORDER);
                var VRes = vA.mul(vB);
                VRes.intoMemorySegment(resArray.getData(), i * FLOAT_BYTES, ORDER);
            }
            for (; i < a.getSize(); i++) {
                float valA = a.getData().getAtIndex(ValueLayout.JAVA_FLOAT, i);
                float valB = b.getData().getAtIndex(ValueLayout.JAVA_FLOAT, i);
                resArray.getData().setAtIndex(ValueLayout.JAVA_FLOAT, i, (float)(valA * valB));
            }
        } else {
            int vl = SPECIES.length();
            int[] mapA = new int[vl];
            int[] mapB = new int[vl];
            int[] mapRes = new int[vl];
                         
            float[] bufA = new float[vl];
            float[] bufB = new float[vl];
            float[] bufRes = new float[vl];
            var iterA = new NDIter(resArray.internalShapeUnsafe(), a.internalStridesUnsafe());
            var iterB = new NDIter(resArray.internalShapeUnsafe(), b.internalStridesUnsafe());
            var iterRes = new NDIter(resArray.internalShapeUnsafe(), resArray.internalStridesUnsafe());
            while (iterA.hasNext) {
                int validLanes = iterA.nextVector(mapA, vl);
                iterB.nextVector(mapB, vl);
                iterRes.nextVector(mapRes, vl);
                for(int k=0; k < validLanes; k++) {
                    bufA[k] = a.getData().getAtIndex(ValueLayout.JAVA_FLOAT, mapA[k]);
                    bufB[k] = b.getData().getAtIndex(ValueLayout.JAVA_FLOAT, mapB[k]);
                }
                var mask = SPECIES.indexInRange(0, validLanes);
                var vA = FloatVector.fromArray(SPECIES, bufA, 0, mask);
                var vB = FloatVector.fromArray(SPECIES, bufB, 0, mask);
                                 
                var vRes = vA.mul(vB);
                                 
                vRes.intoArray(bufRes, 0, mask);
                for(int k=0; k < validLanes; k++) {
                    resArray.getData().setAtIndex(ValueLayout.JAVA_FLOAT, mapRes[k], bufRes[k]);
                }
            }
        }
        return resArray;
    }


    public static NDArray mulFloat(NDArray a, float b, NDArray resArray) {
        var vB = FloatVector.broadcast(SPECIES, b);
                 
        if (a.isContiguous() && resArray.isContiguous()) {
            long i = 0;
            long loopbound = a.getSize() - (a.getSize() % (VL * 2));
                         
            for (; i < loopbound; i += VL * 2) {
                var vA1 = FloatVector.fromMemorySegment(SPECIES, a.getData(), i * FLOAT_BYTES, ORDER);
                var vA2 = FloatVector.fromMemorySegment(SPECIES, a.getData(), (i + VL) * FLOAT_BYTES, ORDER);
                var VRes1 = vA1.mul(vB);
                var VRes2 = vA2.mul(vB);
                                 
                VRes1.intoMemorySegment(resArray.getData(), i * FLOAT_BYTES, ORDER);
                VRes2.intoMemorySegment(resArray.getData(), (i + VL) * FLOAT_BYTES, ORDER);
            }
            loopbound = SPECIES.loopBound(a.getSize());
            for (; i < loopbound; i += VL) {
                var vA = FloatVector.fromMemorySegment(SPECIES, a.getData(), i * FLOAT_BYTES, ORDER);
                var VRes = vA.mul(vB);
                VRes.intoMemorySegment(resArray.getData(), i * FLOAT_BYTES, ORDER);
            }
            for (; i < a.getSize(); i++) {
                float valA = a.getData().getAtIndex(ValueLayout.JAVA_FLOAT, i);
                resArray.getData().setAtIndex(ValueLayout.JAVA_FLOAT, i, (float)(valA * b));
            }
        } else {
            int vl = SPECIES.length();
            int[] mapA = new int[vl];
            int[] mapRes = new int[vl];
                         
            float[] bufA = new float[vl];
            float[] bufRes = new float[vl];
            var iterA = new NDIter(resArray.internalShapeUnsafe(), a.internalStridesUnsafe());
            var iterRes = new NDIter(resArray.internalShapeUnsafe(), resArray.internalStridesUnsafe());
            while (iterA.hasNext) {
                int validLanes = iterA.nextVector(mapA, vl);
                iterRes.nextVector(mapRes, vl);
                for(int k=0; k < validLanes; k++) {
                    bufA[k] = a.getData().getAtIndex(ValueLayout.JAVA_FLOAT, mapA[k]);
                }
                var mask = SPECIES.indexInRange(0, validLanes);
                var vA = FloatVector.fromArray(SPECIES, bufA, 0, mask);
                                 
                var vRes = vA.mul(vB);
                                 
                vRes.intoArray(bufRes, 0, mask);
                for(int k=0; k < validLanes; k++) {
                    resArray.getData().setAtIndex(ValueLayout.JAVA_FLOAT, mapRes[k], bufRes[k]);
                }
            }
        }
        return resArray;
    }


    public static NDArray mulDouble(NDArray a, NDArray b, NDArray resArray) {
        if (a.isContiguous() && b.isContiguous() && resArray.isContiguous()) {
            long i = 0;
            long loopbound = a.getSize() - (a.getSize() % (DB_VL * 2));
                         
            for (; i < loopbound; i += DB_VL * 2) {
                var vA1 = DoubleVector.fromMemorySegment(SPECIESDB, a.getData(), i * DB_BYTES, ORDER);
                var vA2 = DoubleVector.fromMemorySegment(SPECIESDB, a.getData(), (i + DB_VL) * DB_BYTES, ORDER);
                var vB1 = DoubleVector.fromMemorySegment(SPECIESDB, b.getData(), i * DB_BYTES, ORDER);
                var vB2 = DoubleVector.fromMemorySegment(SPECIESDB, b.getData(), (i + DB_VL) * DB_BYTES, ORDER);
                                 
                var VRes1 = vA1.mul(vB1);
                var VRes2 = vA2.mul(vB2);
                                 
                VRes1.intoMemorySegment(resArray.getData(), i * DB_BYTES, ORDER);
                VRes2.intoMemorySegment(resArray.getData(), (i + DB_VL) * DB_BYTES, ORDER);
            }
            loopbound = SPECIESDB.loopBound(a.getSize());
            for (; i < loopbound; i += DB_VL) {
                var vA = DoubleVector.fromMemorySegment(SPECIESDB, a.getData(), i * DB_BYTES, ORDER);
                var vB = DoubleVector.fromMemorySegment(SPECIESDB, b.getData(), i * DB_BYTES, ORDER);
                var VRes = vA.mul(vB);
                VRes.intoMemorySegment(resArray.getData(), i * DB_BYTES, ORDER);
            }
            for (; i < a.getSize(); i++) {
                double valA = a.getData().getAtIndex(ValueLayout.JAVA_DOUBLE, i);
                double valB = b.getData().getAtIndex(ValueLayout.JAVA_DOUBLE, i);
                resArray.getData().setAtIndex(ValueLayout.JAVA_DOUBLE, i, (double)(valA * valB));
            }
        } else {
            int vl = SPECIESDB.length();
            int[] mapA = new int[vl];
            int[] mapB = new int[vl];
            int[] mapRes = new int[vl];
                         
            double[] bufA = new double[vl];
            double[] bufB = new double[vl];
            double[] bufRes = new double[vl];
            var iterA = new NDIter(resArray.internalShapeUnsafe(), a.internalStridesUnsafe());
            var iterB = new NDIter(resArray.internalShapeUnsafe(), b.internalStridesUnsafe());
            var iterRes = new NDIter(resArray.internalShapeUnsafe(), resArray.internalStridesUnsafe());
            while (iterA.hasNext) {
                int validLanes = iterA.nextVector(mapA, vl);
                iterB.nextVector(mapB, vl);
                iterRes.nextVector(mapRes, vl);
                for(int k=0; k < validLanes; k++) {
                    bufA[k] = a.getData().getAtIndex(ValueLayout.JAVA_DOUBLE, mapA[k]);
                    bufB[k] = b.getData().getAtIndex(ValueLayout.JAVA_DOUBLE, mapB[k]);
                }
                var mask = SPECIESDB.indexInRange(0, validLanes);
                var vA = DoubleVector.fromArray(SPECIESDB, bufA, 0, mask);
                var vB = DoubleVector.fromArray(SPECIESDB, bufB, 0, mask);
                                 
                var vRes = vA.mul(vB);
                                 
                vRes.intoArray(bufRes, 0, mask);
                for(int k=0; k < validLanes; k++) {
                    resArray.getData().setAtIndex(ValueLayout.JAVA_DOUBLE, mapRes[k], bufRes[k]);
                }
            }
        }
        return resArray;
    }


    public static NDArray mulDouble(NDArray a, double b, NDArray resArray) {
        var vB = DoubleVector.broadcast(SPECIESDB, b);
                 
        if (a.isContiguous() && resArray.isContiguous()) {
            long i = 0;
            long loopbound = a.getSize() - (a.getSize() % (DB_VL * 2));
                         
            for (; i < loopbound; i += DB_VL * 2) {
                var vA1 = DoubleVector.fromMemorySegment(SPECIESDB, a.getData(), i * DB_BYTES, ORDER);
                var vA2 = DoubleVector.fromMemorySegment(SPECIESDB, a.getData(), (i + DB_VL) * DB_BYTES, ORDER);
                var VRes1 = vA1.mul(vB);
                var VRes2 = vA2.mul(vB);
                                 
                VRes1.intoMemorySegment(resArray.getData(), i * DB_BYTES, ORDER);
                VRes2.intoMemorySegment(resArray.getData(), (i + DB_VL) * DB_BYTES, ORDER);
            }
            loopbound = SPECIESDB.loopBound(a.getSize());
            for (; i < loopbound; i += DB_VL) {
                var vA = DoubleVector.fromMemorySegment(SPECIESDB, a.getData(), i * DB_BYTES, ORDER);
                var VRes = vA.mul(vB);
                VRes.intoMemorySegment(resArray.getData(), i * DB_BYTES, ORDER);
            }
            for (; i < a.getSize(); i++) {
                double valA = a.getData().getAtIndex(ValueLayout.JAVA_DOUBLE, i);
                resArray.getData().setAtIndex(ValueLayout.JAVA_DOUBLE, i, (double)(valA * b));
            }
        } else {
            int vl = SPECIESDB.length();
            int[] mapA = new int[vl];
            int[] mapRes = new int[vl];
                         
            double[] bufA = new double[vl];
            double[] bufRes = new double[vl];
            var iterA = new NDIter(resArray.internalShapeUnsafe(), a.internalStridesUnsafe());
            var iterRes = new NDIter(resArray.internalShapeUnsafe(), resArray.internalStridesUnsafe());
            while (iterA.hasNext) {
                int validLanes = iterA.nextVector(mapA, vl);
                iterRes.nextVector(mapRes, vl);
                for(int k=0; k < validLanes; k++) {
                    bufA[k] = a.getData().getAtIndex(ValueLayout.JAVA_DOUBLE, mapA[k]);
                }
                var mask = SPECIESDB.indexInRange(0, validLanes);
                var vA = DoubleVector.fromArray(SPECIESDB, bufA, 0, mask);
                                 
                var vRes = vA.mul(vB);
                                 
                vRes.intoArray(bufRes, 0, mask);
                for(int k=0; k < validLanes; k++) {
                    resArray.getData().setAtIndex(ValueLayout.JAVA_DOUBLE, mapRes[k], bufRes[k]);
                }
            }
        }
        return resArray;
    }


    public static NDArray mulInt(NDArray a, NDArray b, NDArray resArray) {
        if (a.isContiguous() && b.isContiguous() && resArray.isContiguous()) {
            long i = 0;
            long loopbound = a.getSize() - (a.getSize() % (INT_VL * 2));
                         
            for (; i < loopbound; i += INT_VL * 2) {
                var vA1 = IntVector.fromMemorySegment(SPECIESINT, a.getData(), i * INT_BYTES, ORDER);
                var vA2 = IntVector.fromMemorySegment(SPECIESINT, a.getData(), (i + INT_VL) * INT_BYTES, ORDER);
                var vB1 = IntVector.fromMemorySegment(SPECIESINT, b.getData(), i * INT_BYTES, ORDER);
                var vB2 = IntVector.fromMemorySegment(SPECIESINT, b.getData(), (i + INT_VL) * INT_BYTES, ORDER);
                                 
                var VRes1 = vA1.mul(vB1);
                var VRes2 = vA2.mul(vB2);
                                 
                VRes1.intoMemorySegment(resArray.getData(), i * INT_BYTES, ORDER);
                VRes2.intoMemorySegment(resArray.getData(), (i + INT_VL) * INT_BYTES, ORDER);
            }
            loopbound = SPECIESINT.loopBound(a.getSize());
            for (; i < loopbound; i += INT_VL) {
                var vA = IntVector.fromMemorySegment(SPECIESINT, a.getData(), i * INT_BYTES, ORDER);
                var vB = IntVector.fromMemorySegment(SPECIESINT, b.getData(), i * INT_BYTES, ORDER);
                var VRes = vA.mul(vB);
                VRes.intoMemorySegment(resArray.getData(), i * INT_BYTES, ORDER);
            }
            for (; i < a.getSize(); i++) {
                int valA = a.getData().getAtIndex(ValueLayout.JAVA_INT, i);
                int valB = b.getData().getAtIndex(ValueLayout.JAVA_INT, i);
                resArray.getData().setAtIndex(ValueLayout.JAVA_INT, i, (int)(valA * valB));
            }
        } else {
            int vl = SPECIESINT.length();
            int[] mapA = new int[vl];
            int[] mapB = new int[vl];
            int[] mapRes = new int[vl];
                         
            int[] bufA = new int[vl];
            int[] bufB = new int[vl];
            int[] bufRes = new int[vl];
            var iterA = new NDIter(resArray.internalShapeUnsafe(), a.internalStridesUnsafe());
            var iterB = new NDIter(resArray.internalShapeUnsafe(), b.internalStridesUnsafe());
            var iterRes = new NDIter(resArray.internalShapeUnsafe(), resArray.internalStridesUnsafe());
            while (iterA.hasNext) {
                int validLanes = iterA.nextVector(mapA, vl);
                iterB.nextVector(mapB, vl);
                iterRes.nextVector(mapRes, vl);
                for(int k=0; k < validLanes; k++) {
                    bufA[k] = a.getData().getAtIndex(ValueLayout.JAVA_INT, mapA[k]);
                    bufB[k] = b.getData().getAtIndex(ValueLayout.JAVA_INT, mapB[k]);
                }
                var mask = SPECIESINT.indexInRange(0, validLanes);
                var vA = IntVector.fromArray(SPECIESINT, bufA, 0, mask);
                var vB = IntVector.fromArray(SPECIESINT, bufB, 0, mask);
                                 
                var vRes = vA.mul(vB);
                                 
                vRes.intoArray(bufRes, 0, mask);
                for(int k=0; k < validLanes; k++) {
                    resArray.getData().setAtIndex(ValueLayout.JAVA_INT, mapRes[k], bufRes[k]);
                }
            }
        }
        return resArray;
    }


    public static NDArray mulInt(NDArray a, int b, NDArray resArray) {
        var vB = IntVector.broadcast(SPECIESINT, b);
                 
        if (a.isContiguous() && resArray.isContiguous()) {
            long i = 0;
            long loopbound = a.getSize() - (a.getSize() % (INT_VL * 2));
                         
            for (; i < loopbound; i += INT_VL * 2) {
                var vA1 = IntVector.fromMemorySegment(SPECIESINT, a.getData(), i * INT_BYTES, ORDER);
                var vA2 = IntVector.fromMemorySegment(SPECIESINT, a.getData(), (i + INT_VL) * INT_BYTES, ORDER);
                var VRes1 = vA1.mul(vB);
                var VRes2 = vA2.mul(vB);
                                 
                VRes1.intoMemorySegment(resArray.getData(), i * INT_BYTES, ORDER);
                VRes2.intoMemorySegment(resArray.getData(), (i + INT_VL) * INT_BYTES, ORDER);
            }
            loopbound = SPECIESINT.loopBound(a.getSize());
            for (; i < loopbound; i += INT_VL) {
                var vA = IntVector.fromMemorySegment(SPECIESINT, a.getData(), i * INT_BYTES, ORDER);
                var VRes = vA.mul(vB);
                VRes.intoMemorySegment(resArray.getData(), i * INT_BYTES, ORDER);
            }
            for (; i < a.getSize(); i++) {
                int valA = a.getData().getAtIndex(ValueLayout.JAVA_INT, i);
                resArray.getData().setAtIndex(ValueLayout.JAVA_INT, i, (int)(valA * b));
            }
        } else {
            int vl = SPECIESINT.length();
            int[] mapA = new int[vl];
            int[] mapRes = new int[vl];
                         
            int[] bufA = new int[vl];
            int[] bufRes = new int[vl];
            var iterA = new NDIter(resArray.internalShapeUnsafe(), a.internalStridesUnsafe());
            var iterRes = new NDIter(resArray.internalShapeUnsafe(), resArray.internalStridesUnsafe());
            while (iterA.hasNext) {
                int validLanes = iterA.nextVector(mapA, vl);
                iterRes.nextVector(mapRes, vl);
                for(int k=0; k < validLanes; k++) {
                    bufA[k] = a.getData().getAtIndex(ValueLayout.JAVA_INT, mapA[k]);
                }
                var mask = SPECIESINT.indexInRange(0, validLanes);
                var vA = IntVector.fromArray(SPECIESINT, bufA, 0, mask);
                                 
                var vRes = vA.mul(vB);
                                 
                vRes.intoArray(bufRes, 0, mask);
                for(int k=0; k < validLanes; k++) {
                    resArray.getData().setAtIndex(ValueLayout.JAVA_INT, mapRes[k], bufRes[k]);
                }
            }
        }
        return resArray;
    }


    public static NDArray divFloat(NDArray a, NDArray b, NDArray resArray) {
        if (a.isContiguous() && b.isContiguous() && resArray.isContiguous()) {
            long i = 0;
            long loopbound = a.getSize() - (a.getSize() % (VL * 2));
                         
            for (; i < loopbound; i += VL * 2) {
                var vA1 = FloatVector.fromMemorySegment(SPECIES, a.getData(), i * FLOAT_BYTES, ORDER);
                var vA2 = FloatVector.fromMemorySegment(SPECIES, a.getData(), (i + VL) * FLOAT_BYTES, ORDER);
                var vB1 = FloatVector.fromMemorySegment(SPECIES, b.getData(), i * FLOAT_BYTES, ORDER);
                var vB2 = FloatVector.fromMemorySegment(SPECIES, b.getData(), (i + VL) * FLOAT_BYTES, ORDER);
                                 
                var VRes1 = vA1.div(vB1);
                var VRes2 = vA2.div(vB2);
                                 
                VRes1.intoMemorySegment(resArray.getData(), i * FLOAT_BYTES, ORDER);
                VRes2.intoMemorySegment(resArray.getData(), (i + VL) * FLOAT_BYTES, ORDER);
            }
            loopbound = SPECIES.loopBound(a.getSize());
            for (; i < loopbound; i += VL) {
                var vA = FloatVector.fromMemorySegment(SPECIES, a.getData(), i * FLOAT_BYTES, ORDER);
                var vB = FloatVector.fromMemorySegment(SPECIES, b.getData(), i * FLOAT_BYTES, ORDER);
                var VRes = vA.div(vB);
                VRes.intoMemorySegment(resArray.getData(), i * FLOAT_BYTES, ORDER);
            }
            for (; i < a.getSize(); i++) {
                float valA = a.getData().getAtIndex(ValueLayout.JAVA_FLOAT, i);
                float valB = b.getData().getAtIndex(ValueLayout.JAVA_FLOAT, i);
                resArray.getData().setAtIndex(ValueLayout.JAVA_FLOAT, i, (float)(valA / valB));
            }
        } else {
            int vl = SPECIES.length();
            int[] mapA = new int[vl];
            int[] mapB = new int[vl];
            int[] mapRes = new int[vl];
                         
            float[] bufA = new float[vl];
            float[] bufB = new float[vl];
            float[] bufRes = new float[vl];
            var iterA = new NDIter(resArray.internalShapeUnsafe(), a.internalStridesUnsafe());
            var iterB = new NDIter(resArray.internalShapeUnsafe(), b.internalStridesUnsafe());
            var iterRes = new NDIter(resArray.internalShapeUnsafe(), resArray.internalStridesUnsafe());
            while (iterA.hasNext) {
                int validLanes = iterA.nextVector(mapA, vl);
                iterB.nextVector(mapB, vl);
                iterRes.nextVector(mapRes, vl);
                for(int k=0; k < validLanes; k++) {
                    bufA[k] = a.getData().getAtIndex(ValueLayout.JAVA_FLOAT, mapA[k]);
                    bufB[k] = b.getData().getAtIndex(ValueLayout.JAVA_FLOAT, mapB[k]);
                }
                var mask = SPECIES.indexInRange(0, validLanes);
                var vA = FloatVector.fromArray(SPECIES, bufA, 0, mask);
                var vB = FloatVector.fromArray(SPECIES, bufB, 0, mask);
                                 
                var vRes = vA.div(vB);
                                 
                vRes.intoArray(bufRes, 0, mask);
                for(int k=0; k < validLanes; k++) {
                    resArray.getData().setAtIndex(ValueLayout.JAVA_FLOAT, mapRes[k], bufRes[k]);
                }
            }
        }
        return resArray;
    }


    public static NDArray divFloat(NDArray a, float b, NDArray resArray) {
        var vB = FloatVector.broadcast(SPECIES, b);
                 
        if (a.isContiguous() && resArray.isContiguous()) {
            long i = 0;
            long loopbound = a.getSize() - (a.getSize() % (VL * 2));
                         
            for (; i < loopbound; i += VL * 2) {
                var vA1 = FloatVector.fromMemorySegment(SPECIES, a.getData(), i * FLOAT_BYTES, ORDER);
                var vA2 = FloatVector.fromMemorySegment(SPECIES, a.getData(), (i + VL) * FLOAT_BYTES, ORDER);
                var VRes1 = vA1.div(vB);
                var VRes2 = vA2.div(vB);
                                 
                VRes1.intoMemorySegment(resArray.getData(), i * FLOAT_BYTES, ORDER);
                VRes2.intoMemorySegment(resArray.getData(), (i + VL) * FLOAT_BYTES, ORDER);
            }
            loopbound = SPECIES.loopBound(a.getSize());
            for (; i < loopbound; i += VL) {
                var vA = FloatVector.fromMemorySegment(SPECIES, a.getData(), i * FLOAT_BYTES, ORDER);
                var VRes = vA.div(vB);
                VRes.intoMemorySegment(resArray.getData(), i * FLOAT_BYTES, ORDER);
            }
            for (; i < a.getSize(); i++) {
                float valA = a.getData().getAtIndex(ValueLayout.JAVA_FLOAT, i);
                resArray.getData().setAtIndex(ValueLayout.JAVA_FLOAT, i, (float)(valA / b));
            }
        } else {
            int vl = SPECIES.length();
            int[] mapA = new int[vl];
            int[] mapRes = new int[vl];
                         
            float[] bufA = new float[vl];
            float[] bufRes = new float[vl];
            var iterA = new NDIter(resArray.internalShapeUnsafe(), a.internalStridesUnsafe());
            var iterRes = new NDIter(resArray.internalShapeUnsafe(), resArray.internalStridesUnsafe());
            while (iterA.hasNext) {
                int validLanes = iterA.nextVector(mapA, vl);
                iterRes.nextVector(mapRes, vl);
                for(int k=0; k < validLanes; k++) {
                    bufA[k] = a.getData().getAtIndex(ValueLayout.JAVA_FLOAT, mapA[k]);
                }
                var mask = SPECIES.indexInRange(0, validLanes);
                var vA = FloatVector.fromArray(SPECIES, bufA, 0, mask);
                                 
                var vRes = vA.div(vB);
                                 
                vRes.intoArray(bufRes, 0, mask);
                for(int k=0; k < validLanes; k++) {
                    resArray.getData().setAtIndex(ValueLayout.JAVA_FLOAT, mapRes[k], bufRes[k]);
                }
            }
        }
        return resArray;
    }


    public static NDArray divDouble(NDArray a, NDArray b, NDArray resArray) {
        if (a.isContiguous() && b.isContiguous() && resArray.isContiguous()) {
            long i = 0;
            long loopbound = a.getSize() - (a.getSize() % (DB_VL * 2));
                         
            for (; i < loopbound; i += DB_VL * 2) {
                var vA1 = DoubleVector.fromMemorySegment(SPECIESDB, a.getData(), i * DB_BYTES, ORDER);
                var vA2 = DoubleVector.fromMemorySegment(SPECIESDB, a.getData(), (i + DB_VL) * DB_BYTES, ORDER);
                var vB1 = DoubleVector.fromMemorySegment(SPECIESDB, b.getData(), i * DB_BYTES, ORDER);
                var vB2 = DoubleVector.fromMemorySegment(SPECIESDB, b.getData(), (i + DB_VL) * DB_BYTES, ORDER);
                                 
                var VRes1 = vA1.div(vB1);
                var VRes2 = vA2.div(vB2);
                                 
                VRes1.intoMemorySegment(resArray.getData(), i * DB_BYTES, ORDER);
                VRes2.intoMemorySegment(resArray.getData(), (i + DB_VL) * DB_BYTES, ORDER);
            }
            loopbound = SPECIESDB.loopBound(a.getSize());
            for (; i < loopbound; i += DB_VL) {
                var vA = DoubleVector.fromMemorySegment(SPECIESDB, a.getData(), i * DB_BYTES, ORDER);
                var vB = DoubleVector.fromMemorySegment(SPECIESDB, b.getData(), i * DB_BYTES, ORDER);
                var VRes = vA.div(vB);
                VRes.intoMemorySegment(resArray.getData(), i * DB_BYTES, ORDER);
            }
            for (; i < a.getSize(); i++) {
                double valA = a.getData().getAtIndex(ValueLayout.JAVA_DOUBLE, i);
                double valB = b.getData().getAtIndex(ValueLayout.JAVA_DOUBLE, i);
                resArray.getData().setAtIndex(ValueLayout.JAVA_DOUBLE, i, (double)(valA / valB));
            }
        } else {
            int vl = SPECIESDB.length();
            int[] mapA = new int[vl];
            int[] mapB = new int[vl];
            int[] mapRes = new int[vl];
                         
            double[] bufA = new double[vl];
            double[] bufB = new double[vl];
            double[] bufRes = new double[vl];
            var iterA = new NDIter(resArray.internalShapeUnsafe(), a.internalStridesUnsafe());
            var iterB = new NDIter(resArray.internalShapeUnsafe(), b.internalStridesUnsafe());
            var iterRes = new NDIter(resArray.internalShapeUnsafe(), resArray.internalStridesUnsafe());
            while (iterA.hasNext) {
                int validLanes = iterA.nextVector(mapA, vl);
                iterB.nextVector(mapB, vl);
                iterRes.nextVector(mapRes, vl);
                for(int k=0; k < validLanes; k++) {
                    bufA[k] = a.getData().getAtIndex(ValueLayout.JAVA_DOUBLE, mapA[k]);
                    bufB[k] = b.getData().getAtIndex(ValueLayout.JAVA_DOUBLE, mapB[k]);
                }
                var mask = SPECIESDB.indexInRange(0, validLanes);
                var vA = DoubleVector.fromArray(SPECIESDB, bufA, 0, mask);
                var vB = DoubleVector.fromArray(SPECIESDB, bufB, 0, mask);
                                 
                var vRes = vA.div(vB);
                                 
                vRes.intoArray(bufRes, 0, mask);
                for(int k=0; k < validLanes; k++) {
                    resArray.getData().setAtIndex(ValueLayout.JAVA_DOUBLE, mapRes[k], bufRes[k]);
                }
            }
        }
        return resArray;
    }


    public static NDArray divDouble(NDArray a, double b, NDArray resArray) {
        var vB = DoubleVector.broadcast(SPECIESDB, b);
                 
        if (a.isContiguous() && resArray.isContiguous()) {
            long i = 0;
            long loopbound = a.getSize() - (a.getSize() % (DB_VL * 2));
                         
            for (; i < loopbound; i += DB_VL * 2) {
                var vA1 = DoubleVector.fromMemorySegment(SPECIESDB, a.getData(), i * DB_BYTES, ORDER);
                var vA2 = DoubleVector.fromMemorySegment(SPECIESDB, a.getData(), (i + DB_VL) * DB_BYTES, ORDER);
                var VRes1 = vA1.div(vB);
                var VRes2 = vA2.div(vB);
                                 
                VRes1.intoMemorySegment(resArray.getData(), i * DB_BYTES, ORDER);
                VRes2.intoMemorySegment(resArray.getData(), (i + DB_VL) * DB_BYTES, ORDER);
            }
            loopbound = SPECIESDB.loopBound(a.getSize());
            for (; i < loopbound; i += DB_VL) {
                var vA = DoubleVector.fromMemorySegment(SPECIESDB, a.getData(), i * DB_BYTES, ORDER);
                var VRes = vA.div(vB);
                VRes.intoMemorySegment(resArray.getData(), i * DB_BYTES, ORDER);
            }
            for (; i < a.getSize(); i++) {
                double valA = a.getData().getAtIndex(ValueLayout.JAVA_DOUBLE, i);
                resArray.getData().setAtIndex(ValueLayout.JAVA_DOUBLE, i, (double)(valA / b));
            }
        } else {
            int vl = SPECIESDB.length();
            int[] mapA = new int[vl];
            int[] mapRes = new int[vl];
                         
            double[] bufA = new double[vl];
            double[] bufRes = new double[vl];
            var iterA = new NDIter(resArray.internalShapeUnsafe(), a.internalStridesUnsafe());
            var iterRes = new NDIter(resArray.internalShapeUnsafe(), resArray.internalStridesUnsafe());
            while (iterA.hasNext) {
                int validLanes = iterA.nextVector(mapA, vl);
                iterRes.nextVector(mapRes, vl);
                for(int k=0; k < validLanes; k++) {
                    bufA[k] = a.getData().getAtIndex(ValueLayout.JAVA_DOUBLE, mapA[k]);
                }
                var mask = SPECIESDB.indexInRange(0, validLanes);
                var vA = DoubleVector.fromArray(SPECIESDB, bufA, 0, mask);
                                 
                var vRes = vA.div(vB);
                                 
                vRes.intoArray(bufRes, 0, mask);
                for(int k=0; k < validLanes; k++) {
                    resArray.getData().setAtIndex(ValueLayout.JAVA_DOUBLE, mapRes[k], bufRes[k]);
                }
            }
        }
        return resArray;
    }


    public static NDArray divInt(NDArray a, NDArray b, NDArray resArray) {
        if (a.isContiguous() && b.isContiguous() && resArray.isContiguous()) {
            long i = 0;
            long loopbound = a.getSize() - (a.getSize() % (INT_VL * 2));
                         
            for (; i < loopbound; i += INT_VL * 2) {
                var vA1 = IntVector.fromMemorySegment(SPECIESINT, a.getData(), i * INT_BYTES, ORDER);
                var vA2 = IntVector.fromMemorySegment(SPECIESINT, a.getData(), (i + INT_VL) * INT_BYTES, ORDER);
                var vB1 = IntVector.fromMemorySegment(SPECIESINT, b.getData(), i * INT_BYTES, ORDER);
                var vB2 = IntVector.fromMemorySegment(SPECIESINT, b.getData(), (i + INT_VL) * INT_BYTES, ORDER);
                                 
                var VRes1 = vA1.div(vB1);
                var VRes2 = vA2.div(vB2);
                                 
                VRes1.intoMemorySegment(resArray.getData(), i * INT_BYTES, ORDER);
                VRes2.intoMemorySegment(resArray.getData(), (i + INT_VL) * INT_BYTES, ORDER);
            }
            loopbound = SPECIESINT.loopBound(a.getSize());
            for (; i < loopbound; i += INT_VL) {
                var vA = IntVector.fromMemorySegment(SPECIESINT, a.getData(), i * INT_BYTES, ORDER);
                var vB = IntVector.fromMemorySegment(SPECIESINT, b.getData(), i * INT_BYTES, ORDER);
                var VRes = vA.div(vB);
                VRes.intoMemorySegment(resArray.getData(), i * INT_BYTES, ORDER);
            }
            for (; i < a.getSize(); i++) {
                int valA = a.getData().getAtIndex(ValueLayout.JAVA_INT, i);
                int valB = b.getData().getAtIndex(ValueLayout.JAVA_INT, i);
                resArray.getData().setAtIndex(ValueLayout.JAVA_INT, i, (int)(valA / valB));
            }
        } else {
            int vl = SPECIESINT.length();
            int[] mapA = new int[vl];
            int[] mapB = new int[vl];
            int[] mapRes = new int[vl];
                         
            int[] bufA = new int[vl];
            int[] bufB = new int[vl];
            int[] bufRes = new int[vl];
            var iterA = new NDIter(resArray.internalShapeUnsafe(), a.internalStridesUnsafe());
            var iterB = new NDIter(resArray.internalShapeUnsafe(), b.internalStridesUnsafe());
            var iterRes = new NDIter(resArray.internalShapeUnsafe(), resArray.internalStridesUnsafe());
            while (iterA.hasNext) {
                int validLanes = iterA.nextVector(mapA, vl);
                iterB.nextVector(mapB, vl);
                iterRes.nextVector(mapRes, vl);
                for(int k=0; k < validLanes; k++) {
                    bufA[k] = a.getData().getAtIndex(ValueLayout.JAVA_INT, mapA[k]);
                    bufB[k] = b.getData().getAtIndex(ValueLayout.JAVA_INT, mapB[k]);
                }
                var mask = SPECIESINT.indexInRange(0, validLanes);
                var vA = IntVector.fromArray(SPECIESINT, bufA, 0, mask);
                var vB = IntVector.fromArray(SPECIESINT, bufB, 0, mask);
                                 
                var vRes = vA.div(vB);
                                 
                vRes.intoArray(bufRes, 0, mask);
                for(int k=0; k < validLanes; k++) {
                    resArray.getData().setAtIndex(ValueLayout.JAVA_INT, mapRes[k], bufRes[k]);
                }
            }
        }
        return resArray;
    }


    public static NDArray divInt(NDArray a, int b, NDArray resArray) {
        var vB = IntVector.broadcast(SPECIESINT, b);
                 
        if (a.isContiguous() && resArray.isContiguous()) {
            long i = 0;
            long loopbound = a.getSize() - (a.getSize() % (INT_VL * 2));
                         
            for (; i < loopbound; i += INT_VL * 2) {
                var vA1 = IntVector.fromMemorySegment(SPECIESINT, a.getData(), i * INT_BYTES, ORDER);
                var vA2 = IntVector.fromMemorySegment(SPECIESINT, a.getData(), (i + INT_VL) * INT_BYTES, ORDER);
                var VRes1 = vA1.div(vB);
                var VRes2 = vA2.div(vB);
                                 
                VRes1.intoMemorySegment(resArray.getData(), i * INT_BYTES, ORDER);
                VRes2.intoMemorySegment(resArray.getData(), (i + INT_VL) * INT_BYTES, ORDER);
            }
            loopbound = SPECIESINT.loopBound(a.getSize());
            for (; i < loopbound; i += INT_VL) {
                var vA = IntVector.fromMemorySegment(SPECIESINT, a.getData(), i * INT_BYTES, ORDER);
                var VRes = vA.div(vB);
                VRes.intoMemorySegment(resArray.getData(), i * INT_BYTES, ORDER);
            }
            for (; i < a.getSize(); i++) {
                int valA = a.getData().getAtIndex(ValueLayout.JAVA_INT, i);
                resArray.getData().setAtIndex(ValueLayout.JAVA_INT, i, (int)(valA / b));
            }
        } else {
            int vl = SPECIESINT.length();
            int[] mapA = new int[vl];
            int[] mapRes = new int[vl];
                         
            int[] bufA = new int[vl];
            int[] bufRes = new int[vl];
            var iterA = new NDIter(resArray.internalShapeUnsafe(), a.internalStridesUnsafe());
            var iterRes = new NDIter(resArray.internalShapeUnsafe(), resArray.internalStridesUnsafe());
            while (iterA.hasNext) {
                int validLanes = iterA.nextVector(mapA, vl);
                iterRes.nextVector(mapRes, vl);
                for(int k=0; k < validLanes; k++) {
                    bufA[k] = a.getData().getAtIndex(ValueLayout.JAVA_INT, mapA[k]);
                }
                var mask = SPECIESINT.indexInRange(0, validLanes);
                var vA = IntVector.fromArray(SPECIESINT, bufA, 0, mask);
                                 
                var vRes = vA.div(vB);
                                 
                vRes.intoArray(bufRes, 0, mask);
                for(int k=0; k < validLanes; k++) {
                    resArray.getData().setAtIndex(ValueLayout.JAVA_INT, mapRes[k], bufRes[k]);
                }
            }
        }
        return resArray;
    }


}