package jnum.internal.kernel.reduce;

import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;
import jdk.incubator.vector.VectorOperators;
import jnum.NDArray;
import jnum.internal.layout.NDIter;
import jnum.internal.layout.ShapeUtil;

public final class Min {
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

    private Min(){
        throw new AssertionError();
    }

    public static double minFloat(NDArray a){
        if(!a.isContiguous()){
            float finalMin = Float.POSITIVE_INFINITY;
            NDIter iter = new NDIter(a.internalShapeUnsafe());
            while(iter.hasNext){
                long byteOffset = ShapeUtil.getByteOffset(iter.coords, a.internalStridesUnsafe(), a.getDType());
                finalMin = Math.min(finalMin, a.getData().get(ValueLayout.JAVA_FLOAT, byteOffset));
                iter.next();
            }
            return (double) finalMin;
        }
        long i=0;
        long loopbound=SPECIES.loopBound(a.getSize());
        var vMin=FloatVector.broadcast(SPECIES,Float.POSITIVE_INFINITY);
        for(;i<loopbound;i+=VL){
            var v1=FloatVector.fromMemorySegment(SPECIES, a.getData(),i*FLOAT_BYTES,ORDER);
            vMin=vMin.min(v1);
        }
        float finalMin=vMin.reduceLanes(VectorOperators.MIN);
        for (; i < a.getSize(); i++) {
            float tailVal = a.getData().getAtIndex(ValueLayout.JAVA_FLOAT, i);
            if (tailVal < finalMin) {
                finalMin = tailVal;
            }
        }
        return (double) finalMin;
    }

    public static NDArray minFloatAxis(NDArray a,int axis,NDArray resArray){
        long size= a.internalShapeUnsafe()[axis];
        long strideA= a.internalStridesUnsafe()[axis];

        for(int i=0;i<resArray.getSize();i++){
            long tempIndex=i;
            long baseOffset=0;
            long offsetRes=0;
            for (int d = resArray.internalShapeUnsafe().length - 1; d >= 0; d--) {
                long coord = tempIndex % resArray.internalShapeUnsafe()[d];
                tempIndex /= resArray.internalShapeUnsafe()[d];
                offsetRes += coord * resArray.internalStridesUnsafe()[d];
                
                int aDim = (d >= axis) ? d + 1 : d;
                baseOffset += coord * a.internalStridesUnsafe()[aDim];
            }

            if(strideA==1){
                var vAcc = FloatVector.broadcast(SPECIES, Float.POSITIVE_INFINITY);
                long k = 0;
                long loopbound = SPECIES.loopBound(size);
                for(; k<loopbound ;k += VL){
                    var vVal = FloatVector.fromMemorySegment(SPECIES, a.getData(), (baseOffset + k) * 4L, ORDER);
                    vAcc = vAcc.min(vVal);
                }
                float acc = vAcc.reduceLanes(VectorOperators.MIN);
                for (; k < size; k++) {
                    acc = Math.min(acc, a.getData().getAtIndex(ValueLayout.JAVA_FLOAT, baseOffset + k));
                }
                resArray.getData().setAtIndex(ValueLayout.JAVA_FLOAT, offsetRes, acc);
            }else{
                float acc = Float.POSITIVE_INFINITY;
                for (long k = 0; k < size; k++) {
                    acc = Math.min(acc, a.getData().getAtIndex(ValueLayout.JAVA_FLOAT, baseOffset + k * strideA));
                }
                resArray.getData().setAtIndex(ValueLayout.JAVA_FLOAT, offsetRes, acc);
            }
        }
        return resArray;
    }

    public static double minInt(NDArray a) {
        if(!a.isContiguous()){
            int finalMin = Integer.MAX_VALUE;
            NDIter iter = new NDIter(a.internalShapeUnsafe());
            while(iter.hasNext){
                long byteOffset = ShapeUtil.getByteOffset(iter.coords, a.internalStridesUnsafe(), a.getDType());
                finalMin = Math.min(finalMin, a.getData().get(ValueLayout.JAVA_INT, byteOffset));
                iter.next();
            }
            return (double) finalMin;
        }
        long i = 0;
        long loopbound = SPECIESINT.loopBound(a.getSize());
        var vMin = IntVector.broadcast(SPECIESINT, Integer.MAX_VALUE);
        for (; i < loopbound; i += INT_VL) {
            var v1 = IntVector.fromMemorySegment(SPECIESINT, a.getData(), i * INT_BYTES, ORDER);
            vMin = vMin.min(v1);
        }
        int finalMin = vMin.reduceLanes(VectorOperators.MIN);
        for (; i < a.getSize(); i++) {
            int tailVal = a.getData().getAtIndex(ValueLayout.JAVA_INT, i);
            if (tailVal < finalMin) finalMin = tailVal;
        }
        return (double) finalMin;
    }

    public static NDArray minIntAxis(NDArray a,int axis,NDArray resArray){
        long size= a.internalShapeUnsafe()[axis];
        long strideA= a.internalStridesUnsafe()[axis];

        for(int i=0;i<resArray.getSize();i++){
            long tempIndex=i;
            long baseOffset=0;
            long offsetRes=0;
            for (int d = resArray.internalShapeUnsafe().length - 1; d >= 0; d--) {
                long coord = tempIndex % resArray.internalShapeUnsafe()[d];
                tempIndex /= resArray.internalShapeUnsafe()[d];
                offsetRes += coord * resArray.internalStridesUnsafe()[d];
                
                int aDim = (d >= axis) ? d + 1 : d;
                baseOffset += coord * a.internalStridesUnsafe()[aDim];
            }

            if(strideA==1){
                var vAcc = IntVector.broadcast(SPECIESINT, Integer.MAX_VALUE);
                long k = 0;
                long loopbound = SPECIESINT.loopBound(size);
                for(; k<loopbound ;k += INT_VL){
                    var vVal = IntVector.fromMemorySegment(SPECIESINT, a.getData(), (baseOffset + k) * 4L, ORDER);
                    vAcc = vAcc.min(vVal);
                }
                int acc = vAcc.reduceLanes(VectorOperators.MIN);
                for (; k < size; k++) {
                    acc = Math.min(acc, a.getData().getAtIndex(ValueLayout.JAVA_INT, baseOffset + k));
                }
                resArray.getData().setAtIndex(ValueLayout.JAVA_INT, offsetRes, acc);
            }else{
                int acc = Integer.MAX_VALUE;
                for (long k = 0; k < size; k++) {
                    acc = Math.min(acc, a.getData().getAtIndex(ValueLayout.JAVA_INT, baseOffset + k * strideA));
                }
                resArray.getData().setAtIndex(ValueLayout.JAVA_INT, offsetRes, acc);
            }
        }
        return resArray;
    }

    public static double minDouble(NDArray a) {
        if(!a.isContiguous()){
            double finalMin = Double.POSITIVE_INFINITY;
            NDIter iter = new NDIter(a.internalShapeUnsafe());
            while(iter.hasNext){
                long byteOffset = ShapeUtil.getByteOffset(iter.coords, a.internalStridesUnsafe(), a.getDType());
                finalMin = Math.min(finalMin, a.getData().get(ValueLayout.JAVA_DOUBLE, byteOffset));
                iter.next();
            }
            return finalMin;
        }
        long i = 0;
        long loopbound = SPECIESDB.loopBound(a.getSize());
        var vMin = DoubleVector.broadcast(SPECIESDB, Double.POSITIVE_INFINITY);
        for (; i < loopbound; i += DB_VL) {
            var v1 = DoubleVector.fromMemorySegment(SPECIESDB, a.getData(), i * DB_BYTES, ORDER);
            vMin = vMin.min(v1);
        }
        double finalMin = vMin.reduceLanes(VectorOperators.MIN);
        for (; i < a.getSize(); i++) {
            double tailVal = a.getData().getAtIndex(ValueLayout.JAVA_DOUBLE, i);
            if (tailVal < finalMin) finalMin = tailVal;
        }
        return (double) finalMin;
    }

    public static NDArray minDoubleAxis(NDArray a,int axis,NDArray resArray){
        long size= a.internalShapeUnsafe()[axis];
        long strideA= a.internalStridesUnsafe()[axis];

        for(int i=0;i<resArray.getSize();i++){
            long tempIndex=i;
            long baseOffset=0;
            long offsetRes=0;
            for (int d = resArray.internalShapeUnsafe().length - 1; d >= 0; d--) {
                long coord = tempIndex % resArray.internalShapeUnsafe()[d];
                tempIndex /= resArray.internalShapeUnsafe()[d];
                offsetRes += coord * resArray.internalStridesUnsafe()[d];
                
                int aDim = (d >= axis) ? d + 1 : d;
                baseOffset += coord * a.internalStridesUnsafe()[aDim];
            }

            if(strideA==1){
                var vAcc = DoubleVector.broadcast(SPECIESDB, Double.POSITIVE_INFINITY);
                long k = 0;
                long loopbound = SPECIESDB.loopBound(size);
                for(; k<loopbound ;k += DB_VL){
                    var vVal = DoubleVector.fromMemorySegment(SPECIESDB, a.getData(), (baseOffset + k) * 8L, ORDER);
                    vAcc = vAcc.min(vVal);
                }
                double acc = vAcc.reduceLanes(VectorOperators.MIN);
                for (; k < size; k++) {
                    acc = Math.min(acc, a.getData().getAtIndex(ValueLayout.JAVA_DOUBLE, baseOffset + k));
                }
                resArray.getData().setAtIndex(ValueLayout.JAVA_DOUBLE, offsetRes, acc);
            }else{
                double acc = Double.POSITIVE_INFINITY;
                for (long k = 0; k < size; k++) {
                    acc = Math.min(acc, a.getData().getAtIndex(ValueLayout.JAVA_DOUBLE, baseOffset + k * strideA));
                }
                resArray.getData().setAtIndex(ValueLayout.JAVA_DOUBLE, offsetRes, acc);
            }
        }
        return resArray;
    }
}
