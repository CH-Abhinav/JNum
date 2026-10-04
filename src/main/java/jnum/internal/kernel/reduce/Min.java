package jnum.internal.kernel.reduce;

import static jnum.internal.Constants.*;

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
        long loopbound=SPECIES_F32.loopBound(a.getSize());
        var vMin=FloatVector.broadcast(SPECIES_F32,Float.POSITIVE_INFINITY);
        for(;i<loopbound;i+=VL_F32){
            var v1=FloatVector.fromMemorySegment(SPECIES_F32, a.getData(),i*BYTES_F32,NATIVE_ORDER);
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
                var vAcc = FloatVector.broadcast(SPECIES_F32, Float.POSITIVE_INFINITY);
                long k = 0;
                long loopbound = SPECIES_F32.loopBound(size);
                for(; k<loopbound ;k += VL_F32){
                    var vVal = FloatVector.fromMemorySegment(SPECIES_F32, a.getData(), (baseOffset + k) * 4L, NATIVE_ORDER);
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
        long loopbound = SPECIES_I32.loopBound(a.getSize());
        var vMin = IntVector.broadcast(SPECIES_I32, Integer.MAX_VALUE);
        for (; i < loopbound; i += VL_I32) {
            var v1 = IntVector.fromMemorySegment(SPECIES_I32, a.getData(), i * BYTES_I32, NATIVE_ORDER);
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
                var vAcc = IntVector.broadcast(SPECIES_I32, Integer.MAX_VALUE);
                long k = 0;
                long loopbound = SPECIES_I32.loopBound(size);
                for(; k<loopbound ;k += VL_I32){
                    var vVal = IntVector.fromMemorySegment(SPECIES_I32, a.getData(), (baseOffset + k) * 4L, NATIVE_ORDER);
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
        long loopbound = SPECIES_F64.loopBound(a.getSize());
        var vMin = DoubleVector.broadcast(SPECIES_F64, Double.POSITIVE_INFINITY);
        for (; i < loopbound; i += VL_F64) {
            var v1 = DoubleVector.fromMemorySegment(SPECIES_F64, a.getData(), i * BYTES_F64, NATIVE_ORDER);
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
                var vAcc = DoubleVector.broadcast(SPECIES_F64, Double.POSITIVE_INFINITY);
                long k = 0;
                long loopbound = SPECIES_F64.loopBound(size);
                for(; k<loopbound ;k += VL_F64){
                    var vVal = DoubleVector.fromMemorySegment(SPECIES_F64, a.getData(), (baseOffset + k) * 8L, NATIVE_ORDER);
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
