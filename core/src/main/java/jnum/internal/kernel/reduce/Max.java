package jnum.internal.kernel.reduce;

import static jnum.internal.Constants.*;

import java.lang.foreign.ValueLayout;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorOperators;
import jnum.NDArray;
import jnum.internal.layout.NDIter;
import jnum.internal.layout.ShapeUtil;

public final class Max {

    private Max(){
        throw new AssertionError();
    }

    public static double maxFloat(NDArray a){
        if(!a.isContiguous()){
            float finalMax = Float.NEGATIVE_INFINITY;
            NDIter iter = new NDIter(a.internalShapeUnsafe());
            while(iter.hasNext){
                long byteOffset = ShapeUtil.getByteOffset(iter.coords, a.internalStridesUnsafe(), a.getDType());
                finalMax = Math.max(finalMax, a.getData().get(ValueLayout.JAVA_FLOAT, byteOffset));
                iter.next();
            }
            return (double) finalMax;
        }
        long i=0;
        long loopbound=SPECIES_F32.loopBound(a.getSize());
        var vMax = FloatVector.broadcast(SPECIES_F32, Float.NEGATIVE_INFINITY);
        for(;i<loopbound;i+=VL_F32){
            var v1=FloatVector.fromMemorySegment(SPECIES_F32, a.getData(),i*BYTES_F32,NATIVE_ORDER);
            vMax=vMax.max(v1);
        }
        float finalMax=vMax.reduceLanes(VectorOperators.MAX);
        for (; i < a.getSize(); i++) {
            float tailVal = a.getData().getAtIndex(ValueLayout.JAVA_FLOAT, i);
            if (tailVal > finalMax) {
                finalMax = tailVal;
            }
        }
        return (double) finalMax;
    }

    public static NDArray maxFloatAxis(NDArray a,int axis,NDArray resArray){
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
                var vAcc = FloatVector.broadcast(SPECIES_F32, Float.NEGATIVE_INFINITY);
                long k = 0;
                long loopbound = SPECIES_F32.loopBound(size);
                for(; k<loopbound ;k += VL_F32){
                    var vVal = FloatVector.fromMemorySegment(SPECIES_F32, a.getData(), (baseOffset + k) * 4L, NATIVE_ORDER);
                    vAcc = vAcc.max(vVal);
                }
                float acc = vAcc.reduceLanes(VectorOperators.MAX);
                for (; k < size; k++) {
                    acc = Math.max(acc, a.getData().getAtIndex(ValueLayout.JAVA_FLOAT, baseOffset + k));
                }
                resArray.getData().setAtIndex(ValueLayout.JAVA_FLOAT, offsetRes, acc);
            }else{
                float acc = Float.NEGATIVE_INFINITY;
                for (long k = 0; k < size; k++) {
                    acc = Math.max(acc, a.getData().getAtIndex(ValueLayout.JAVA_FLOAT, baseOffset + k * strideA));
                }
                resArray.getData().setAtIndex(ValueLayout.JAVA_FLOAT, offsetRes, acc);
            }
        }
        return resArray;
    }

    public static double maxInt(NDArray a) {
        if(!a.isContiguous()){
            int finalMax = Integer.MIN_VALUE;
            NDIter iter = new NDIter(a.internalShapeUnsafe());
            while(iter.hasNext){
                long byteOffset = ShapeUtil.getByteOffset(iter.coords, a.internalStridesUnsafe(), a.getDType());
                finalMax = Math.max(finalMax, a.getData().get(ValueLayout.JAVA_INT, byteOffset));
                iter.next();
            }
            return (double) finalMax;
        }
        long i = 0;
        long loopbound = SPECIES_I32.loopBound(a.getSize());
        var vMax = IntVector.broadcast(SPECIES_I32, Integer.MIN_VALUE);
        for (; i < loopbound; i += VL_I32) {
            var v1 = IntVector.fromMemorySegment(SPECIES_I32, a.getData(), i * BYTES_I32, NATIVE_ORDER);
            vMax = vMax.max(v1);
        }
        int finalMax = vMax.reduceLanes(VectorOperators.MAX);
        for (; i < a.getSize(); i++) {
            int tailVal = a.getData().getAtIndex(ValueLayout.JAVA_INT, i);
            if (tailVal > finalMax) finalMax = tailVal;
        }
        return (double) finalMax;
    }

    public static NDArray maxIntAxis(NDArray a,int axis,NDArray resArray){
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
                var vAcc = IntVector.broadcast(SPECIES_I32, Integer.MIN_VALUE);
                long k = 0;
                long loopbound = SPECIES_I32.loopBound(size);
                for(; k<loopbound ;k += VL_I32){
                    var vVal = IntVector.fromMemorySegment(SPECIES_I32, a.getData(), (baseOffset + k) * 4L, NATIVE_ORDER);
                    vAcc = vAcc.max(vVal);
                }
                int acc = vAcc.reduceLanes(VectorOperators.MAX);
                for (; k < size; k++) {
                    acc = Math.max(acc, a.getData().getAtIndex(ValueLayout.JAVA_INT, baseOffset + k));
                }
                resArray.getData().setAtIndex(ValueLayout.JAVA_INT, offsetRes, acc);
            }else{
                int acc = Integer.MIN_VALUE;
                for (long k = 0; k < size; k++) {
                    acc = Math.max(acc, a.getData().getAtIndex(ValueLayout.JAVA_INT, baseOffset + k * strideA));
                }
                resArray.getData().setAtIndex(ValueLayout.JAVA_INT, offsetRes, acc);
            }
        }
        return resArray;
    }

    public static double maxDouble(NDArray a) {
        if(!a.isContiguous()){
            double finalMax = Double.NEGATIVE_INFINITY;
            NDIter iter = new NDIter(a.internalShapeUnsafe());
            while(iter.hasNext){
                long byteOffset = ShapeUtil.getByteOffset(iter.coords, a.internalStridesUnsafe(), a.getDType());
                finalMax = Math.max(finalMax, a.getData().get(ValueLayout.JAVA_DOUBLE, byteOffset));
                iter.next();
            }
            return finalMax;
        }
        long i = 0;
        long loopbound = SPECIES_F64.loopBound(a.getSize());
        var vMax = DoubleVector.broadcast(SPECIES_F64, Double.NEGATIVE_INFINITY);
        for (; i < loopbound; i += VL_F64) {
            var v1 = DoubleVector.fromMemorySegment(SPECIES_F64, a.getData(), i * BYTES_F64, NATIVE_ORDER);
            vMax = vMax.max(v1);
        }
        double finalMax = vMax.reduceLanes(VectorOperators.MAX);
        for (; i < a.getSize(); i++) {
            double tailVal = a.getData().getAtIndex(ValueLayout.JAVA_DOUBLE, i);
            if (tailVal > finalMax) finalMax = tailVal;
        }
        return (double) finalMax;
    }

    public static NDArray maxDoubleAxis(NDArray a,int axis,NDArray resArray){
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
                var vAcc = DoubleVector.broadcast(SPECIES_F64, Double.NEGATIVE_INFINITY);
                long k = 0;
                long loopbound = SPECIES_F64.loopBound(size);
                for(; k<loopbound ;k += VL_F64){
                    var vVal = DoubleVector.fromMemorySegment(SPECIES_F64, a.getData(), (baseOffset + k) * 8L, NATIVE_ORDER);
                    vAcc = vAcc.max(vVal);
                }
                double acc = vAcc.reduceLanes(VectorOperators.MAX);
                for (; k < size; k++) {
                    acc = Math.max(acc, a.getData().getAtIndex(ValueLayout.JAVA_DOUBLE, baseOffset + k));
                }
                resArray.getData().setAtIndex(ValueLayout.JAVA_DOUBLE, offsetRes, acc);
            }else{
                double acc = Double.NEGATIVE_INFINITY;
                for (long k = 0; k < size; k++) {
                    acc = Math.max(acc, a.getData().getAtIndex(ValueLayout.JAVA_DOUBLE, baseOffset + k * strideA));
                }
                resArray.getData().setAtIndex(ValueLayout.JAVA_DOUBLE, offsetRes, acc);
            }
        }
        return resArray;
    }
}
