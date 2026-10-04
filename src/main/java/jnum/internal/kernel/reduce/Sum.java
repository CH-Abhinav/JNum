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

public final class Sum {

    private Sum(){
        throw new AssertionError();
    }

    public static double sumFloat(NDArray a){
        if(!a.isContiguous()){
            double total = 0.0;
            NDIter iter = new NDIter(a.internalShapeUnsafe());
            while(iter.hasNext){
                long byteOffset = ShapeUtil.getByteOffset(iter.coords, a.internalStridesUnsafe(), a.getDType());
                total += a.getData().get(ValueLayout.JAVA_FLOAT, byteOffset);
                iter.next();
            }
            return total;
        }
        long i=0;
        long loopbound=SPECIES_F32.loopBound(a.getSize());
        var vSum=FloatVector.zero(SPECIES_F32);
        for(;i<loopbound;i+=VL_F32){
            var v1=FloatVector.fromMemorySegment(SPECIES_F32, a.getData(),i*BYTES_F32,NATIVE_ORDER);
            vSum=vSum.add(v1);
        }
        float total=vSum.reduceLanes(VectorOperators.ADD);
        for(; i< a.getSize(); i++){
            total+= a.getData().getAtIndex(ValueLayout.JAVA_FLOAT,i);
        }
        return (double) total;
    }

    public static double sumInt(NDArray a){
        if(!a.isContiguous()){
            double total = 0.0;
            NDIter iter = new NDIter(a.internalShapeUnsafe());
            while(iter.hasNext){
                long byteOffset = ShapeUtil.getByteOffset(iter.coords, a.internalStridesUnsafe(), a.getDType());
                total += a.getData().get(ValueLayout.JAVA_INT, byteOffset);
                iter.next();
            }
            return total;
        }
        long i=0;
        long loopbound=SPECIES_I32.loopBound(a.getSize());
        var vSum=IntVector.zero(SPECIES_I32);
        for(;i<loopbound;i+=VL_I32){
            var v1=IntVector.fromMemorySegment(SPECIES_I32, a.getData(),i*BYTES_I32,NATIVE_ORDER);
            vSum=vSum.add(v1);
        }
        int total=vSum.reduceLanes(VectorOperators.ADD);
        for(; i< a.getSize(); i++){
            total+= a.getData().getAtIndex(ValueLayout.JAVA_INT,i);
        }
        return (double) total;
    }

    public static double sumDouble(NDArray a){
        if(!a.isContiguous()){
            double total = 0.0;
            NDIter iter = new NDIter(a.internalShapeUnsafe());
            while(iter.hasNext){
                long byteOffset = ShapeUtil.getByteOffset(iter.coords, a.internalStridesUnsafe(), a.getDType());
                total += a.getData().get(ValueLayout.JAVA_DOUBLE, byteOffset);
                iter.next();
            }
            return total;
        }
        long i=0;
        long loopbound=SPECIES_F64.loopBound(a.getSize());
        var vSum=DoubleVector.zero(SPECIES_F64);
        for(;i<loopbound;i+=VL_F64){
            var v1=DoubleVector.fromMemorySegment(SPECIES_F64, a.getData(),i*BYTES_F64,NATIVE_ORDER);
            vSum=vSum.add(v1);
        }
        double total=vSum.reduceLanes(VectorOperators.ADD);
        for(; i< a.getSize(); i++){
            total+= a.getData().getAtIndex(ValueLayout.JAVA_DOUBLE,i);
        }
        return (double) total;
    }

    public static NDArray sumFloatAxis(NDArray a,int axis,NDArray resArray){
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
                var vAcc = FloatVector.zero(SPECIES_F32);
                long k = 0;
                long loopbound = SPECIES_F32.loopBound(size);
                for(; k<loopbound ;k += VL_F32){
                    var vVal = FloatVector.fromMemorySegment(SPECIES_F32, a.getData(), (baseOffset + k) * 4L, NATIVE_ORDER);
                    vAcc = vAcc.add(vVal);
                }
                float acc = vAcc.reduceLanes(VectorOperators.ADD);
                for (; k < size; k++) {
                    acc += a.getData().getAtIndex(ValueLayout.JAVA_FLOAT, baseOffset + k);
                }
                resArray.getData().setAtIndex(ValueLayout.JAVA_FLOAT, offsetRes, acc);
            }else{
                float acc = 0f;
                for (long k = 0; k < size; k++) {
                    acc += a.getData().getAtIndex(ValueLayout.JAVA_FLOAT, baseOffset + k * strideA);
                }
                resArray.getData().setAtIndex(ValueLayout.JAVA_FLOAT, offsetRes, acc);
            }
        }
        return resArray;
    }

    public static NDArray sumIntAxis(NDArray a,int axis,NDArray resArray){
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
                var vAcc = IntVector.zero(SPECIES_I32);
                long k = 0;
                long loopbound = SPECIES_I32.loopBound(size);
                for(; k<loopbound ;k += VL_I32){
                    var vVal = IntVector.fromMemorySegment(SPECIES_I32, a.getData(), (baseOffset + k) * 4L, NATIVE_ORDER);
                    vAcc = vAcc.add(vVal);
                }
                int acc = vAcc.reduceLanes(VectorOperators.ADD);
                for (; k < size; k++) {
                    acc += a.getData().getAtIndex(ValueLayout.JAVA_INT, baseOffset + k);
                }
                resArray.getData().setAtIndex(ValueLayout.JAVA_INT, offsetRes, acc);
            }else{
                int acc = 0;
                for (long k = 0; k < size; k++) {
                    acc += a.getData().getAtIndex(ValueLayout.JAVA_INT, baseOffset + k * strideA);
                }
                resArray.getData().setAtIndex(ValueLayout.JAVA_INT, offsetRes, acc);
            }
        }
        return resArray;
    }

    public static NDArray sumDoubleAxis(NDArray a,int axis,NDArray resArray){
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
                var vAcc = DoubleVector.zero(SPECIES_F64);
                long k = 0;
                long loopbound = SPECIES_F64.loopBound(size);
                for(; k<loopbound ;k += VL_F64){
                    var vVal = DoubleVector.fromMemorySegment(SPECIES_F64, a.getData(), (baseOffset + k) * 8L, NATIVE_ORDER);
                    vAcc = vAcc.add(vVal);
                }
                double acc = vAcc.reduceLanes(VectorOperators.ADD);
                for (; k < size; k++) {
                    acc += a.getData().getAtIndex(ValueLayout.JAVA_DOUBLE, baseOffset + k);
                }
                resArray.getData().setAtIndex(ValueLayout.JAVA_DOUBLE, offsetRes, acc);
            }else{
                double acc = 0f;
                for (long k = 0; k < size; k++) {
                    acc += a.getData().getAtIndex(ValueLayout.JAVA_DOUBLE, baseOffset + k * strideA);
                }
                resArray.getData().setAtIndex(ValueLayout.JAVA_DOUBLE, offsetRes, acc);
            }
        }
        return resArray;
    }
}
