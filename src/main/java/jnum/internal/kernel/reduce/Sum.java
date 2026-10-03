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

public final class Sum {
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
        long loopbound=SPECIES.loopBound(a.getSize());
        var vSum=FloatVector.zero(SPECIES);
        for(;i<loopbound;i+=VL){
            var v1=FloatVector.fromMemorySegment(SPECIES, a.getData(),i*FLOAT_BYTES,ORDER);
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
        long loopbound=SPECIESINT.loopBound(a.getSize());
        var vSum=IntVector.zero(SPECIESINT);
        for(;i<loopbound;i+=INT_VL){
            var v1=IntVector.fromMemorySegment(SPECIESINT, a.getData(),i*INT_BYTES,ORDER);
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
        long loopbound=SPECIESDB.loopBound(a.getSize());
        var vSum=DoubleVector.zero(SPECIESDB);
        for(;i<loopbound;i+=DB_VL){
            var v1=DoubleVector.fromMemorySegment(SPECIESDB, a.getData(),i*DB_BYTES,ORDER);
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
                var vAcc = FloatVector.zero(SPECIES);
                long k = 0;
                long loopbound = SPECIES.loopBound(size);
                for(; k<loopbound ;k += VL){
                    var vVal = FloatVector.fromMemorySegment(SPECIES, a.getData(), (baseOffset + k) * 4L, ORDER);
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
                var vAcc = IntVector.zero(SPECIESINT);
                long k = 0;
                long loopbound = SPECIESINT.loopBound(size);
                for(; k<loopbound ;k += INT_VL){
                    var vVal = IntVector.fromMemorySegment(SPECIESINT, a.getData(), (baseOffset + k) * 4L, ORDER);
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
                var vAcc = DoubleVector.zero(SPECIESDB);
                long k = 0;
                long loopbound = SPECIESDB.loopBound(size);
                for(; k<loopbound ;k += DB_VL){
                    var vVal = DoubleVector.fromMemorySegment(SPECIESDB, a.getData(), (baseOffset + k) * 8L, ORDER);
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
