package jnum.jnumops;

import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;
import jdk.incubator.vector.VectorOperators;
import jnum.NDArray;
import jnum.jnumutils.ShapeUtil;

public class ReduceOps {
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

    //non instatitable utility class
    private ReduceOps(){
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
        int size= a.internalShapeUnsafe()[axis];
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
                int k = 0;
                int loopbound = SPECIES.loopBound(size);
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
                for (int k = 0; k < size; k++) {
                    acc += a.getData().getAtIndex(ValueLayout.JAVA_FLOAT, baseOffset + k * strideA);
                }
                resArray.getData().setAtIndex(ValueLayout.JAVA_FLOAT, offsetRes, acc);
            }
        }
        return resArray;
    }

    public static NDArray sumIntAxis(NDArray a,int axis,NDArray resArray){
        int size= a.internalShapeUnsafe()[axis];
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
                int k = 0;
                int loopbound = SPECIESINT.loopBound(size);
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
                for (int k = 0; k < size; k++) {
                    acc += a.getData().getAtIndex(ValueLayout.JAVA_INT, baseOffset + k * strideA);
                }
                resArray.getData().setAtIndex(ValueLayout.JAVA_INT, offsetRes, acc);
            }
        }
        return resArray;
    }

    public static NDArray sumDoubleAxis(NDArray a,int axis,NDArray resArray){
        int size= a.internalShapeUnsafe()[axis];
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
                int k = 0;
                int loopbound = SPECIESDB.loopBound(size);
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
                for (int k = 0; k < size; k++) {
                    acc += a.getData().getAtIndex(ValueLayout.JAVA_DOUBLE, baseOffset + k * strideA);
                }
                resArray.getData().setAtIndex(ValueLayout.JAVA_DOUBLE, offsetRes, acc);
            }
        }
        return resArray;
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
        long loopbound=SPECIES.loopBound(a.getSize());
        var vMax = FloatVector.broadcast(SPECIES, Float.NEGATIVE_INFINITY);
        for(;i<loopbound;i+=VL){
            var v1=FloatVector.fromMemorySegment(SPECIES, a.getData(),i*FLOAT_BYTES,ORDER);
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
        int size= a.internalShapeUnsafe()[axis];
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
                var vAcc = FloatVector.broadcast(SPECIES, Float.NEGATIVE_INFINITY);
                int k = 0;
                int loopbound = SPECIES.loopBound(size);
                for(; k<loopbound ;k += VL){
                    var vVal = FloatVector.fromMemorySegment(SPECIES, a.getData(), (baseOffset + k) * 4L, ORDER);
                    vAcc = vAcc.max(vVal);
                }
                float acc = vAcc.reduceLanes(VectorOperators.MAX);
                for (; k < size; k++) {
                    acc = Math.max(acc, a.getData().getAtIndex(ValueLayout.JAVA_FLOAT, baseOffset + k));
                }
                resArray.getData().setAtIndex(ValueLayout.JAVA_FLOAT, offsetRes, acc);
            }else{
                float acc = Float.NEGATIVE_INFINITY;
                for (int k = 0; k < size; k++) {
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
        long loopbound = SPECIESINT.loopBound(a.getSize());
        var vMax = IntVector.broadcast(SPECIESINT, Integer.MIN_VALUE);
        for (; i < loopbound; i += INT_VL) {
            var v1 = IntVector.fromMemorySegment(SPECIESINT, a.getData(), i * INT_BYTES, ORDER);
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
        int size= a.internalShapeUnsafe()[axis];
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
                var vAcc = IntVector.broadcast(SPECIESINT, Integer.MIN_VALUE);
                int k = 0;
                int loopbound = SPECIESINT.loopBound(size);
                for(; k<loopbound ;k += INT_VL){
                    var vVal = IntVector.fromMemorySegment(SPECIESINT, a.getData(), (baseOffset + k) * 4L, ORDER);
                    vAcc = vAcc.max(vVal);
                }
                int acc = vAcc.reduceLanes(VectorOperators.MAX);
                for (; k < size; k++) {
                    acc = Math.max(acc, a.getData().getAtIndex(ValueLayout.JAVA_INT, baseOffset + k));
                }
                resArray.getData().setAtIndex(ValueLayout.JAVA_INT, offsetRes, acc);
            }else{
                int acc = Integer.MIN_VALUE;
                for (int k = 0; k < size; k++) {
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
        long loopbound = SPECIESDB.loopBound(a.getSize());
        var vMax = DoubleVector.broadcast(SPECIESDB, Double.NEGATIVE_INFINITY);
        for (; i < loopbound; i += DB_VL) {
            var v1 = DoubleVector.fromMemorySegment(SPECIESDB, a.getData(), i * DB_BYTES, ORDER);
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
        int size= a.internalShapeUnsafe()[axis];
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
                var vAcc = DoubleVector.broadcast(SPECIESDB, Double.NEGATIVE_INFINITY);
                int k = 0;
                int loopbound = SPECIESDB.loopBound(size);
                for(; k<loopbound ;k += DB_VL){
                    var vVal = DoubleVector.fromMemorySegment(SPECIESDB, a.getData(), (baseOffset + k) * 8L, ORDER);
                    vAcc = vAcc.max(vVal);
                }
                double acc = vAcc.reduceLanes(VectorOperators.MAX);
                for (; k < size; k++) {
                    acc = Math.max(acc, a.getData().getAtIndex(ValueLayout.JAVA_DOUBLE, baseOffset + k));
                }
                resArray.getData().setAtIndex(ValueLayout.JAVA_DOUBLE, offsetRes, acc);
            }else{
                double acc = Double.NEGATIVE_INFINITY;
                for (int k = 0; k < size; k++) {
                    acc = Math.max(acc, a.getData().getAtIndex(ValueLayout.JAVA_DOUBLE, baseOffset + k * strideA));
                }
                resArray.getData().setAtIndex(ValueLayout.JAVA_DOUBLE, offsetRes, acc);
            }
        }
        return resArray;
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
        int size= a.internalShapeUnsafe()[axis];
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
                int k = 0;
                int loopbound = SPECIES.loopBound(size);
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
                for (int k = 0; k < size; k++) {
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
        int size= a.internalShapeUnsafe()[axis];
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
                int k = 0;
                int loopbound = SPECIESINT.loopBound(size);
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
                for (int k = 0; k < size; k++) {
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
        int size= a.internalShapeUnsafe()[axis];
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
                int k = 0;
                int loopbound = SPECIESDB.loopBound(size);
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
                for (int k = 0; k < size; k++) {
                    acc = Math.min(acc, a.getData().getAtIndex(ValueLayout.JAVA_DOUBLE, baseOffset + k * strideA));
                }
                resArray.getData().setAtIndex(ValueLayout.JAVA_DOUBLE, offsetRes, acc);
            }
        }
        return resArray;
    }

    public static double dotFloat(NDArray a,NDArray b){
        if(!a.isContiguous() || !b.isContiguous()){
            double total = 0.0;
            NDIter iterA = new NDIter(a.internalShapeUnsafe());
            NDIter iterB = new NDIter(b.internalShapeUnsafe());
            while(iterA.hasNext){
                long byteOffsetA = ShapeUtil.getByteOffset(iterA.coords, a.internalStridesUnsafe(), a.getDType());
                long byteOffsetB = ShapeUtil.getByteOffset(iterB.coords, b.internalStridesUnsafe(), b.getDType());
                total += a.getData().get(ValueLayout.JAVA_FLOAT, byteOffsetA) * b.getData().get(ValueLayout.JAVA_FLOAT, byteOffsetB);
                iterA.next();
                iterB.next();
            }
            return total;
        }
        long i=0;
        long loopbound= a.getSize() - (a.getSize() % (VL * 2));
        var vSum1 = FloatVector.zero(SPECIES);
        var vSum2 = FloatVector.zero(SPECIES);

        for(;i<loopbound;i+=VL*2){
            var va1=FloatVector.fromMemorySegment(SPECIES, a.getData(), i*FLOAT_BYTES, ORDER);
            var vb1=FloatVector.fromMemorySegment(SPECIES, b.getData(), i*FLOAT_BYTES, ORDER);
            var va2=FloatVector.fromMemorySegment(SPECIES, a.getData(), (i+VL)*FLOAT_BYTES, ORDER);
            var vb2=FloatVector.fromMemorySegment(SPECIES, b.getData(), (i+VL)*FLOAT_BYTES, ORDER);
            vSum1=va1.fma(vb1,vSum1);
            vSum2=va2.fma(vb2, vSum2);
        }

        loopbound=SPECIES.loopBound(a.getSize());

        for(;i<loopbound;i+=VL){
            var v1=FloatVector.fromMemorySegment(SPECIES, a.getData(), i*FLOAT_BYTES, ORDER);
            var v2=FloatVector.fromMemorySegment(SPECIES, b.getData(), i*FLOAT_BYTES, ORDER);
            vSum1=v1.fma(v2,vSum1);
        }

        double total = vSum1.reduceLanes(VectorOperators.ADD);
        total+=vSum2.reduceLanes(VectorOperators.ADD);
        for(; i< a.getSize(); i++){
            float valA = a.getData().getAtIndex(ValueLayout.JAVA_FLOAT, i);
            float valB = b.getData().getAtIndex(ValueLayout.JAVA_FLOAT, i);
            total += (valA * valB);
        }
        return total;
    }

    public static double dotInt(NDArray a,NDArray b){
        if(!a.isContiguous() || !b.isContiguous()){
            double total = 0.0;
            NDIter iterA = new NDIter(a.internalShapeUnsafe());
            NDIter iterB = new NDIter(b.internalShapeUnsafe());
            while(iterA.hasNext){
                long byteOffsetA = ShapeUtil.getByteOffset(iterA.coords, a.internalStridesUnsafe(), a.getDType());
                long byteOffsetB = ShapeUtil.getByteOffset(iterB.coords, b.internalStridesUnsafe(), b.getDType());
                total += (double) a.getData().get(ValueLayout.JAVA_INT, byteOffsetA) * (double) b.getData().get(ValueLayout.JAVA_INT, byteOffsetB);
                iterA.next();
                iterB.next();
            }
            return total;
        }
        long i=0;
        long loopbound= a.getSize() - (a.getSize() % (INT_VL * 2));
        var vSum1 = IntVector.zero(SPECIESINT);
        var vSum2 = IntVector.zero(SPECIESINT);

        for(;i<loopbound;i+=INT_VL*2){
            var va1=IntVector.fromMemorySegment(SPECIESINT, a.getData(), i*INT_BYTES, ORDER);
            var vb1=IntVector.fromMemorySegment(SPECIESINT, b.getData(), i*INT_BYTES, ORDER);
            var va2=IntVector.fromMemorySegment(SPECIESINT, a.getData(), (i+INT_VL)*INT_BYTES, ORDER);
            var vb2=IntVector.fromMemorySegment(SPECIESINT, b.getData(), (i+INT_VL)*INT_BYTES, ORDER);
            vSum1 = va1.mul(vb1).add(vSum1);
            vSum2 = va2.mul(vb2).add(vSum2);
        }

        loopbound=SPECIES.loopBound(a.getSize());

        for(;i<loopbound;i+=INT_VL){
            var v1=IntVector.fromMemorySegment(SPECIESINT, a.getData(), i*INT_BYTES, ORDER);
            var v2=IntVector.fromMemorySegment(SPECIESINT, b.getData(), i*INT_BYTES, ORDER);
            vSum1=v1.mul(v2).add(vSum1);
        }

        double total = vSum1.reduceLanes(VectorOperators.ADD);
        total+=vSum2.reduceLanes(VectorOperators.ADD);
        for(; i< a.getSize(); i++){
            int valA = a.getData().getAtIndex(ValueLayout.JAVA_INT, i);
            int valB = b.getData().getAtIndex(ValueLayout.JAVA_INT, i);
            total += ((double)valA * (double)valB);
        }
        return total;
    }

    public static double dotDouble(NDArray a,NDArray b){
        if(!a.isContiguous() || !b.isContiguous()){
            double total = 0.0;
            NDIter iterA = new NDIter(a.internalShapeUnsafe());
            NDIter iterB = new NDIter(b.internalShapeUnsafe());
            while(iterA.hasNext){
                long byteOffsetA = ShapeUtil.getByteOffset(iterA.coords, a.internalStridesUnsafe(), a.getDType());
                long byteOffsetB = ShapeUtil.getByteOffset(iterB.coords, b.internalStridesUnsafe(), b.getDType());
                total += a.getData().get(ValueLayout.JAVA_DOUBLE, byteOffsetA) * b.getData().get(ValueLayout.JAVA_DOUBLE, byteOffsetB);
                iterA.next();
                iterB.next();
            }
            return total;
        }
        long i=0;
        long loopbound= a.getSize() - (a.getSize() % (DB_VL * 2));
        var vSum1 = DoubleVector.zero(SPECIESDB);
        var vSum2 = DoubleVector.zero(SPECIESDB);

        for(;i<loopbound;i+=DB_VL*2){
            var va1=DoubleVector.fromMemorySegment(SPECIESDB, a.getData(), i*DB_BYTES, ORDER);
            var vb1=DoubleVector.fromMemorySegment(SPECIESDB, b.getData(), i*DB_BYTES, ORDER);
            var va2=DoubleVector.fromMemorySegment(SPECIESDB, a.getData(), (i+DB_VL)*DB_BYTES, ORDER);
            var vb2=DoubleVector.fromMemorySegment(SPECIESDB, b.getData(), (i+DB_VL)*DB_BYTES, ORDER);
            vSum1=va1.fma(vb1, vSum1);
            vSum2=va2.fma(vb2, vSum2);
        }

        loopbound=SPECIES.loopBound(a.getSize());

        for(;i<loopbound;i+=DB_VL){
            var v1=DoubleVector.fromMemorySegment(SPECIESDB, a.getData(), i*DB_BYTES, ORDER);
            var v2=DoubleVector.fromMemorySegment(SPECIESDB, b.getData(), i*DB_BYTES, ORDER);
            vSum1=v1.fma(v2,vSum1);
        }

        double total = vSum1.reduceLanes(VectorOperators.ADD);
        total+=vSum2.reduceLanes(VectorOperators.ADD);
        for(; i< a.getSize(); i++){
            double valA = a.getData().getAtIndex(ValueLayout.JAVA_DOUBLE, i);
            double valB = b.getData().getAtIndex(ValueLayout.JAVA_DOUBLE, i);
            total += (valA * valB);
        }
        return total;
    }

}
