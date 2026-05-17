package jnum.jnumops;

import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;
import jdk.incubator.vector.VectorOperators;

import jnum.NDArray;

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
        long i=0;
        long loopbound=SPECIES.loopBound(a.size);
        var vSum=FloatVector.zero(SPECIES);
        for(;i<loopbound;i+=VL){
            var v1=FloatVector.fromMemorySegment(SPECIES,a.data,i*FLOAT_BYTES,ORDER);
            vSum=vSum.add(v1);
        }
        float total=vSum.reduceLanes(VectorOperators.ADD);
        for(;i<a.size;i++){
            total+=a.data.getAtIndex(ValueLayout.JAVA_FLOAT,i);
        }
        return (double) total;
    }

    public static double sumInt(NDArray a){
        long i=0;
        long loopbound=SPECIESINT.loopBound(a.size);
        var vSum=IntVector.zero(SPECIESINT);
        for(;i<loopbound;i+=INT_VL){
            var v1=IntVector.fromMemorySegment(SPECIESINT,a.data,i*INT_BYTES,ORDER);
            vSum=vSum.add(v1);
        }
        int total=vSum.reduceLanes(VectorOperators.ADD);
        for(;i<a.size;i++){
            total+=a.data.getAtIndex(ValueLayout.JAVA_INT,i);
        }
        return (double) total;
    }

    public static double sumDouble(NDArray a){
        long i=0;
        long loopbound=SPECIESDB.loopBound(a.size);
        var vSum=DoubleVector.zero(SPECIESDB);
        for(;i<loopbound;i+=DB_VL){
            var v1=DoubleVector.fromMemorySegment(SPECIESDB,a.data,i*DB_BYTES,ORDER);
            vSum=vSum.add(v1);
        }
        double total=vSum.reduceLanes(VectorOperators.ADD);
        for(;i<a.size;i++){
            total+=a.data.getAtIndex(ValueLayout.JAVA_DOUBLE,i);
        }
        return (double) total;
    }

    public static NDArray sumFloatAxis(NDArray a,int axis,NDArray resArray){
        int size=a.shape[axis];
        long strideA=a.strides[axis];

        for(int i=0;i<resArray.size();i++){
            long tempIndex=i;
            long baseOffset=0;
            long offsetRes=0;
            for (int d = resArray.shape.length - 1; d >= 0; d--) {
                long coord = tempIndex % resArray.shape[d];
                tempIndex /= resArray.shape[d];
                offsetRes += coord * resArray.strides[d];
                
                int aDim = (d >= axis) ? d + 1 : d;
                baseOffset += coord * a.strides[aDim];
            }

            if(strideA==1){
                var vAcc = FloatVector.zero(SPECIES);
                int k = 0;
                int loopbound = SPECIES.loopBound(size);
                for(; k<loopbound ;k += VL){
                    var vVal = FloatVector.fromMemorySegment(SPECIES, a.data, (baseOffset + k) * 4L, ORDER);
                    vAcc = vAcc.add(vVal);
                }
                float acc = vAcc.reduceLanes(VectorOperators.ADD);
                for (; k < size; k++) {
                    acc += a.data.getAtIndex(ValueLayout.JAVA_FLOAT, baseOffset + k);
                }
                resArray.data.setAtIndex(ValueLayout.JAVA_FLOAT, offsetRes, acc);
            }else{
                float acc = 0f;
                for (int k = 0; k < size; k++) {
                    acc += a.data.getAtIndex(ValueLayout.JAVA_FLOAT, baseOffset + k * strideA);
                }
                resArray.data.setAtIndex(ValueLayout.JAVA_FLOAT, offsetRes, acc);
            }
        }
        return resArray;
    }

    public static NDArray sumIntAxis(NDArray a,int axis,NDArray resArray){
        int size=a.shape[axis];
        long strideA=a.strides[axis];

        for(int i=0;i<resArray.size();i++){
            long tempIndex=i;
            long baseOffset=0;
            long offsetRes=0;
            for (int d = resArray.shape.length - 1; d >= 0; d--) {
                long coord = tempIndex % resArray.shape[d];
                tempIndex /= resArray.shape[d];
                offsetRes += coord * resArray.strides[d];
                
                int aDim = (d >= axis) ? d + 1 : d;
                baseOffset += coord * a.strides[aDim];
            }

            if(strideA==1){
                var vAcc = IntVector.zero(SPECIESINT);
                int k = 0;
                int loopbound = SPECIESINT.loopBound(size);
                for(; k<loopbound ;k += INT_VL){
                    var vVal = IntVector.fromMemorySegment(SPECIESINT, a.data, (baseOffset + k) * 4L, ORDER);
                    vAcc = vAcc.add(vVal);
                }
                int acc = vAcc.reduceLanes(VectorOperators.ADD);
                for (; k < size; k++) {
                    acc += a.data.getAtIndex(ValueLayout.JAVA_INT, baseOffset + k);
                }
                resArray.data.setAtIndex(ValueLayout.JAVA_INT, offsetRes, acc);
            }else{
                int acc = 0;
                for (int k = 0; k < size; k++) {
                    acc += a.data.getAtIndex(ValueLayout.JAVA_INT, baseOffset + k * strideA);
                }
                resArray.data.setAtIndex(ValueLayout.JAVA_INT, offsetRes, acc);
            }
        }
        return resArray;
    }

    public static NDArray sumDoubleAxis(NDArray a,int axis,NDArray resArray){
        int size=a.shape[axis];
        long strideA=a.strides[axis];

        for(int i=0;i<resArray.size();i++){
            long tempIndex=i;
            long baseOffset=0;
            long offsetRes=0;
            for (int d = resArray.shape.length - 1; d >= 0; d--) {
                long coord = tempIndex % resArray.shape[d];
                tempIndex /= resArray.shape[d];
                offsetRes += coord * resArray.strides[d];
                
                int aDim = (d >= axis) ? d + 1 : d;
                baseOffset += coord * a.strides[aDim];
            }

            if(strideA==1){
                var vAcc = DoubleVector.zero(SPECIESDB);
                int k = 0;
                int loopbound = SPECIESDB.loopBound(size);
                for(; k<loopbound ;k += DB_VL){
                    var vVal = DoubleVector.fromMemorySegment(SPECIESDB, a.data, (baseOffset + k) * 8L, ORDER);
                    vAcc = vAcc.add(vVal);
                }
                double acc = vAcc.reduceLanes(VectorOperators.ADD);
                for (; k < size; k++) {
                    acc += a.data.getAtIndex(ValueLayout.JAVA_DOUBLE, baseOffset + k);
                }
                resArray.data.setAtIndex(ValueLayout.JAVA_DOUBLE, offsetRes, acc);
            }else{
                double acc = 0f;
                for (int k = 0; k < size; k++) {
                    acc += a.data.getAtIndex(ValueLayout.JAVA_DOUBLE, baseOffset + k * strideA);
                }
                resArray.data.setAtIndex(ValueLayout.JAVA_DOUBLE, offsetRes, acc);
            }
        }
        return resArray;
    }

    public static double maxFloat(NDArray a){
        long i=0;
        long loopbound=SPECIES.loopBound(a.size);
        var vMax = FloatVector.broadcast(SPECIES, Float.NEGATIVE_INFINITY);
        for(;i<loopbound;i+=VL){
            var v1=FloatVector.fromMemorySegment(SPECIES,a.data,i*FLOAT_BYTES,ORDER);
            vMax=vMax.max(v1);
        }
        float finalMax=vMax.reduceLanes(VectorOperators.MAX);
        for (; i < a.size; i++) {
            float tailVal = a.data.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            if (tailVal > finalMax) {
                finalMax = tailVal;
            }
        }
        return (double) finalMax;
    }

    public static NDArray maxFloatAxis(NDArray a,int axis,NDArray resArray){
        int size=a.shape[axis];
        long strideA=a.strides[axis];

        for(int i=0;i<resArray.size();i++){
            long tempIndex=i;
            long baseOffset=0;
            long offsetRes=0;
            for (int d = resArray.shape.length - 1; d >= 0; d--) {
                long coord = tempIndex % resArray.shape[d];
                tempIndex /= resArray.shape[d];
                offsetRes += coord * resArray.strides[d];
                
                int aDim = (d >= axis) ? d + 1 : d;
                baseOffset += coord * a.strides[aDim];
            }

            if(strideA==1){
                var vAcc = FloatVector.broadcast(SPECIES, Float.NEGATIVE_INFINITY);
                int k = 0;
                int loopbound = SPECIES.loopBound(size);
                for(; k<loopbound ;k += VL){
                    var vVal = FloatVector.fromMemorySegment(SPECIES, a.data, (baseOffset + k) * 4L, ORDER);
                    vAcc = vAcc.max(vVal);
                }
                float acc = vAcc.reduceLanes(VectorOperators.MAX);
                for (; k < size; k++) {
                    acc = Math.max(acc, a.data.getAtIndex(ValueLayout.JAVA_FLOAT, baseOffset + k));
                }
                resArray.data.setAtIndex(ValueLayout.JAVA_FLOAT, offsetRes, acc);
            }else{
                float acc = Float.NEGATIVE_INFINITY;
                for (int k = 0; k < size; k++) {
                    acc = Math.max(acc, a.data.getAtIndex(ValueLayout.JAVA_FLOAT, baseOffset + k * strideA));
                }
                resArray.data.setAtIndex(ValueLayout.JAVA_FLOAT, offsetRes, acc);
            }
        }
        return resArray;
    }

    public static double maxInt(NDArray a) {
        long i = 0;
        long loopbound = SPECIESINT.loopBound(a.size);
        var vMax = IntVector.broadcast(SPECIESINT, Integer.MIN_VALUE);
        for (; i < loopbound; i += INT_VL) {
            var v1 = IntVector.fromMemorySegment(SPECIESINT, a.data, i * INT_BYTES, ORDER);
            vMax = vMax.max(v1);
        }
        int finalMax = vMax.reduceLanes(VectorOperators.MAX);
        for (; i < a.size; i++) {
            int tailVal = a.data.getAtIndex(ValueLayout.JAVA_INT, i);
            if (tailVal > finalMax) finalMax = tailVal;
        }
        return (double) finalMax;
    }

    public static NDArray maxIntAxis(NDArray a,int axis,NDArray resArray){
        int size=a.shape[axis];
        long strideA=a.strides[axis];

        for(int i=0;i<resArray.size();i++){
            long tempIndex=i;
            long baseOffset=0;
            long offsetRes=0;
            for (int d = resArray.shape.length - 1; d >= 0; d--) {
                long coord = tempIndex % resArray.shape[d];
                tempIndex /= resArray.shape[d];
                offsetRes += coord * resArray.strides[d];
                
                int aDim = (d >= axis) ? d + 1 : d;
                baseOffset += coord * a.strides[aDim];
            }

            if(strideA==1){
                var vAcc = IntVector.broadcast(SPECIESINT, Integer.MIN_VALUE);
                int k = 0;
                int loopbound = SPECIESINT.loopBound(size);
                for(; k<loopbound ;k += INT_VL){
                    var vVal = IntVector.fromMemorySegment(SPECIESINT, a.data, (baseOffset + k) * 4L, ORDER);
                    vAcc = vAcc.max(vVal);
                }
                int acc = vAcc.reduceLanes(VectorOperators.MAX);
                for (; k < size; k++) {
                    acc = Math.max(acc, a.data.getAtIndex(ValueLayout.JAVA_INT, baseOffset + k));
                }
                resArray.data.setAtIndex(ValueLayout.JAVA_INT, offsetRes, acc);
            }else{
                int acc = Integer.MIN_VALUE;
                for (int k = 0; k < size; k++) {
                    acc = Math.max(acc, a.data.getAtIndex(ValueLayout.JAVA_INT, baseOffset + k * strideA));
                }
                resArray.data.setAtIndex(ValueLayout.JAVA_INT, offsetRes, acc);
            }
        }
        return resArray;
    }

    public static double maxDouble(NDArray a) {
        long i = 0;
        long loopbound = SPECIESDB.loopBound(a.size);
        var vMax = DoubleVector.broadcast(SPECIESDB, Double.NEGATIVE_INFINITY);
        for (; i < loopbound; i += DB_VL) {
            var v1 = DoubleVector.fromMemorySegment(SPECIESDB, a.data, i * DB_BYTES, ORDER);
            vMax = vMax.max(v1);
        }
        double finalMax = vMax.reduceLanes(VectorOperators.MAX);
        for (; i < a.size; i++) {
            double tailVal = a.data.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
            if (tailVal > finalMax) finalMax = tailVal;
        }
        return (double) finalMax;
    }

    public static NDArray maxDoubleAxis(NDArray a,int axis,NDArray resArray){
        int size=a.shape[axis];
        long strideA=a.strides[axis];

        for(int i=0;i<resArray.size();i++){
            long tempIndex=i;
            long baseOffset=0;
            long offsetRes=0;
            for (int d = resArray.shape.length - 1; d >= 0; d--) {
                long coord = tempIndex % resArray.shape[d];
                tempIndex /= resArray.shape[d];
                offsetRes += coord * resArray.strides[d];
                
                int aDim = (d >= axis) ? d + 1 : d;
                baseOffset += coord * a.strides[aDim];
            }

            if(strideA==1){
                var vAcc = DoubleVector.broadcast(SPECIESDB, Double.NEGATIVE_INFINITY);
                int k = 0;
                int loopbound = SPECIESDB.loopBound(size);
                for(; k<loopbound ;k += DB_VL){
                    var vVal = DoubleVector.fromMemorySegment(SPECIESDB, a.data, (baseOffset + k) * 8L, ORDER);
                    vAcc = vAcc.max(vVal);
                }
                double acc = vAcc.reduceLanes(VectorOperators.MAX);
                for (; k < size; k++) {
                    acc = Math.max(acc, a.data.getAtIndex(ValueLayout.JAVA_DOUBLE, baseOffset + k));
                }
                resArray.data.setAtIndex(ValueLayout.JAVA_DOUBLE, offsetRes, acc);
            }else{
                double acc = Double.NEGATIVE_INFINITY;
                for (int k = 0; k < size; k++) {
                    acc = Math.max(acc, a.data.getAtIndex(ValueLayout.JAVA_DOUBLE, baseOffset + k * strideA));
                }
                resArray.data.setAtIndex(ValueLayout.JAVA_DOUBLE, offsetRes, acc);
            }
        }
        return resArray;
    }

    public static double minFloat(NDArray a){
        long i=0;
        long loopbound=SPECIES.loopBound(a.size);
        var vMin=FloatVector.broadcast(SPECIES,Float.POSITIVE_INFINITY);
        for(;i<loopbound;i+=VL){
            var v1=FloatVector.fromMemorySegment(SPECIES,a.data,i*FLOAT_BYTES,ORDER);
            vMin=vMin.min(v1);
        }
        float finalMin=vMin.reduceLanes(VectorOperators.MIN);
        for (; i < a.size; i++) {
            float tailVal = a.data.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            if (tailVal < finalMin) {
                finalMin = tailVal;
            }
        }
        return (double) finalMin;
    }

    public static NDArray minFloatAxis(NDArray a,int axis,NDArray resArray){
        int size=a.shape[axis];
        long strideA=a.strides[axis];

        for(int i=0;i<resArray.size();i++){
            long tempIndex=i;
            long baseOffset=0;
            long offsetRes=0;
            for (int d = resArray.shape.length - 1; d >= 0; d--) {
                long coord = tempIndex % resArray.shape[d];
                tempIndex /= resArray.shape[d];
                offsetRes += coord * resArray.strides[d];
                
                int aDim = (d >= axis) ? d + 1 : d;
                baseOffset += coord * a.strides[aDim];
            }

            if(strideA==1){
                var vAcc = FloatVector.broadcast(SPECIES, Float.POSITIVE_INFINITY);
                int k = 0;
                int loopbound = SPECIES.loopBound(size);
                for(; k<loopbound ;k += VL){
                    var vVal = FloatVector.fromMemorySegment(SPECIES, a.data, (baseOffset + k) * 4L, ORDER);
                    vAcc = vAcc.min(vVal);
                }
                float acc = vAcc.reduceLanes(VectorOperators.MIN);
                for (; k < size; k++) {
                    acc = Math.min(acc, a.data.getAtIndex(ValueLayout.JAVA_FLOAT, baseOffset + k));
                }
                resArray.data.setAtIndex(ValueLayout.JAVA_FLOAT, offsetRes, acc);
            }else{
                float acc = Float.POSITIVE_INFINITY;
                for (int k = 0; k < size; k++) {
                    acc = Math.min(acc, a.data.getAtIndex(ValueLayout.JAVA_FLOAT, baseOffset + k * strideA));
                }
                resArray.data.setAtIndex(ValueLayout.JAVA_FLOAT, offsetRes, acc);
            }
        }
        return resArray;
    }

    public static double minInt(NDArray a) {
        long i = 0;
        long loopbound = SPECIESINT.loopBound(a.size);
        var vMin = IntVector.broadcast(SPECIESINT, Integer.MAX_VALUE);
        for (; i < loopbound; i += INT_VL) {
            var v1 = IntVector.fromMemorySegment(SPECIESINT, a.data, i * INT_BYTES, ORDER);
            vMin = vMin.min(v1);
        }
        int finalMin = vMin.reduceLanes(VectorOperators.MIN);
        for (; i < a.size; i++) {
            int tailVal = a.data.getAtIndex(ValueLayout.JAVA_INT, i);
            if (tailVal < finalMin) finalMin = tailVal;
        }
        return (double) finalMin;
    }

    public static NDArray minIntAxis(NDArray a,int axis,NDArray resArray){
        int size=a.shape[axis];
        long strideA=a.strides[axis];

        for(int i=0;i<resArray.size();i++){
            long tempIndex=i;
            long baseOffset=0;
            long offsetRes=0;
            for (int d = resArray.shape.length - 1; d >= 0; d--) {
                long coord = tempIndex % resArray.shape[d];
                tempIndex /= resArray.shape[d];
                offsetRes += coord * resArray.strides[d];
                
                int aDim = (d >= axis) ? d + 1 : d;
                baseOffset += coord * a.strides[aDim];
            }

            if(strideA==1){
                var vAcc = IntVector.broadcast(SPECIESINT, Integer.MAX_VALUE);
                int k = 0;
                int loopbound = SPECIESINT.loopBound(size);
                for(; k<loopbound ;k += INT_VL){
                    var vVal = IntVector.fromMemorySegment(SPECIESINT, a.data, (baseOffset + k) * 4L, ORDER);
                    vAcc = vAcc.min(vVal);
                }
                int acc = vAcc.reduceLanes(VectorOperators.MIN);
                for (; k < size; k++) {
                    acc = Math.min(acc, a.data.getAtIndex(ValueLayout.JAVA_INT, baseOffset + k));
                }
                resArray.data.setAtIndex(ValueLayout.JAVA_INT, offsetRes, acc);
            }else{
                int acc = Integer.MAX_VALUE;
                for (int k = 0; k < size; k++) {
                    acc = Math.min(acc, a.data.getAtIndex(ValueLayout.JAVA_INT, baseOffset + k * strideA));
                }
                resArray.data.setAtIndex(ValueLayout.JAVA_INT, offsetRes, acc);
            }
        }
        return resArray;
    }

    public static double minDouble(NDArray a) {
        long i = 0;
        long loopbound = SPECIESDB.loopBound(a.size);
        var vMin = DoubleVector.broadcast(SPECIESDB, Double.POSITIVE_INFINITY);
        for (; i < loopbound; i += DB_VL) {
            var v1 = DoubleVector.fromMemorySegment(SPECIESDB, a.data, i * DB_BYTES, ORDER);
            vMin = vMin.min(v1);
        }
        double finalMin = vMin.reduceLanes(VectorOperators.MIN);
        for (; i < a.size; i++) {
            double tailVal = a.data.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
            if (tailVal < finalMin) finalMin = tailVal;
        }
        return (double) finalMin;
    }

    public static NDArray minDoubleAxis(NDArray a,int axis,NDArray resArray){
        int size=a.shape[axis];
        long strideA=a.strides[axis];

        for(int i=0;i<resArray.size();i++){
            long tempIndex=i;
            long baseOffset=0;
            long offsetRes=0;
            for (int d = resArray.shape.length - 1; d >= 0; d--) {
                long coord = tempIndex % resArray.shape[d];
                tempIndex /= resArray.shape[d];
                offsetRes += coord * resArray.strides[d];
                
                int aDim = (d >= axis) ? d + 1 : d;
                baseOffset += coord * a.strides[aDim];
            }

            if(strideA==1){
                var vAcc = DoubleVector.broadcast(SPECIESDB, Double.POSITIVE_INFINITY);
                int k = 0;
                int loopbound = SPECIESDB.loopBound(size);
                for(; k<loopbound ;k += DB_VL){
                    var vVal = DoubleVector.fromMemorySegment(SPECIESDB, a.data, (baseOffset + k) * 8L, ORDER);
                    vAcc = vAcc.min(vVal);
                }
                double acc = vAcc.reduceLanes(VectorOperators.MIN);
                for (; k < size; k++) {
                    acc = Math.min(acc, a.data.getAtIndex(ValueLayout.JAVA_DOUBLE, baseOffset + k));
                }
                resArray.data.setAtIndex(ValueLayout.JAVA_DOUBLE, offsetRes, acc);
            }else{
                double acc = Double.POSITIVE_INFINITY;
                for (int k = 0; k < size; k++) {
                    acc = Math.min(acc, a.data.getAtIndex(ValueLayout.JAVA_DOUBLE, baseOffset + k * strideA));
                }
                resArray.data.setAtIndex(ValueLayout.JAVA_DOUBLE, offsetRes, acc);
            }
        }
        return resArray;
    }

    public static double dotFloat(NDArray a,NDArray b){
        long i=0;
        long loopbound=a.size - (a.size % (VL * 2));
        var vSum1 = FloatVector.zero(SPECIES);
        var vSum2 = FloatVector.zero(SPECIES);

        for(;i<loopbound;i+=VL*2){
            var va1=FloatVector.fromMemorySegment(SPECIES, a.data, i*FLOAT_BYTES, ORDER);
            var vb1=FloatVector.fromMemorySegment(SPECIES, b.data, i*FLOAT_BYTES, ORDER);
            var va2=FloatVector.fromMemorySegment(SPECIES, a.data, (i+VL)*FLOAT_BYTES, ORDER);
            var vb2=FloatVector.fromMemorySegment(SPECIES, b.data, (i+VL)*FLOAT_BYTES, ORDER);
            vSum1=va1.fma(vb1,vSum1);
            vSum2=va2.fma(vb2, vSum2);
        }

        loopbound=SPECIES.loopBound(a.size);

        for(;i<loopbound;i+=VL){
            var v1=FloatVector.fromMemorySegment(SPECIES, a.data, i*FLOAT_BYTES, ORDER);
            var v2=FloatVector.fromMemorySegment(SPECIES, b.data, i*FLOAT_BYTES, ORDER);
            vSum1=v1.fma(v2,vSum1);
        }

        double total = vSum1.reduceLanes(VectorOperators.ADD);
        total+=vSum2.reduceLanes(VectorOperators.ADD);
        for(;i<a.size;i++){
            float valA = a.data.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            float valB = b.data.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            total += (valA * valB);
        }
        return total;
    }

    public static double dotInt(NDArray a,NDArray b){
        long i=0;
        long loopbound=a.size - (a.size % (INT_VL * 2));
        var vSum1 = IntVector.zero(SPECIESINT);
        var vSum2 = IntVector.zero(SPECIESINT);

        for(;i<loopbound;i+=INT_VL*2){
            var va1=IntVector.fromMemorySegment(SPECIESINT, a.data, i*INT_BYTES, ORDER);
            var vb1=IntVector.fromMemorySegment(SPECIESINT, b.data, i*INT_BYTES, ORDER);
            var va2=IntVector.fromMemorySegment(SPECIESINT, a.data, (i+INT_VL)*INT_BYTES, ORDER);
            var vb2=IntVector.fromMemorySegment(SPECIESINT, b.data, (i+INT_VL)*INT_BYTES, ORDER);
            vSum1 = va1.mul(vb1).add(vSum1);
            vSum2 = va2.mul(vb2).add(vSum2);
        }

        loopbound=SPECIES.loopBound(a.size);

        for(;i<loopbound;i+=INT_VL){
            var v1=IntVector.fromMemorySegment(SPECIESINT, a.data, i*INT_BYTES, ORDER);
            var v2=IntVector.fromMemorySegment(SPECIESINT, b.data, i*INT_BYTES, ORDER);
            vSum1=v1.mul(v2).add(vSum1);
        }

        double total = vSum1.reduceLanes(VectorOperators.ADD);
        total+=vSum2.reduceLanes(VectorOperators.ADD);
        for(;i<a.size;i++){
            int valA = a.data.getAtIndex(ValueLayout.JAVA_INT, i);
            int valB = b.data.getAtIndex(ValueLayout.JAVA_INT, i);
            total += ((double)valA * (double)valB);
        }
        return total;
    }

    public static double dotDouble(NDArray a,NDArray b){
        long i=0;
        long loopbound=a.size - (a.size % (DB_VL * 2));
        var vSum1 = DoubleVector.zero(SPECIESDB);
        var vSum2 = DoubleVector.zero(SPECIESDB);

        for(;i<loopbound;i+=DB_VL*2){
            var va1=DoubleVector.fromMemorySegment(SPECIESDB, a.data, i*DB_BYTES, ORDER);
            var vb1=DoubleVector.fromMemorySegment(SPECIESDB, b.data, i*DB_BYTES, ORDER);
            var va2=DoubleVector.fromMemorySegment(SPECIESDB, a.data, (i+DB_VL)*DB_BYTES, ORDER);
            var vb2=DoubleVector.fromMemorySegment(SPECIESDB, b.data, (i+DB_VL)*DB_BYTES, ORDER);
            vSum1=va1.fma(vb1, vSum1);
            vSum2=va2.fma(vb2, vSum2);
        }

        loopbound=SPECIES.loopBound(a.size);

        for(;i<loopbound;i+=DB_VL){
            var v1=DoubleVector.fromMemorySegment(SPECIESDB, a.data, i*DB_BYTES, ORDER);
            var v2=DoubleVector.fromMemorySegment(SPECIESDB, b.data, i*DB_BYTES, ORDER);
            vSum1=v1.fma(v2,vSum1);
        }

        double total = vSum1.reduceLanes(VectorOperators.ADD);
        total+=vSum2.reduceLanes(VectorOperators.ADD);
        for(;i<a.size;i++){
            double valA = a.data.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
            double valB = b.data.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
            total += (valA * valB);
        }
        return total;
    }

}
