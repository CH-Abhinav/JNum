package jnum.jnumops;


import jnum.NDArray;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

public class CompareOps {

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

    private CompareOps() {
        throw new AssertionError();
    }

    public static NDArray maximumFloat(NDArray a,NDArray b,NDArray resArray){
        if (a.isContiguous() && b.isContiguous() && resArray.isContiguous()) {
            long i = 0;
            long loopBound = SPECIES.loopBound(a.size);
            
            for (; i < loopBound; i += VL) {
                var vA = FloatVector.fromMemorySegment(SPECIES, a.data, i * FLOAT_BYTES, ORDER);
                var vB = FloatVector.fromMemorySegment(SPECIES, b.data, i * FLOAT_BYTES, ORDER);
                var vRes = vA.max(vB);
                vRes.intoMemorySegment(resArray.data, i * FLOAT_BYTES, ORDER);
            }

            for (; i < a.size; i++) {
                float valA = a.data.getAtIndex(ValueLayout.JAVA_FLOAT, i);
                float valB = b.data.getAtIndex(ValueLayout.JAVA_FLOAT, i);
                resArray.data.setAtIndex(ValueLayout.JAVA_FLOAT, i, Math.max(valA, valB));
            }
    }else {
            var iterA = new NDIter(resArray.shape, a.strides);
            var iterB = new NDIter(resArray.shape, b.strides);
            var iterRes = new NDIter(resArray.shape, resArray.strides);

            while (iterA.hasNext) {
                float valA = a.data.getAtIndex(ValueLayout.JAVA_FLOAT, iterA.offset);
                float valB = b.data.getAtIndex(ValueLayout.JAVA_FLOAT, iterB.offset);
                resArray.data.setAtIndex(ValueLayout.JAVA_FLOAT, iterRes.offset, Math.max(valA, valB));
                
                iterA.next(); iterB.next(); iterRes.next();
            }
        }
        return resArray;
    }

    public static NDArray maximumFloat(NDArray a,float b,NDArray resArray){
        var vB=FloatVector.broadcast(SPECIES, b);

        if(a.isContiguous() && resArray.isContiguous()){
            long i = 0;
            long loopBound = SPECIES.loopBound(a.size);
            
            for (; i < loopBound; i += VL) {
                var vA = FloatVector.fromMemorySegment(SPECIES, a.data, i * FLOAT_BYTES, ORDER);
                var vRes = vA.max(vB);
                vRes.intoMemorySegment(resArray.data, i * FLOAT_BYTES, ORDER);
            }

            for (; i < a.size; i++) {
                float valA = a.data.getAtIndex(ValueLayout.JAVA_FLOAT, i);
                resArray.data.setAtIndex(ValueLayout.JAVA_FLOAT, i, Math.max(valA, b));
            }
        }else{
            var iterA=new NDIter(resArray.shape, a.strides);
            var iterRes=new NDIter(resArray.shape, resArray.strides);

            while(iterA.hasNext){
                float valA= a.data.getAtIndex(ValueLayout.JAVA_FLOAT, iterA.offset);
                resArray.data.setAtIndex(ValueLayout.JAVA_FLOAT, iterRes.offset, Math.max(valA,b));
                
                iterA.next(); iterRes.next();
            }
        }
        return resArray;
    }

    public static NDArray maximumDouble(NDArray a,NDArray b,NDArray resArray){
        if (a.isContiguous() && b.isContiguous() && resArray.isContiguous()) {
            long i = 0;
            long loopBound = SPECIESDB.loopBound(a.size);
            
            for (; i < loopBound; i += DB_VL) {
                var vA = DoubleVector.fromMemorySegment(SPECIESDB, a.data, i * DB_BYTES, ORDER);
                var vB = DoubleVector.fromMemorySegment(SPECIESDB, b.data, i * DB_BYTES, ORDER);
                var vRes = vA.max(vB);
                vRes.intoMemorySegment(resArray.data, i * DB_BYTES, ORDER);
            }

            for (; i < a.size; i++) {
                double valA = a.data.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
                double valB = b.data.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
                resArray.data.setAtIndex(ValueLayout.JAVA_DOUBLE, i, Math.max(valA, valB));
            }
        } else {
            var iterA = new NDIter(resArray.shape, a.strides);
            var iterB = new NDIter(resArray.shape, b.strides);
            var iterRes = new NDIter(resArray.shape, resArray.strides);

            while (iterA.hasNext) {
                double valA = a.data.getAtIndex(ValueLayout.JAVA_DOUBLE, iterA.offset);
                double valB = b.data.getAtIndex(ValueLayout.JAVA_DOUBLE, iterB.offset);
                resArray.data.setAtIndex(ValueLayout.JAVA_DOUBLE, iterRes.offset, Math.max(valA, valB));

                iterA.next(); iterB.next(); iterRes.next();
            }
        }
        return resArray;
    }

    public static NDArray maximumDouble(NDArray a,double b,NDArray resArray){
        var vB = DoubleVector.broadcast(SPECIESDB, b);

        if(a.isContiguous() && resArray.isContiguous()){
            long i = 0;
            long loopBound = SPECIESDB.loopBound(a.size);
            
            for (; i < loopBound; i += DB_VL) {
                var vA = DoubleVector.fromMemorySegment(SPECIESDB, a.data, i * DB_BYTES, ORDER);
                var vRes = vA.max(vB);
                vRes.intoMemorySegment(resArray.data, i * DB_BYTES, ORDER);
            }

            for (; i < a.size; i++) {
                double valA = a.data.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
                resArray.data.setAtIndex(ValueLayout.JAVA_DOUBLE, i, Math.max(valA, b));
            }
        } else {
            var iterA = new NDIter(resArray.shape, a.strides);
            var iterRes = new NDIter(resArray.shape, resArray.strides);

            while(iterA.hasNext){
                double valA = a.data.getAtIndex(ValueLayout.JAVA_DOUBLE, iterA.offset);
                resArray.data.setAtIndex(ValueLayout.JAVA_DOUBLE, iterRes.offset, Math.max(valA, b));
                
                iterA.next(); iterRes.next();
            }
        }
        return resArray;
    }

    public static NDArray maximumInt(NDArray a,NDArray b,NDArray resArray){
        if (a.isContiguous() && b.isContiguous() && resArray.isContiguous()) {
            long i = 0;
            long loopBound = SPECIESINT.loopBound(a.size);
            
            for (; i < loopBound; i += INT_VL) {
                var vA = IntVector.fromMemorySegment(SPECIESINT, a.data, i * INT_BYTES, ORDER);
                var vB = IntVector.fromMemorySegment(SPECIESINT, b.data, i * INT_BYTES, ORDER);
                var vRes = vA.max(vB);
                vRes.intoMemorySegment(resArray.data, i * INT_BYTES, ORDER);
            }

            for (; i < a.size; i++) {
                int valA = a.data.getAtIndex(ValueLayout.JAVA_INT, i);
                int valB = b.data.getAtIndex(ValueLayout.JAVA_INT, i);
                resArray.data.setAtIndex(ValueLayout.JAVA_INT, i, Math.max(valA, valB));
            }
        } else {
            var iterA = new NDIter(resArray.shape, a.strides);
            var iterB = new NDIter(resArray.shape, b.strides);
            var iterRes = new NDIter(resArray.shape, resArray.strides);

            while (iterA.hasNext) {
                int valA = a.data.getAtIndex(ValueLayout.JAVA_INT, iterA.offset);
                int valB = b.data.getAtIndex(ValueLayout.JAVA_INT, iterB.offset);
                resArray.data.setAtIndex(ValueLayout.JAVA_INT, iterRes.offset, Math.max(valA, valB));

                iterA.next(); iterB.next(); iterRes.next();
            }
        }
        return resArray;
    }

    public static NDArray maximumInt(NDArray a,int b,NDArray resArray){
        var vB = IntVector.broadcast(SPECIESINT, b);

        if(a.isContiguous() && resArray.isContiguous()){
            long i = 0;
            long loopBound = SPECIESINT.loopBound(a.size);
            
            for (; i < loopBound; i += INT_VL) {
                var vA = IntVector.fromMemorySegment(SPECIESINT, a.data, i * INT_BYTES, ORDER);
                var vRes = vA.max(vB);
                vRes.intoMemorySegment(resArray.data, i * INT_BYTES, ORDER);
            }

            for (; i < a.size; i++) {
                int valA = a.data.getAtIndex(ValueLayout.JAVA_INT, i);
                resArray.data.setAtIndex(ValueLayout.JAVA_INT, i, Math.max(valA, b));
            }
        } else {
            var iterA = new NDIter(resArray.shape, a.strides);
            var iterRes = new NDIter(resArray.shape, resArray.strides);

            while(iterA.hasNext){
                int valA = a.data.getAtIndex(ValueLayout.JAVA_INT, iterA.offset);
                resArray.data.setAtIndex(ValueLayout.JAVA_INT, iterRes.offset, Math.max(valA, b));
                
                iterA.next(); iterRes.next();
            }
        }
        return resArray;
    }

    public static NDArray minimumFloat(NDArray a,NDArray b,NDArray resArray){
        if (a.isContiguous() && b.isContiguous() && resArray.isContiguous()) {
            long i = 0;
            long loopBound = SPECIES.loopBound(a.size);
            
            for (; i < loopBound; i += VL) {
                var vA = FloatVector.fromMemorySegment(SPECIES, a.data, i * FLOAT_BYTES, ORDER);
                var vB = FloatVector.fromMemorySegment(SPECIES, b.data, i * FLOAT_BYTES, ORDER);
                var vRes = vA.min(vB);
                vRes.intoMemorySegment(resArray.data, i * FLOAT_BYTES, ORDER);
            }

            for (; i < a.size; i++) {
                float valA = a.data.getAtIndex(ValueLayout.JAVA_FLOAT, i);
                float valB = b.data.getAtIndex(ValueLayout.JAVA_FLOAT, i);
                resArray.data.setAtIndex(ValueLayout.JAVA_FLOAT, i, Math.min(valA, valB));
            }
        } else {
            var iterA = new NDIter(resArray.shape, a.strides);
            var iterB = new NDIter(resArray.shape, b.strides);
            var iterRes = new NDIter(resArray.shape, resArray.strides);

            while (iterA.hasNext) {
                float valA = a.data.getAtIndex(ValueLayout.JAVA_FLOAT, iterA.offset);
                float valB = b.data.getAtIndex(ValueLayout.JAVA_FLOAT, iterB.offset);
                resArray.data.setAtIndex(ValueLayout.JAVA_FLOAT, iterRes.offset, Math.min(valA, valB));
                
                iterA.next(); iterB.next(); iterRes.next();
            }
        }
        return resArray;
    }

    public static NDArray minimumFloat(NDArray a,float b,NDArray resArray){
        var vB = FloatVector.broadcast(SPECIES, b);

        if(a.isContiguous() && resArray.isContiguous()){
            long i = 0;
            long loopBound = SPECIES.loopBound(a.size);
            
            for (; i < loopBound; i += VL) {
                var vA = FloatVector.fromMemorySegment(SPECIES, a.data, i * FLOAT_BYTES, ORDER);
                var vRes = vA.min(vB);
                vRes.intoMemorySegment(resArray.data, i * FLOAT_BYTES, ORDER);
            }

            for (; i < a.size; i++) {
                float valA = a.data.getAtIndex(ValueLayout.JAVA_FLOAT, i);
                resArray.data.setAtIndex(ValueLayout.JAVA_FLOAT, i, Math.min(valA, b));
            }
        } else {
            var iterA = new NDIter(resArray.shape, a.strides);
            var iterRes = new NDIter(resArray.shape, resArray.strides);

            while(iterA.hasNext){
                float valA = a.data.getAtIndex(ValueLayout.JAVA_FLOAT, iterA.offset);
                resArray.data.setAtIndex(ValueLayout.JAVA_FLOAT, iterRes.offset, Math.min(valA, b));
                
                iterA.next(); iterRes.next();
            }
        }
        return resArray;
    }

    public static NDArray minimumDouble(NDArray a,NDArray b,NDArray resArray){
        if (a.isContiguous() && b.isContiguous() && resArray.isContiguous()) {
            long i = 0;
            long loopBound = SPECIESDB.loopBound(a.size);
            
            for (; i < loopBound; i += DB_VL) {
                var vA = DoubleVector.fromMemorySegment(SPECIESDB, a.data, i * DB_BYTES, ORDER);
                var vB = DoubleVector.fromMemorySegment(SPECIESDB, b.data, i * DB_BYTES, ORDER);
                var vRes = vA.min(vB);
                vRes.intoMemorySegment(resArray.data, i * DB_BYTES, ORDER);
            }

            for (; i < a.size; i++) {
                double valA = a.data.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
                double valB = b.data.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
                resArray.data.setAtIndex(ValueLayout.JAVA_DOUBLE, i, Math.min(valA, valB));
            }
        } else {
            var iterA = new NDIter(resArray.shape, a.strides);
            var iterB = new NDIter(resArray.shape, b.strides);
            var iterRes = new NDIter(resArray.shape, resArray.strides);

            while (iterA.hasNext) {
                double valA = a.data.getAtIndex(ValueLayout.JAVA_DOUBLE, iterA.offset);
                double valB = b.data.getAtIndex(ValueLayout.JAVA_DOUBLE, iterB.offset);
                resArray.data.setAtIndex(ValueLayout.JAVA_DOUBLE, iterRes.offset, Math.min(valA, valB));

                iterA.next(); iterB.next(); iterRes.next();
            }
        }
        return resArray;
    }

    public static NDArray minimumDouble(NDArray a,double b,NDArray resArray){
        var vB = DoubleVector.broadcast(SPECIESDB, b);

        if(a.isContiguous() && resArray.isContiguous()){
            long i = 0;
            long loopBound = SPECIESDB.loopBound(a.size);
            
            for (; i < loopBound; i += DB_VL) {
                var vA = DoubleVector.fromMemorySegment(SPECIESDB, a.data, i * DB_BYTES, ORDER);
                var vRes = vA.min(vB);
                vRes.intoMemorySegment(resArray.data, i * DB_BYTES, ORDER);
            }

            for (; i < a.size; i++) {
                double valA = a.data.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
                resArray.data.setAtIndex(ValueLayout.JAVA_DOUBLE, i, Math.min(valA, b));
            }
        } else {
            var iterA = new NDIter(resArray.shape, a.strides);
            var iterRes = new NDIter(resArray.shape, resArray.strides);

            while(iterA.hasNext){
                double valA = a.data.getAtIndex(ValueLayout.JAVA_DOUBLE, iterA.offset);
                resArray.data.setAtIndex(ValueLayout.JAVA_DOUBLE, iterRes.offset, Math.min(valA, b));
                
                iterA.next(); iterRes.next();
            }
        }
        return resArray;
    }

    public static NDArray minimumInt(NDArray a,NDArray b,NDArray resArray){
        if (a.isContiguous() && b.isContiguous() && resArray.isContiguous()) {
            long i = 0;
            long loopBound = SPECIESINT.loopBound(a.size);
            
            for (; i < loopBound; i += INT_VL) {
                var vA = IntVector.fromMemorySegment(SPECIESINT, a.data, i * INT_BYTES, ORDER);
                var vB = IntVector.fromMemorySegment(SPECIESINT, b.data, i * INT_BYTES, ORDER);
                var vRes = vA.min(vB);
                vRes.intoMemorySegment(resArray.data, i * INT_BYTES, ORDER);
            }

            for (; i < a.size; i++) {
                int valA = a.data.getAtIndex(ValueLayout.JAVA_INT, i);
                int valB = b.data.getAtIndex(ValueLayout.JAVA_INT, i);
                resArray.data.setAtIndex(ValueLayout.JAVA_INT, i, Math.min(valA, valB));
            }
        } else {
            var iterA = new NDIter(resArray.shape, a.strides);
            var iterB = new NDIter(resArray.shape, b.strides);
            var iterRes = new NDIter(resArray.shape, resArray.strides);

            while (iterA.hasNext) {
                int valA = a.data.getAtIndex(ValueLayout.JAVA_INT, iterA.offset);
                int valB = b.data.getAtIndex(ValueLayout.JAVA_INT, iterB.offset);
                resArray.data.setAtIndex(ValueLayout.JAVA_INT, iterRes.offset, Math.min(valA, valB));

                iterA.next(); iterB.next(); iterRes.next();
            }
        }
        return resArray;
    }

    public static NDArray minimumInt(NDArray a,int b,NDArray resArray){
        var vB = IntVector.broadcast(SPECIESINT, b);

        if(a.isContiguous() && resArray.isContiguous()){
            long i = 0;
            long loopBound = SPECIESINT.loopBound(a.size);
            
            for (; i < loopBound; i += INT_VL) {
                var vA = IntVector.fromMemorySegment(SPECIESINT, a.data, i * INT_BYTES, ORDER);
                var vRes = vA.min(vB);
                vRes.intoMemorySegment(resArray.data, i * INT_BYTES, ORDER);
            }

            for (; i < a.size; i++) {
                int valA = a.data.getAtIndex(ValueLayout.JAVA_INT, i);
                resArray.data.setAtIndex(ValueLayout.JAVA_INT, i, Math.min(valA, b));
            }
        } else {
            var iterA = new NDIter(resArray.shape, a.strides);
            var iterRes = new NDIter(resArray.shape, resArray.strides);

            while(iterA.hasNext){
                int valA = a.data.getAtIndex(ValueLayout.JAVA_INT, iterA.offset);
                resArray.data.setAtIndex(ValueLayout.JAVA_INT, iterRes.offset, Math.min(valA, b));
                
                iterA.next(); iterRes.next();
            }
        }
        return resArray;
    }

}
