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
            long loopBound = SPECIES.loopBound(a.getSize());
            
            for (; i < loopBound; i += VL) {
                var vA = FloatVector.fromMemorySegment(SPECIES, a.getData(), i * FLOAT_BYTES, ORDER);
                var vB = FloatVector.fromMemorySegment(SPECIES, b.getData(), i * FLOAT_BYTES, ORDER);
                var vRes = vA.max(vB);
                vRes.intoMemorySegment(resArray.getData(), i * FLOAT_BYTES, ORDER);
            }

            for (; i < a.getSize(); i++) {
                float valA = a.getData().getAtIndex(ValueLayout.JAVA_FLOAT, i);
                float valB = b.getData().getAtIndex(ValueLayout.JAVA_FLOAT, i);
                resArray.getData().setAtIndex(ValueLayout.JAVA_FLOAT, i, Math.max(valA, valB));
            }
    }else {
            var iterA = new NDIter(resArray.internalShapeUnsafe(), a.internalStridesUnsafe());
            var iterB = new NDIter(resArray.internalShapeUnsafe(), b.internalStridesUnsafe());
            var iterRes = new NDIter(resArray.internalShapeUnsafe(), resArray.internalStridesUnsafe());

            while (iterA.hasNext) {
                float valA = a.getData().getAtIndex(ValueLayout.JAVA_FLOAT, iterA.offset);
                float valB = b.getData().getAtIndex(ValueLayout.JAVA_FLOAT, iterB.offset);
                resArray.getData().setAtIndex(ValueLayout.JAVA_FLOAT, iterRes.offset, Math.max(valA, valB));
                
                iterA.next(); iterB.next(); iterRes.next();
            }
        }
        return resArray;
    }

    public static NDArray maximumFloat(NDArray a,float b,NDArray resArray){
        var vB=FloatVector.broadcast(SPECIES, b);

        if(a.isContiguous() && resArray.isContiguous()){
            long i = 0;
            long loopBound = SPECIES.loopBound(a.getSize());
            
            for (; i < loopBound; i += VL) {
                var vA = FloatVector.fromMemorySegment(SPECIES, a.getData(), i * FLOAT_BYTES, ORDER);
                var vRes = vA.max(vB);
                vRes.intoMemorySegment(resArray.getData(), i * FLOAT_BYTES, ORDER);
            }

            for (; i < a.getSize(); i++) {
                float valA = a.getData().getAtIndex(ValueLayout.JAVA_FLOAT, i);
                resArray.getData().setAtIndex(ValueLayout.JAVA_FLOAT, i, Math.max(valA, b));
            }
        }else{
            var iterA=new NDIter(resArray.internalShapeUnsafe(), a.internalStridesUnsafe());
            var iterRes=new NDIter(resArray.internalShapeUnsafe(), resArray.internalStridesUnsafe());

            while(iterA.hasNext){
                float valA= a.getData().getAtIndex(ValueLayout.JAVA_FLOAT, iterA.offset);
                resArray.getData().setAtIndex(ValueLayout.JAVA_FLOAT, iterRes.offset, Math.max(valA,b));
                
                iterA.next(); iterRes.next();
            }
        }
        return resArray;
    }

    public static NDArray maximumDouble(NDArray a,NDArray b,NDArray resArray){
        if (a.isContiguous() && b.isContiguous() && resArray.isContiguous()) {
            long i = 0;
            long loopBound = SPECIESDB.loopBound(a.getSize());
            
            for (; i < loopBound; i += DB_VL) {
                var vA = DoubleVector.fromMemorySegment(SPECIESDB, a.getData(), i * DB_BYTES, ORDER);
                var vB = DoubleVector.fromMemorySegment(SPECIESDB, b.getData(), i * DB_BYTES, ORDER);
                var vRes = vA.max(vB);
                vRes.intoMemorySegment(resArray.getData(), i * DB_BYTES, ORDER);
            }

            for (; i < a.getSize(); i++) {
                double valA = a.getData().getAtIndex(ValueLayout.JAVA_DOUBLE, i);
                double valB = b.getData().getAtIndex(ValueLayout.JAVA_DOUBLE, i);
                resArray.getData().setAtIndex(ValueLayout.JAVA_DOUBLE, i, Math.max(valA, valB));
            }
        } else {
            var iterA = new NDIter(resArray.internalShapeUnsafe(), a.internalStridesUnsafe());
            var iterB = new NDIter(resArray.internalShapeUnsafe(), b.internalStridesUnsafe());
            var iterRes = new NDIter(resArray.internalShapeUnsafe(), resArray.internalStridesUnsafe());

            while (iterA.hasNext) {
                double valA = a.getData().getAtIndex(ValueLayout.JAVA_DOUBLE, iterA.offset);
                double valB = b.getData().getAtIndex(ValueLayout.JAVA_DOUBLE, iterB.offset);
                resArray.getData().setAtIndex(ValueLayout.JAVA_DOUBLE, iterRes.offset, Math.max(valA, valB));

                iterA.next(); iterB.next(); iterRes.next();
            }
        }
        return resArray;
    }

    public static NDArray maximumDouble(NDArray a,double b,NDArray resArray){
        var vB = DoubleVector.broadcast(SPECIESDB, b);

        if(a.isContiguous() && resArray.isContiguous()){
            long i = 0;
            long loopBound = SPECIESDB.loopBound(a.getSize());
            
            for (; i < loopBound; i += DB_VL) {
                var vA = DoubleVector.fromMemorySegment(SPECIESDB, a.getData(), i * DB_BYTES, ORDER);
                var vRes = vA.max(vB);
                vRes.intoMemorySegment(resArray.getData(), i * DB_BYTES, ORDER);
            }

            for (; i < a.getSize(); i++) {
                double valA = a.getData().getAtIndex(ValueLayout.JAVA_DOUBLE, i);
                resArray.getData().setAtIndex(ValueLayout.JAVA_DOUBLE, i, Math.max(valA, b));
            }
        } else {
            var iterA = new NDIter(resArray.internalShapeUnsafe(), a.internalStridesUnsafe());
            var iterRes = new NDIter(resArray.internalShapeUnsafe(), resArray.internalStridesUnsafe());

            while(iterA.hasNext){
                double valA = a.getData().getAtIndex(ValueLayout.JAVA_DOUBLE, iterA.offset);
                resArray.getData().setAtIndex(ValueLayout.JAVA_DOUBLE, iterRes.offset, Math.max(valA, b));
                
                iterA.next(); iterRes.next();
            }
        }
        return resArray;
    }

    public static NDArray maximumInt(NDArray a,NDArray b,NDArray resArray){
        if (a.isContiguous() && b.isContiguous() && resArray.isContiguous()) {
            long i = 0;
            long loopBound = SPECIESINT.loopBound(a.getSize());

            for (; i < loopBound; i += INT_VL) {
                var vA = IntVector.fromMemorySegment(SPECIESINT, a.getData(), i * INT_BYTES, ORDER);
                var vB = IntVector.fromMemorySegment(SPECIESINT, b.getData(), i * INT_BYTES, ORDER);
                var vRes = vA.max(vB);
                vRes.intoMemorySegment(resArray.getData(), i * INT_BYTES, ORDER);
            }

            for (; i < a.getSize(); i++) {
                int valA = a.getData().getAtIndex(ValueLayout.JAVA_INT, i);
                int valB = b.getData().getAtIndex(ValueLayout.JAVA_INT, i);
                resArray.getData().setAtIndex(ValueLayout.JAVA_INT, i, Math.max(valA, valB));
            }
        } else {
            var iterA = new NDIter(resArray.internalShapeUnsafe(), a.internalStridesUnsafe());
            var iterB = new NDIter(resArray.internalShapeUnsafe(), b.internalStridesUnsafe());
            var iterRes = new NDIter(resArray.internalShapeUnsafe(), resArray.internalStridesUnsafe());

            while (iterA.hasNext) {
                int valA = a.getData().getAtIndex(ValueLayout.JAVA_INT, iterA.offset);
                int valB = b.getData().getAtIndex(ValueLayout.JAVA_INT, iterB.offset);
                resArray.getData().setAtIndex(ValueLayout.JAVA_INT, iterRes.offset, Math.max(valA, valB));

                iterA.next(); iterB.next(); iterRes.next();
            }
        }
        return resArray;
    }

    public static NDArray maximumInt(NDArray a,int b,NDArray resArray){
        var vB = IntVector.broadcast(SPECIESINT, b);

        if(a.isContiguous() && resArray.isContiguous()){
            long i = 0;
            long loopBound = SPECIESINT.loopBound(a.getSize());
            
            for (; i < loopBound; i += INT_VL) {
                var vA = IntVector.fromMemorySegment(SPECIESINT, a.getData(), i * INT_BYTES, ORDER);
                var vRes = vA.max(vB);
                vRes.intoMemorySegment(resArray.getData(), i * INT_BYTES, ORDER);
            }

            for (; i < a.getSize(); i++) {
                int valA = a.getData().getAtIndex(ValueLayout.JAVA_INT, i);
                resArray.getData().setAtIndex(ValueLayout.JAVA_INT, i, Math.max(valA, b));
            }
        } else {
            var iterA = new NDIter(resArray.internalShapeUnsafe(), a.internalStridesUnsafe());
            var iterRes = new NDIter(resArray.internalShapeUnsafe(), resArray.internalStridesUnsafe());

            while(iterA.hasNext){
                int valA = a.getData().getAtIndex(ValueLayout.JAVA_INT, iterA.offset);
                resArray.getData().setAtIndex(ValueLayout.JAVA_INT, iterRes.offset, Math.max(valA, b));
                
                iterA.next(); iterRes.next();
            }
        }
        return resArray;
    }

    public static NDArray minimumFloat(NDArray a,NDArray b,NDArray resArray){
        if (a.isContiguous() && b.isContiguous() && resArray.isContiguous()) {
            long i = 0;
            long loopBound = SPECIES.loopBound(a.getSize());
            
            for (; i < loopBound; i += VL) {
                var vA = FloatVector.fromMemorySegment(SPECIES, a.getData(), i * FLOAT_BYTES, ORDER);
                var vB = FloatVector.fromMemorySegment(SPECIES, b.getData(), i * FLOAT_BYTES, ORDER);
                var vRes = vA.min(vB);
                vRes.intoMemorySegment(resArray.getData(), i * FLOAT_BYTES, ORDER);
            }

            for (; i < a.getSize(); i++) {
                float valA = a.getData().getAtIndex(ValueLayout.JAVA_FLOAT, i);
                float valB = b.getData().getAtIndex(ValueLayout.JAVA_FLOAT, i);
                resArray.getData().setAtIndex(ValueLayout.JAVA_FLOAT, i, Math.min(valA, valB));
            }
        } else {
            var iterA = new NDIter(resArray.internalShapeUnsafe(), a.internalStridesUnsafe());
            var iterB = new NDIter(resArray.internalShapeUnsafe(), b.internalStridesUnsafe());
            var iterRes = new NDIter(resArray.internalShapeUnsafe(), resArray.internalStridesUnsafe());

            while (iterA.hasNext) {
                float valA = a.getData().getAtIndex(ValueLayout.JAVA_FLOAT, iterA.offset);
                float valB = b.getData().getAtIndex(ValueLayout.JAVA_FLOAT, iterB.offset);
                resArray.getData().setAtIndex(ValueLayout.JAVA_FLOAT, iterRes.offset, Math.min(valA, valB));
                
                iterA.next(); iterB.next(); iterRes.next();
            }
        }
        return resArray;
    }

    public static NDArray minimumFloat(NDArray a,float b,NDArray resArray){
        var vB = FloatVector.broadcast(SPECIES, b);

        if(a.isContiguous() && resArray.isContiguous()){
            long i = 0;
            long loopBound = SPECIES.loopBound(a.getSize());
            
            for (; i < loopBound; i += VL) {
                var vA = FloatVector.fromMemorySegment(SPECIES, a.getData(), i * FLOAT_BYTES, ORDER);
                var vRes = vA.min(vB);
                vRes.intoMemorySegment(resArray.getData(), i * FLOAT_BYTES, ORDER);
            }

            for (; i < a.getSize(); i++) {
                float valA = a.getData().getAtIndex(ValueLayout.JAVA_FLOAT, i);
                resArray.getData().setAtIndex(ValueLayout.JAVA_FLOAT, i, Math.min(valA, b));
            }
        } else {
            var iterA = new NDIter(resArray.internalShapeUnsafe(), a.internalStridesUnsafe());
            var iterRes = new NDIter(resArray.internalShapeUnsafe(), resArray.internalStridesUnsafe());

            while(iterA.hasNext){
                float valA = a.getData().getAtIndex(ValueLayout.JAVA_FLOAT, iterA.offset);
                resArray.getData().setAtIndex(ValueLayout.JAVA_FLOAT, iterRes.offset, Math.min(valA, b));
                
                iterA.next(); iterRes.next();
            }
        }
        return resArray;
    }

    public static NDArray minimumDouble(NDArray a,NDArray b,NDArray resArray){
        if (a.isContiguous() && b.isContiguous() && resArray.isContiguous()) {
            long i = 0;
            long loopBound = SPECIESDB.loopBound(a.getSize());
            
            for (; i < loopBound; i += DB_VL) {
                var vA = DoubleVector.fromMemorySegment(SPECIESDB, a.getData(), i * DB_BYTES, ORDER);
                var vB = DoubleVector.fromMemorySegment(SPECIESDB, b.getData(), i * DB_BYTES, ORDER);
                var vRes = vA.min(vB);
                vRes.intoMemorySegment(resArray.getData(), i * DB_BYTES, ORDER);
            }

            for (; i < a.getSize(); i++) {
                double valA = a.getData().getAtIndex(ValueLayout.JAVA_DOUBLE, i);
                double valB = b.getData().getAtIndex(ValueLayout.JAVA_DOUBLE, i);
                resArray.getData().setAtIndex(ValueLayout.JAVA_DOUBLE, i, Math.min(valA, valB));
            }
        } else {
            var iterA = new NDIter(resArray.internalShapeUnsafe(), a.internalStridesUnsafe());
            var iterB = new NDIter(resArray.internalShapeUnsafe(), b.internalStridesUnsafe());
            var iterRes = new NDIter(resArray.internalShapeUnsafe(), resArray.internalStridesUnsafe());

            while (iterA.hasNext) {
                double valA = a.getData().getAtIndex(ValueLayout.JAVA_DOUBLE, iterA.offset);
                double valB = b.getData().getAtIndex(ValueLayout.JAVA_DOUBLE, iterB.offset);
                resArray.getData().setAtIndex(ValueLayout.JAVA_DOUBLE, iterRes.offset, Math.min(valA, valB));

                iterA.next(); iterB.next(); iterRes.next();
            }
        }
        return resArray;
    }

    public static NDArray minimumDouble(NDArray a,double b,NDArray resArray){
        var vB = DoubleVector.broadcast(SPECIESDB, b);

        if(a.isContiguous() && resArray.isContiguous()){
            long i = 0;
            long loopBound = SPECIESDB.loopBound(a.getSize());
            
            for (; i < loopBound; i += DB_VL) {
                var vA = DoubleVector.fromMemorySegment(SPECIESDB, a.getData(), i * DB_BYTES, ORDER);
                var vRes = vA.min(vB);
                vRes.intoMemorySegment(resArray.getData(), i * DB_BYTES, ORDER);
            }

            for (; i < a.getSize(); i++) {
                double valA = a.getData().getAtIndex(ValueLayout.JAVA_DOUBLE, i);
                resArray.getData().setAtIndex(ValueLayout.JAVA_DOUBLE, i, Math.min(valA, b));
            }
        } else {
            var iterA = new NDIter(resArray.internalShapeUnsafe(), a.internalStridesUnsafe());
            var iterRes = new NDIter(resArray.internalShapeUnsafe(), resArray.internalStridesUnsafe());

            while(iterA.hasNext){
                double valA = a.getData().getAtIndex(ValueLayout.JAVA_DOUBLE, iterA.offset);
                resArray.getData().setAtIndex(ValueLayout.JAVA_DOUBLE, iterRes.offset, Math.min(valA, b));
                
                iterA.next(); iterRes.next();
            }
        }
        return resArray;
    }

    public static NDArray minimumInt(NDArray a,NDArray b,NDArray resArray){
        if (a.isContiguous() && b.isContiguous() && resArray.isContiguous()) {
            long i = 0;
            long loopBound = SPECIESINT.loopBound(a.getSize());
            
            for (; i < loopBound; i += INT_VL) {
                var vA = IntVector.fromMemorySegment(SPECIESINT, a.getData(), i * INT_BYTES, ORDER);
                var vB = IntVector.fromMemorySegment(SPECIESINT, b.getData(), i * INT_BYTES, ORDER);
                var vRes = vA.min(vB);
                vRes.intoMemorySegment(resArray.getData(), i * INT_BYTES, ORDER);
            }

            for (; i < a.getSize(); i++) {
                int valA = a.getData().getAtIndex(ValueLayout.JAVA_INT, i);
                int valB = b.getData().getAtIndex(ValueLayout.JAVA_INT, i);
                resArray.getData().setAtIndex(ValueLayout.JAVA_INT, i, Math.min(valA, valB));
            }
        } else {
            var iterA = new NDIter(resArray.internalShapeUnsafe(), a.internalStridesUnsafe());
            var iterB = new NDIter(resArray.internalShapeUnsafe(), b.internalStridesUnsafe());
            var iterRes = new NDIter(resArray.internalShapeUnsafe(), resArray.internalStridesUnsafe());

            while (iterA.hasNext) {
                int valA = a.getData().getAtIndex(ValueLayout.JAVA_INT, iterA.offset);
                int valB = b.getData().getAtIndex(ValueLayout.JAVA_INT, iterB.offset);
                resArray.getData().setAtIndex(ValueLayout.JAVA_INT, iterRes.offset, Math.min(valA, valB));

                iterA.next(); iterB.next(); iterRes.next();
            }
        }
        return resArray;
    }

    public static NDArray minimumInt(NDArray a,int b,NDArray resArray){
        var vB = IntVector.broadcast(SPECIESINT, b);

        if(a.isContiguous() && resArray.isContiguous()){
            long i = 0;
            long loopBound = SPECIESINT.loopBound(a.getSize());
            
            for (; i < loopBound; i += INT_VL) {
                var vA = IntVector.fromMemorySegment(SPECIESINT, a.getData(), i * INT_BYTES, ORDER);
                var vRes = vA.min(vB);
                vRes.intoMemorySegment(resArray.getData(), i * INT_BYTES, ORDER);
            }

            for (; i < a.getSize(); i++) {
                int valA = a.getData().getAtIndex(ValueLayout.JAVA_INT, i);
                resArray.getData().setAtIndex(ValueLayout.JAVA_INT, i, Math.min(valA, b));
            }
        } else {
            var iterA = new NDIter(resArray.internalShapeUnsafe(), a.internalStridesUnsafe());
            var iterRes = new NDIter(resArray.internalShapeUnsafe(), resArray.internalStridesUnsafe());

            while(iterA.hasNext){
                int valA = a.getData().getAtIndex(ValueLayout.JAVA_INT, iterA.offset);
                resArray.getData().setAtIndex(ValueLayout.JAVA_INT, iterRes.offset, Math.min(valA, b));
                
                iterA.next(); iterRes.next();
            }
        }
        return resArray;
    }

}
