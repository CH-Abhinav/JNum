package jnum.internal.kernel.compare;

import static jnum.internal.Constants.*;

import jnum.NDArray;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.DoubleVector;
import jnum.internal.layout.NDIter;

import java.lang.foreign.ValueLayout;

public final class Maximum {

    private Maximum() {
        throw new AssertionError();
    }

    public static NDArray maximumFloat(NDArray a,NDArray b,NDArray resArray){
        if (a.isContiguous() && b.isContiguous() && resArray.isContiguous()) {
            long i = 0;
            long loopBound = SPECIES_F32.loopBound(a.getSize());
            
            for (; i < loopBound; i += VL_F32) {
                var vA = FloatVector.fromMemorySegment(SPECIES_F32, a.getData(), i * BYTES_F32, NATIVE_ORDER);
                var vB = FloatVector.fromMemorySegment(SPECIES_F32, b.getData(), i * BYTES_F32, NATIVE_ORDER);
                var vRes = vA.max(vB);
                vRes.intoMemorySegment(resArray.getData(), i * BYTES_F32, NATIVE_ORDER);
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
        var vB=FloatVector.broadcast(SPECIES_F32, b);

        if(a.isContiguous() && resArray.isContiguous()){
            long i = 0;
            long loopBound = SPECIES_F32.loopBound(a.getSize());
            
            for (; i < loopBound; i += VL_F32) {
                var vA = FloatVector.fromMemorySegment(SPECIES_F32, a.getData(), i * BYTES_F32, NATIVE_ORDER);
                var vRes = vA.max(vB);
                vRes.intoMemorySegment(resArray.getData(), i * BYTES_F32, NATIVE_ORDER);
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
            long loopBound = SPECIES_F64.loopBound(a.getSize());
            
            for (; i < loopBound; i += VL_F64) {
                var vA = DoubleVector.fromMemorySegment(SPECIES_F64, a.getData(), i * BYTES_F64, NATIVE_ORDER);
                var vB = DoubleVector.fromMemorySegment(SPECIES_F64, b.getData(), i * BYTES_F64, NATIVE_ORDER);
                var vRes = vA.max(vB);
                vRes.intoMemorySegment(resArray.getData(), i * BYTES_F64, NATIVE_ORDER);
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
        var vB = DoubleVector.broadcast(SPECIES_F64, b);

        if(a.isContiguous() && resArray.isContiguous()){
            long i = 0;
            long loopBound = SPECIES_F64.loopBound(a.getSize());
            
            for (; i < loopBound; i += VL_F64) {
                var vA = DoubleVector.fromMemorySegment(SPECIES_F64, a.getData(), i * BYTES_F64, NATIVE_ORDER);
                var vRes = vA.max(vB);
                vRes.intoMemorySegment(resArray.getData(), i * BYTES_F64, NATIVE_ORDER);
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
            long loopBound = SPECIES_I32.loopBound(a.getSize());

            for (; i < loopBound; i += VL_I32) {
                var vA = IntVector.fromMemorySegment(SPECIES_I32, a.getData(), i * BYTES_I32, NATIVE_ORDER);
                var vB = IntVector.fromMemorySegment(SPECIES_I32, b.getData(), i * BYTES_I32, NATIVE_ORDER);
                var vRes = vA.max(vB);
                vRes.intoMemorySegment(resArray.getData(), i * BYTES_I32, NATIVE_ORDER);
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
        var vB = IntVector.broadcast(SPECIES_I32, b);

        if(a.isContiguous() && resArray.isContiguous()){
            long i = 0;
            long loopBound = SPECIES_I32.loopBound(a.getSize());
            
            for (; i < loopBound; i += VL_I32) {
                var vA = IntVector.fromMemorySegment(SPECIES_I32, a.getData(), i * BYTES_I32, NATIVE_ORDER);
                var vRes = vA.max(vB);
                vRes.intoMemorySegment(resArray.getData(), i * BYTES_I32, NATIVE_ORDER);
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
}
