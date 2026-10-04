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

public final class Dot {

    private Dot(){
        throw new AssertionError();
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
        long loopbound= a.getSize() - (a.getSize() % (VL_F32 * 2));
        var vSum1 = FloatVector.zero(SPECIES_F32);
        var vSum2 = FloatVector.zero(SPECIES_F32);

        for(;i<loopbound;i+=VL_F32*2){
            var va1=FloatVector.fromMemorySegment(SPECIES_F32, a.getData(), i*BYTES_F32, NATIVE_ORDER);
            var vb1=FloatVector.fromMemorySegment(SPECIES_F32, b.getData(), i*BYTES_F32, NATIVE_ORDER);
            var va2=FloatVector.fromMemorySegment(SPECIES_F32, a.getData(), (i+VL_F32)*BYTES_F32, NATIVE_ORDER);
            var vb2=FloatVector.fromMemorySegment(SPECIES_F32, b.getData(), (i+VL_F32)*BYTES_F32, NATIVE_ORDER);
            vSum1=va1.fma(vb1,vSum1);
            vSum2=va2.fma(vb2, vSum2);
        }

        loopbound=SPECIES_F32.loopBound(a.getSize());

        for(;i<loopbound;i+=VL_F32){
            var v1=FloatVector.fromMemorySegment(SPECIES_F32, a.getData(), i*BYTES_F32, NATIVE_ORDER);
            var v2=FloatVector.fromMemorySegment(SPECIES_F32, b.getData(), i*BYTES_F32, NATIVE_ORDER);
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
        long loopbound= a.getSize() - (a.getSize() % (VL_I32 * 2));
        var vSum1 = IntVector.zero(SPECIES_I32);
        var vSum2 = IntVector.zero(SPECIES_I32);

        for(;i<loopbound;i+=VL_I32*2){
            var va1=IntVector.fromMemorySegment(SPECIES_I32, a.getData(), i*BYTES_I32, NATIVE_ORDER);
            var vb1=IntVector.fromMemorySegment(SPECIES_I32, b.getData(), i*BYTES_I32, NATIVE_ORDER);
            var va2=IntVector.fromMemorySegment(SPECIES_I32, a.getData(), (i+VL_I32)*BYTES_I32, NATIVE_ORDER);
            var vb2=IntVector.fromMemorySegment(SPECIES_I32, b.getData(), (i+VL_I32)*BYTES_I32, NATIVE_ORDER);
            vSum1 = va1.mul(vb1).add(vSum1);
            vSum2 = va2.mul(vb2).add(vSum2);
        }

        loopbound=SPECIES_F32.loopBound(a.getSize());

        for(;i<loopbound;i+=VL_I32){
            var v1=IntVector.fromMemorySegment(SPECIES_I32, a.getData(), i*BYTES_I32, NATIVE_ORDER);
            var v2=IntVector.fromMemorySegment(SPECIES_I32, b.getData(), i*BYTES_I32, NATIVE_ORDER);
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
        long loopbound= a.getSize() - (a.getSize() % (VL_F64 * 2));
        var vSum1 = DoubleVector.zero(SPECIES_F64);
        var vSum2 = DoubleVector.zero(SPECIES_F64);

        for(;i<loopbound;i+=VL_F64*2){
            var va1=DoubleVector.fromMemorySegment(SPECIES_F64, a.getData(), i*BYTES_F64, NATIVE_ORDER);
            var vb1=DoubleVector.fromMemorySegment(SPECIES_F64, b.getData(), i*BYTES_F64, NATIVE_ORDER);
            var va2=DoubleVector.fromMemorySegment(SPECIES_F64, a.getData(), (i+VL_F64)*BYTES_F64, NATIVE_ORDER);
            var vb2=DoubleVector.fromMemorySegment(SPECIES_F64, b.getData(), (i+VL_F64)*BYTES_F64, NATIVE_ORDER);
            vSum1=va1.fma(vb1, vSum1);
            vSum2=va2.fma(vb2, vSum2);
        }

        loopbound=SPECIES_F32.loopBound(a.getSize());

        for(;i<loopbound;i+=VL_F64){
            var v1=DoubleVector.fromMemorySegment(SPECIES_F64, a.getData(), i*BYTES_F64, NATIVE_ORDER);
            var v2=DoubleVector.fromMemorySegment(SPECIES_F64, b.getData(), i*BYTES_F64, NATIVE_ORDER);
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
