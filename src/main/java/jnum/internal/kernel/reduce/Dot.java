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

public final class Dot {
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
