package jnum.internal.kernel.trig;

import static jnum.internal.Constants.*;

import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;
import jnum.NDArray;
import jdk.incubator.vector.VectorOperators;

public final class Cosh {

    private Cosh(){
        throw new AssertionError();
    }

    public static NDArray coshFloat(NDArray a,NDArray resArray){
        long i=0;
        long loopbound= a.getSize() - (a.getSize() % (VL_F32 * 2));

        for(;i<loopbound;i+=VL_F32*2){
            var v1=FloatVector.fromMemorySegment(SPECIES_F32, a.getData(),i*BYTES_F32,NATIVE_ORDER);
            var v2=FloatVector.fromMemorySegment(SPECIES_F32, a.getData(),(i+VL_F32)*BYTES_F32,NATIVE_ORDER);
            var VRes1=v1.lanewise(VectorOperators.COSH);
            var VRes2=v2.lanewise(VectorOperators.COSH);
            VRes1.intoMemorySegment(resArray.getData(), i*BYTES_F32, NATIVE_ORDER);
            VRes2.intoMemorySegment(resArray.getData(), (i+VL_F32)*BYTES_F32, NATIVE_ORDER);
        }

        loopbound=SPECIES_F32.loopBound(a.getSize());

        for(;i<loopbound;i+=VL_F32){
            var v=FloatVector.fromMemorySegment(SPECIES_F32, a.getData(),i*BYTES_F32,NATIVE_ORDER);
            var VRes=v.lanewise(VectorOperators.COSH);
            VRes.intoMemorySegment(resArray.getData(), i*BYTES_F32, NATIVE_ORDER);
        }

        for(; i< a.getSize(); i++){
            float val= a.getData().get(ValueLayout.JAVA_FLOAT,i*BYTES_F32);
            resArray.getData().set(ValueLayout.JAVA_FLOAT,i*BYTES_F32,(float) Math.cosh(val));
        }
        return resArray;
    }

    public static NDArray coshDouble(NDArray a,NDArray resArray){
        long i=0;
        long loopbound= a.getSize() - (a.getSize() % (VL_F64 * 2));

        for(;i<loopbound;i+=VL_F64*2){
            var v1=DoubleVector.fromMemorySegment(SPECIES_F64, a.getData(),i*BYTES_F64,NATIVE_ORDER);
            var v2=DoubleVector.fromMemorySegment(SPECIES_F64, a.getData(),(i+VL_F64)*BYTES_F64,NATIVE_ORDER);
            var VRes1=v1.lanewise(VectorOperators.COSH);
            var VRes2=v2.lanewise(VectorOperators.COSH);
            VRes1.intoMemorySegment(resArray.getData(),i*BYTES_F64,NATIVE_ORDER);
            VRes2.intoMemorySegment(resArray.getData(),(i+VL_F64)*BYTES_F64,NATIVE_ORDER);
        }

        loopbound=SPECIES_F64.loopBound(a.getSize());

        for(;i<loopbound;i+=VL_F64){
            var v=DoubleVector.fromMemorySegment(SPECIES_F64, a.getData(),i*BYTES_F64,NATIVE_ORDER);
            var VRes=v.lanewise(VectorOperators.COSH);
            VRes.intoMemorySegment(resArray.getData(),i*BYTES_F64,NATIVE_ORDER);
        }

        for(; i< a.getSize(); i++){
            double val = a.getData().get(ValueLayout.JAVA_DOUBLE, i * BYTES_F64);
            resArray.getData().set(ValueLayout.JAVA_DOUBLE, i * BYTES_F64, Math.cosh(val));
        }

        return resArray;
    }

    public static NDArray coshInt(NDArray a,NDArray resArray){
        long i=0;
        long loopbound= a.getSize() - (a.getSize() % (VL_I32 * 2));

        for(;i<loopbound;i+=VL_I32*2){
            var vInt1=IntVector.fromMemorySegment(SPECIES_I32, a.getData(), i*BYTES_I32, NATIVE_ORDER);
            var vInt2=IntVector.fromMemorySegment(SPECIES_I32, a.getData(), (i+VL_I32)*BYTES_I32, NATIVE_ORDER);
            var vFloat1=vInt1.convert(VectorOperators.I2F, 0);
            var vFloat2=vInt2.convert(VectorOperators.I2F, 0);
            var VRes1=vFloat1.lanewise(VectorOperators.COSH);
            var VRes2=vFloat2.lanewise(VectorOperators.COSH);
            VRes1.intoMemorySegment(resArray.getData(), i*BYTES_F32, NATIVE_ORDER);
            VRes2.intoMemorySegment(resArray.getData(), (i+VL_I32)*BYTES_F32, NATIVE_ORDER);
        }

        loopbound=SPECIES_I32.loopBound(a.getSize());

        for(;i<loopbound;i+=VL_I32){
            var vInt=IntVector.fromMemorySegment(SPECIES_I32, a.getData(), i*BYTES_I32, NATIVE_ORDER);
            var vFloat=vInt.convert(VectorOperators.I2F, 0);
            var VRes=vFloat.lanewise(VectorOperators.COSH);
            VRes.intoMemorySegment(resArray.getData(), i*BYTES_F32, NATIVE_ORDER);
        }

        for(; i< a.getSize(); i++){
            float val= a.getData().get(ValueLayout.JAVA_INT, i*BYTES_I32);
            resArray.getData().set(ValueLayout.JAVA_FLOAT,i*BYTES_F32,(float) Math.cosh(val));
        }
        return resArray;
    }
}
