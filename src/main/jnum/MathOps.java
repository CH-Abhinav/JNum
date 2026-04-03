package jnum;

import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;
import jdk.incubator.vector.VectorOperators;
public class MathOps {
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
    private MathOps(){
        throw new AssertionError();
    }

    //SQRT methods 

    public static NDArray sqrtFloat(NDArray a,NDArray resArray){
        long i=0;
        long loopbound=SPECIES.loopBound(a.size);

        for(;i<loopbound;i+=VL){
            var v=FloatVector.fromMemorySegment(SPECIES,a.data,i*FLOAT_BYTES,ORDER);
            var VRes=v.lanewise(VectorOperators.SQRT);
            VRes.intoMemorySegment(resArray.data,i*FLOAT_BYTES,ORDER);
        }

        for(;i<a.size;i++){
            float val = a.data.get(ValueLayout.JAVA_FLOAT, i * FLOAT_BYTES);
            resArray.data.set(ValueLayout.JAVA_FLOAT, i * FLOAT_BYTES, (float) Math.sqrt(val));
        }

        return resArray;
    }

    public static NDArray sqrtDouble(NDArray a,NDArray resArray){
        long i=0;
        long loopbound=SPECIESDB.loopBound(a.size);

        for(;i<loopbound;i+=DB_VL){
            var v=DoubleVector.fromMemorySegment(SPECIESDB,a.data,i*DB_BYTES,ORDER);
            var VRes=v.lanewise(VectorOperators.SQRT);
            VRes.intoMemorySegment(resArray.data,i*DB_BYTES,ORDER);
        }

        for(;i<a.size;i++){
            double val = a.data.get(ValueLayout.JAVA_DOUBLE, i * DB_BYTES);
            resArray.data.set(ValueLayout.JAVA_DOUBLE, i * DB_BYTES, Math.sqrt(val));
        }

        return resArray;
    }

    public static NDArray sqrtInt(NDArray a,NDArray resArray){
        long i=0;
        long loopbound=SPECIESINT.loopBound(a.size);

        for(;i<loopbound;i+=INT_VL){
            var vInt=IntVector.fromMemorySegment(SPECIESINT, a.data, i*INT_BYTES, ORDER);
            var vFloat=vInt.convert(VectorOperators.I2F, 0);
            var VRes=vFloat.lanewise(VectorOperators.SQRT);
            VRes.intoMemorySegment(resArray.data, i*FLOAT_BYTES, ORDER);
        }

        for(;i<a.size;i++){
            float val=a.data.get(ValueLayout.JAVA_INT, i*INT_BYTES);
            resArray.data.set(ValueLayout.JAVA_FLOAT,i*FLOAT_BYTES,(float) Math.sqrt(val));
        }
        return resArray;
    }

    //ABS methods

    public static NDArray absFloat(NDArray a,NDArray resArray){
        long i=0;
        long loopbound=SPECIES.loopBound(a.size);

        for(;i<loopbound;i+=VL){
            var v=FloatVector.fromMemorySegment(SPECIES,a.data,i*FLOAT_BYTES,ORDER);
            var VRes=v.lanewise(VectorOperators.ABS);
            VRes.intoMemorySegment(resArray.data, i*FLOAT_BYTES, ORDER);
        }

        for(;i<a.size;i++){
            float val=a.data.get(ValueLayout.JAVA_FLOAT,i*FLOAT_BYTES);
            resArray.data.set(ValueLayout.JAVA_FLOAT,i*FLOAT_BYTES,Math.abs(val));
        }
        return resArray;
    }

    public static NDArray absDouble(NDArray a,NDArray resArray){
        long i=0;
        long loopbound=SPECIESDB.loopBound(a.size);

        for(;i<loopbound;i+=DB_VL){
            var v=DoubleVector.fromMemorySegment(SPECIESDB,a.data,i*DB_BYTES,ORDER);
            var VRes=v.lanewise(VectorOperators.ABS);
            VRes.intoMemorySegment(resArray.data, i*DB_BYTES, ORDER);
        }

        for(;i<a.size;i++){
            double val=a.data.get(ValueLayout.JAVA_DOUBLE,i*DB_BYTES);
            resArray.data.set(ValueLayout.JAVA_DOUBLE,i*DB_BYTES,Math.abs(val));
        }
        return resArray;
    }

    public static NDArray absInt(NDArray a,NDArray resArray){
        long i=0;
        long loopbound=SPECIESINT.loopBound(a.size);

        for(;i<loopbound;i+=INT_VL){
            var v=IntVector.fromMemorySegment(SPECIESINT,a.data,i*INT_BYTES,ORDER);
            var VRes=v.lanewise(VectorOperators.ABS);
            VRes.intoMemorySegment(resArray.data, i*INT_BYTES, ORDER);
        }

        for(;i<a.size;i++){
            int val=a.data.get(ValueLayout.JAVA_INT,i*INT_BYTES);
            resArray.data.set(ValueLayout.JAVA_INT,i*INT_BYTES,Math.abs(val));
        }
        return resArray;
    }

    //EXP methods

    public static NDArray expFloat(NDArray a,NDArray resArray){
        long i=0;
        long loopbound=SPECIES.loopBound(a.size);

        for(;i<loopbound;i+=VL){
            var v=FloatVector.fromMemorySegment(SPECIES,a.data,i*FLOAT_BYTES,ORDER);
            var VRes=v.lanewise(VectorOperators.EXP);
            VRes.intoMemorySegment(resArray.data, i*FLOAT_BYTES, ORDER);
        }

        for(;i<a.size;i++){
            float val=a.data.get(ValueLayout.JAVA_FLOAT,i*FLOAT_BYTES);
            resArray.data.set(ValueLayout.JAVA_FLOAT,i*FLOAT_BYTES,(float) Math.exp(val));
        }
        return resArray;
    }

    public static NDArray expDouble(NDArray a,NDArray resArray){
        long i=0;
        long loopbound=SPECIESDB.loopBound(a.size);

        for(;i<loopbound;i+=DB_VL){
            var v=DoubleVector.fromMemorySegment(SPECIESDB,a.data,i*DB_BYTES,ORDER);
            var VRes=v.lanewise(VectorOperators.EXP);
            VRes.intoMemorySegment(resArray.data,i*DB_BYTES,ORDER);
        }

        for(;i<a.size;i++){
            double val = a.data.get(ValueLayout.JAVA_DOUBLE, i * DB_BYTES);
            resArray.data.set(ValueLayout.JAVA_DOUBLE, i * DB_BYTES, Math.exp(val));
        }

        return resArray;
    }

    public static NDArray expInt(NDArray a,NDArray resArray){
        long i=0;
        long loopbound=SPECIESINT.loopBound(a.size);

        for(;i<loopbound;i+=INT_VL){
            var vInt=IntVector.fromMemorySegment(SPECIESINT, a.data, i*INT_BYTES, ORDER);
            var vFloat=vInt.convert(VectorOperators.I2F, 0);
            var VRes=vFloat.lanewise(VectorOperators.EXP);
            VRes.intoMemorySegment(resArray.data, i*FLOAT_BYTES, ORDER);
        }

        for(;i<a.size;i++){
            float val=a.data.get(ValueLayout.JAVA_INT, i*INT_BYTES);
            resArray.data.set(ValueLayout.JAVA_FLOAT,i*FLOAT_BYTES,(float) Math.exp(val));
        }
        return resArray;
    }

    //LOG methods

    public static NDArray logFloat(NDArray a,NDArray resArray){
        long i=0;
        long loopbound=SPECIES.loopBound(a.size);

        for(;i<loopbound;i+=VL){
            var v=FloatVector.fromMemorySegment(SPECIES,a.data,i*FLOAT_BYTES,ORDER);
            var VRes=v.lanewise(VectorOperators.LOG);
            VRes.intoMemorySegment(resArray.data,i*FLOAT_BYTES,ORDER);
        }

        for(;i<a.size;i++){
            float val = a.data.get(ValueLayout.JAVA_FLOAT, i * FLOAT_BYTES);
            resArray.data.set(ValueLayout.JAVA_FLOAT, i * FLOAT_BYTES, (float) Math.log(val));
        }

        return resArray;
    }

    public static NDArray logDouble(NDArray a,NDArray resArray){
        long i=0;
        long loopbound=SPECIESDB.loopBound(a.size);

        for(;i<loopbound;i+=DB_VL){
            var v=DoubleVector.fromMemorySegment(SPECIESDB,a.data,i*DB_BYTES,ORDER);
            var VRes=v.lanewise(VectorOperators.LOG);
            VRes.intoMemorySegment(resArray.data,i*DB_BYTES,ORDER);
        }

        for(;i<a.size;i++){
            double val = a.data.get(ValueLayout.JAVA_DOUBLE, i * DB_BYTES);
            resArray.data.set(ValueLayout.JAVA_DOUBLE, i * DB_BYTES, Math.log(val));
        }

        return resArray;
    }

    public static NDArray logInt(NDArray a,NDArray resArray){
        long i=0;
        long loopbound=SPECIESINT.loopBound(a.size);

        for(;i<loopbound;i+=INT_VL){
            var vInt=IntVector.fromMemorySegment(SPECIESINT, a.data, i*INT_BYTES, ORDER);
            var vFloat=vInt.convert(VectorOperators.I2F, 0);
            var VRes=vFloat.lanewise(VectorOperators.LOG);
            VRes.intoMemorySegment(resArray.data, i*FLOAT_BYTES, ORDER);
        }

        for(;i<a.size;i++){
            float val=a.data.get(ValueLayout.JAVA_INT, i*INT_BYTES);
            resArray.data.set(ValueLayout.JAVA_FLOAT,i*FLOAT_BYTES,(float) Math.log(val));
        }
        return resArray;
    }

    //LOG10 methods

    public static NDArray log10Float(NDArray a,NDArray resArray){
        long i=0;
        long loopbound=SPECIES.loopBound(a.size);

        for(;i<loopbound;i+=VL){
            var v=FloatVector.fromMemorySegment(SPECIES,a.data,i*FLOAT_BYTES,ORDER);
            var VRes=v.lanewise(VectorOperators.LOG10);
            VRes.intoMemorySegment(resArray.data,i*FLOAT_BYTES,ORDER);
        }

        for(;i<a.size;i++){
            float val = a.data.get(ValueLayout.JAVA_FLOAT, i * FLOAT_BYTES);
            resArray.data.set(ValueLayout.JAVA_FLOAT, i * FLOAT_BYTES, (float) Math.log10(val));
        }

        return resArray;
    }

    public static NDArray log10Double(NDArray a,NDArray resArray){
        long i=0;
        long loopbound=SPECIESDB.loopBound(a.size);

        for(;i<loopbound;i+=DB_VL){
            var v=DoubleVector.fromMemorySegment(SPECIESDB,a.data,i*DB_BYTES,ORDER);
            var VRes=v.lanewise(VectorOperators.LOG10);
            VRes.intoMemorySegment(resArray.data,i*DB_BYTES,ORDER);
        }

        for(;i<a.size;i++){
            double val = a.data.get(ValueLayout.JAVA_DOUBLE, i * DB_BYTES);
            resArray.data.set(ValueLayout.JAVA_DOUBLE, i * DB_BYTES, Math.log10(val));
        }

        return resArray;
    }

    public static NDArray log10Int(NDArray a,NDArray resArray){
        long i=0;
        long loopbound=SPECIESINT.loopBound(a.size);

        for(;i<loopbound;i+=INT_VL){
            var vInt=IntVector.fromMemorySegment(SPECIESINT, a.data, i*INT_BYTES, ORDER);
            var vFloat=vInt.convert(VectorOperators.I2F, 0);
            var VRes=vFloat.lanewise(VectorOperators.LOG10);
            VRes.intoMemorySegment(resArray.data, i*FLOAT_BYTES, ORDER);
        }

        for(;i<a.size;i++){
            float val=a.data.get(ValueLayout.JAVA_INT, i*INT_BYTES);
            resArray.data.set(ValueLayout.JAVA_FLOAT,i*FLOAT_BYTES,(float) Math.log10(val));
        }
        return resArray;
    }

    //TRIGNOMETRIC and HYPERBOLIC FUCTIONS

    //SIN methods

    public static NDArray sinFloat(NDArray a,NDArray resArray){
        long i=0;
        long loopbound=SPECIES.loopBound(a.size);

        for(;i<loopbound;i+=VL){
            var v=FloatVector.fromMemorySegment(SPECIES,a.data,i*FLOAT_BYTES,ORDER);
            var VRes=v.lanewise(VectorOperators.SIN);
            VRes.intoMemorySegment(resArray.data, i*FLOAT_BYTES, ORDER);
        }

        for(;i<a.size;i++){
            float val=a.data.get(ValueLayout.JAVA_FLOAT,i*FLOAT_BYTES);
            resArray.data.set(ValueLayout.JAVA_FLOAT,i*FLOAT_BYTES,(float) Math.sin(val));
        }
        return resArray;
    }

    public static NDArray sinDouble(NDArray a,NDArray resArray){
        long i=0;
        long loopbound=SPECIESDB.loopBound(a.size);

        for(;i<loopbound;i+=DB_VL){
            var v=DoubleVector.fromMemorySegment(SPECIESDB,a.data,i*DB_BYTES,ORDER);
            var VRes=v.lanewise(VectorOperators.SIN);
            VRes.intoMemorySegment(resArray.data,i*DB_BYTES,ORDER);
        }

        for(;i<a.size;i++){
            double val = a.data.get(ValueLayout.JAVA_DOUBLE, i * DB_BYTES);
            resArray.data.set(ValueLayout.JAVA_DOUBLE, i * DB_BYTES, Math.sin(val));
        }

        return resArray;
    }

    public static NDArray sinInt(NDArray a,NDArray resArray){
        long i=0;
        long loopbound=SPECIESINT.loopBound(a.size);

        for(;i<loopbound;i+=INT_VL){
            var vInt=IntVector.fromMemorySegment(SPECIESINT, a.data, i*INT_BYTES, ORDER);
            var vFloat=vInt.convert(VectorOperators.I2F, 0);
            var VRes=vFloat.lanewise(VectorOperators.SIN);
            VRes.intoMemorySegment(resArray.data, i*FLOAT_BYTES, ORDER);
        }

        for(;i<a.size;i++){
            float val=a.data.get(ValueLayout.JAVA_INT, i*INT_BYTES);
            resArray.data.set(ValueLayout.JAVA_FLOAT,i*FLOAT_BYTES,(float) Math.sin(val));
        }
        return resArray;
    }

    //COS methods

    public static NDArray cosFloat(NDArray a,NDArray resArray){
        long i=0;
        long loopbound=SPECIES.loopBound(a.size);

        for(;i<loopbound;i+=VL){
            var v=FloatVector.fromMemorySegment(SPECIES,a.data,i*FLOAT_BYTES,ORDER);
            var VRes=v.lanewise(VectorOperators.COS);
            VRes.intoMemorySegment(resArray.data, i*FLOAT_BYTES, ORDER);
        }

        for(;i<a.size;i++){
            float val=a.data.get(ValueLayout.JAVA_FLOAT,i*FLOAT_BYTES);
            resArray.data.set(ValueLayout.JAVA_FLOAT,i*FLOAT_BYTES,(float) Math.cos(val));
        }
        return resArray;
    }

    public static NDArray cosDouble(NDArray a,NDArray resArray){
        long i=0;
        long loopbound=SPECIESDB.loopBound(a.size);

        for(;i<loopbound;i+=DB_VL){
            var v=DoubleVector.fromMemorySegment(SPECIESDB,a.data,i*DB_BYTES,ORDER);
            var VRes=v.lanewise(VectorOperators.COS);
            VRes.intoMemorySegment(resArray.data,i*DB_BYTES,ORDER);
        }

        for(;i<a.size;i++){
            double val = a.data.get(ValueLayout.JAVA_DOUBLE, i * DB_BYTES);
            resArray.data.set(ValueLayout.JAVA_DOUBLE, i * DB_BYTES, Math.cos(val));
        }

        return resArray;
    }


    public static NDArray cosInt(NDArray a,NDArray resArray){
        long i=0;
        long loopbound=SPECIESINT.loopBound(a.size);

        for(;i<loopbound;i+=INT_VL){
            var vInt=IntVector.fromMemorySegment(SPECIESINT, a.data, i*INT_BYTES, ORDER);
            var vFloat=vInt.convert(VectorOperators.I2F, 0);
            var VRes=vFloat.lanewise(VectorOperators.COS);
            VRes.intoMemorySegment(resArray.data, i*FLOAT_BYTES, ORDER);
        }

        for(;i<a.size;i++){
            float val=a.data.get(ValueLayout.JAVA_INT, i*INT_BYTES);
            resArray.data.set(ValueLayout.JAVA_FLOAT,i*FLOAT_BYTES,(float) Math.cos(val));
        }
        return resArray;
    }

    //SINH methods

    public static NDArray sinhFloat(NDArray a,NDArray resArray){
        long i=0;
        long loopbound=SPECIES.loopBound(a.size);

        for(;i<loopbound;i+=VL){
            var v=FloatVector.fromMemorySegment(SPECIES,a.data,i*FLOAT_BYTES,ORDER);
            var VRes=v.lanewise(VectorOperators.SINH);
            VRes.intoMemorySegment(resArray.data, i*FLOAT_BYTES, ORDER);
        }

        for(;i<a.size;i++){
            float val=a.data.get(ValueLayout.JAVA_FLOAT,i*FLOAT_BYTES);
            resArray.data.set(ValueLayout.JAVA_FLOAT,i*FLOAT_BYTES,(float) Math.sinh(val));
        }
        return resArray;
    }

    public static NDArray sinhDouble(NDArray a,NDArray resArray){
        long i=0;
        long loopbound=SPECIESDB.loopBound(a.size);

        for(;i<loopbound;i+=DB_VL){
            var v=DoubleVector.fromMemorySegment(SPECIESDB,a.data,i*DB_BYTES,ORDER);
            var VRes=v.lanewise(VectorOperators.SINH);
            VRes.intoMemorySegment(resArray.data,i*DB_BYTES,ORDER);
        }

        for(;i<a.size;i++){
            double val = a.data.get(ValueLayout.JAVA_DOUBLE, i * DB_BYTES);
            resArray.data.set(ValueLayout.JAVA_DOUBLE, i * DB_BYTES, Math.sinh(val));
        }

        return resArray;
    }

    public static NDArray sinhInt(NDArray a,NDArray resArray){
        long i=0;
        long loopbound=SPECIESINT.loopBound(a.size);

        for(;i<loopbound;i+=INT_VL){
            var vInt=IntVector.fromMemorySegment(SPECIESINT, a.data, i*INT_BYTES, ORDER);
            var vFloat=vInt.convert(VectorOperators.I2F, 0);
            var VRes=vFloat.lanewise(VectorOperators.SINH);
            VRes.intoMemorySegment(resArray.data, i*FLOAT_BYTES, ORDER);
        }

        for(;i<a.size;i++){
            float val=a.data.get(ValueLayout.JAVA_INT, i*INT_BYTES);
            resArray.data.set(ValueLayout.JAVA_FLOAT,i*FLOAT_BYTES,(float) Math.sinh(val));
        }
        return resArray;
    }

    //COSH methods 

    public static NDArray coshFloat(NDArray a,NDArray resArray){
        long i=0;
        long loopbound=SPECIES.loopBound(a.size);

        for(;i<loopbound;i+=VL){
            var v=FloatVector.fromMemorySegment(SPECIES,a.data,i*FLOAT_BYTES,ORDER);
            var VRes=v.lanewise(VectorOperators.COSH);
            VRes.intoMemorySegment(resArray.data, i*FLOAT_BYTES, ORDER);
        }

        for(;i<a.size;i++){
            float val=a.data.get(ValueLayout.JAVA_FLOAT,i*FLOAT_BYTES);
            resArray.data.set(ValueLayout.JAVA_FLOAT,i*FLOAT_BYTES,(float) Math.cosh(val));
        }
        return resArray;
    }

    public static NDArray coshDouble(NDArray a,NDArray resArray){
        long i=0;
        long loopbound=SPECIESDB.loopBound(a.size);

        for(;i<loopbound;i+=DB_VL){
            var v=DoubleVector.fromMemorySegment(SPECIESDB,a.data,i*DB_BYTES,ORDER);
            var VRes=v.lanewise(VectorOperators.COSH);
            VRes.intoMemorySegment(resArray.data,i*DB_BYTES,ORDER);
        }

        for(;i<a.size;i++){
            double val = a.data.get(ValueLayout.JAVA_DOUBLE, i * DB_BYTES);
            resArray.data.set(ValueLayout.JAVA_DOUBLE, i * DB_BYTES, Math.cosh(val));
        }

        return resArray;
    }

    public static NDArray coshInt(NDArray a,NDArray resArray){
        long i=0;
        long loopbound=SPECIESINT.loopBound(a.size);

        for(;i<loopbound;i+=INT_VL){
            var vInt=IntVector.fromMemorySegment(SPECIESINT, a.data, i*INT_BYTES, ORDER);
            var vFloat=vInt.convert(VectorOperators.I2F, 0);
            var VRes=vFloat.lanewise(VectorOperators.COSH);
            VRes.intoMemorySegment(resArray.data, i*FLOAT_BYTES, ORDER);
        }

        for(;i<a.size;i++){
            float val=a.data.get(ValueLayout.JAVA_INT, i*INT_BYTES);
            resArray.data.set(ValueLayout.JAVA_FLOAT,i*FLOAT_BYTES,(float) Math.cosh(val));
        }
        return resArray;
    }



}
