package jnum.jnumops;

import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;
import jnum.NDArray;
import jdk.incubator.vector.VectorOperators;
public class TrigOps {
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
    private TrigOps(){
        throw new AssertionError();
    }

    //SIN methods

    public static NDArray sinFloat(NDArray a,NDArray resArray){
        long i=0;
        long loopbound = a.size - (a.size % (VL * 2));

        for(;i<loopbound;i+=VL*2){
            var v1=FloatVector.fromMemorySegment(SPECIES,a.data,i*FLOAT_BYTES,ORDER);
            var v2=FloatVector.fromMemorySegment(SPECIES, a.data, (i+VL)*FLOAT_BYTES, ORDER);
            var VRes1=v1.lanewise(VectorOperators.SIN);
            var VRes2=v2.lanewise(VectorOperators.SIN);
            VRes1.intoMemorySegment(resArray.data, i*FLOAT_BYTES, ORDER);
            VRes2.intoMemorySegment(resArray.data, (i+VL)*FLOAT_BYTES, ORDER);
        }

        loopbound=SPECIES.loopBound(a.size);

        for(;i<loopbound;i+=VL){
            var v=FloatVector.fromMemorySegment(SPECIES,a.data,i*FLOAT_BYTES,ORDER);
            var VRes=v.lanewise(VectorOperators.SIN);
            VRes.intoMemorySegment(resArray.data,i*FLOAT_BYTES,ORDER);
        }

        for(;i<a.size;i++){
            float val=a.data.get(ValueLayout.JAVA_FLOAT,i*FLOAT_BYTES);
            resArray.data.set(ValueLayout.JAVA_FLOAT,i*FLOAT_BYTES,(float) Math.sin(val));
        }
        return resArray;
    }

    public static NDArray sinDouble(NDArray a,NDArray resArray){
        long i=0;
        long loopbound=a.size - (a.size % (DB_VL * 2));

        for(;i<loopbound;i+=DB_VL*2){
            var v1=DoubleVector.fromMemorySegment(SPECIESDB,a.data,i*DB_BYTES,ORDER);
            var v2=DoubleVector.fromMemorySegment(SPECIESDB,a.data,(i+DB_VL)*DB_BYTES,ORDER);
            var VRes1=v1.lanewise(VectorOperators.SIN);
            var VRes2=v2.lanewise(VectorOperators.SIN);
            VRes1.intoMemorySegment(resArray.data,i*DB_BYTES,ORDER);
            VRes2.intoMemorySegment(resArray.data,(i+DB_VL)*DB_BYTES,ORDER);
        }

        loopbound=SPECIESDB.loopBound(a.size);

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
        long loopbound=a.size - (a.size % (INT_VL * 2));

        for(;i<loopbound;i+=INT_VL*2){
            var vInt1=IntVector.fromMemorySegment(SPECIESINT, a.data, i*INT_BYTES, ORDER);
            var vInt2=IntVector.fromMemorySegment(SPECIESINT, a.data, (i+INT_VL)*INT_BYTES, ORDER);
            var vFloat1=vInt1.convert(VectorOperators.I2F, 0);
            var vFloat2=vInt2.convert(VectorOperators.I2F, 0);
            var VRes1=vFloat1.lanewise(VectorOperators.SIN);
            var VRes2=vFloat2.lanewise(VectorOperators.SIN);
            VRes1.intoMemorySegment(resArray.data, i*FLOAT_BYTES, ORDER);
            VRes2.intoMemorySegment(resArray.data, (i+INT_VL)*FLOAT_BYTES, ORDER);
        }

        loopbound=SPECIESINT.loopBound(a.size);

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
        long loopbound=a.size - (a.size % (VL * 2));

        for(;i<loopbound;i+=VL*2){
            var v1=FloatVector.fromMemorySegment(SPECIES,a.data,i*FLOAT_BYTES,ORDER);
            var v2=FloatVector.fromMemorySegment(SPECIES,a.data,(i+VL)*FLOAT_BYTES,ORDER);
            var VRes1=v1.lanewise(VectorOperators.COS);
            var VRes2=v2.lanewise(VectorOperators.COS);
            VRes1.intoMemorySegment(resArray.data, i*FLOAT_BYTES, ORDER);
            VRes2.intoMemorySegment(resArray.data, (i+VL)*FLOAT_BYTES, ORDER);
        }

        loopbound=SPECIES.loopBound(a.size);

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
        long loopbound=a.size - (a.size % (DB_VL * 2));

        for(;i<loopbound;i+=DB_VL*2){
            var v1=DoubleVector.fromMemorySegment(SPECIESDB,a.data,i*DB_BYTES,ORDER);
            var v2=DoubleVector.fromMemorySegment(SPECIESDB,a.data,(i+DB_VL)*DB_BYTES,ORDER);
            var VRes1=v1.lanewise(VectorOperators.COS);
            var VRes2=v2.lanewise(VectorOperators.COS);
            VRes1.intoMemorySegment(resArray.data,i*DB_BYTES,ORDER);
            VRes2.intoMemorySegment(resArray.data,(i+DB_VL)*DB_BYTES,ORDER);
        }

        loopbound=SPECIESDB.loopBound(a.size);

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
        long loopbound=a.size - (a.size % (INT_VL * 2));

        for(;i<loopbound;i+=INT_VL*2){
            var vInt1=IntVector.fromMemorySegment(SPECIESINT, a.data, i*INT_BYTES, ORDER);
            var vInt2=IntVector.fromMemorySegment(SPECIESINT, a.data, (i+INT_VL)*INT_BYTES, ORDER);
            var vFloat1=vInt1.convert(VectorOperators.I2F, 0);
            var vFloat2=vInt2.convert(VectorOperators.I2F, 0);
            var VRes1=vFloat1.lanewise(VectorOperators.COS);
            var VRes2=vFloat2.lanewise(VectorOperators.COS);
            VRes1.intoMemorySegment(resArray.data, i*FLOAT_BYTES, ORDER);
            VRes2.intoMemorySegment(resArray.data, (i+INT_VL)*FLOAT_BYTES, ORDER);
        }

        loopbound=SPECIESINT.loopBound(a.size);

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

    //TAN methods

    public static NDArray tanFloat(NDArray a,NDArray resArray){
        long i=0;
        long loopbound=a.size - (a.size % (VL * 2));

        for(;i<loopbound;i+=VL*2){
            var v1=FloatVector.fromMemorySegment(SPECIES,a.data,i*FLOAT_BYTES,ORDER);
            var v2=FloatVector.fromMemorySegment(SPECIES,a.data,(i+VL)*FLOAT_BYTES,ORDER);
            var VRes1=v1.lanewise(VectorOperators.TAN);
            var VRes2=v2.lanewise(VectorOperators.TAN);
            VRes1.intoMemorySegment(resArray.data, i*FLOAT_BYTES, ORDER);
            VRes2.intoMemorySegment(resArray.data, (i+VL)*FLOAT_BYTES, ORDER);
        }

        loopbound=SPECIES.loopBound(a.size);

        for(;i<loopbound;i+=VL){
            var v=FloatVector.fromMemorySegment(SPECIES,a.data,i*FLOAT_BYTES,ORDER);
            var VRes=v.lanewise(VectorOperators.TAN);
            VRes.intoMemorySegment(resArray.data, i*FLOAT_BYTES, ORDER);
        }

        for(;i<a.size;i++){
            float val=a.data.get(ValueLayout.JAVA_FLOAT,i*FLOAT_BYTES);
            resArray.data.set(ValueLayout.JAVA_FLOAT,i*FLOAT_BYTES,(float) Math.tan(val));
        }
        return resArray;
    }

    public static NDArray tanDouble(NDArray a,NDArray resArray){
        long i=0;
        long loopbound=a.size - (a.size % (DB_VL * 2));

        for(;i<loopbound;i+=DB_VL*2){
            var v1=DoubleVector.fromMemorySegment(SPECIESDB,a.data,i*DB_BYTES,ORDER);
            var v2=DoubleVector.fromMemorySegment(SPECIESDB,a.data,(i+DB_VL)*DB_BYTES,ORDER);
            var VRes1=v1.lanewise(VectorOperators.TAN);
            var VRes2=v2.lanewise(VectorOperators.TAN);
            VRes1.intoMemorySegment(resArray.data,i*DB_BYTES,ORDER);
            VRes2.intoMemorySegment(resArray.data,(i+DB_VL)*DB_BYTES,ORDER);
        }

        loopbound=SPECIESDB.loopBound(a.size);

        for(;i<loopbound;i+=DB_VL){
            var v=DoubleVector.fromMemorySegment(SPECIESDB,a.data,i*DB_BYTES,ORDER);
            var VRes=v.lanewise(VectorOperators.TAN);
            VRes.intoMemorySegment(resArray.data,i*DB_BYTES,ORDER);
        }

        for(;i<a.size;i++){
            double val = a.data.get(ValueLayout.JAVA_DOUBLE, i * DB_BYTES);
            resArray.data.set(ValueLayout.JAVA_DOUBLE, i * DB_BYTES, Math.tan(val));
        }

        return resArray;
    }

    public static NDArray tanInt(NDArray a,NDArray resArray){
        long i=0;
        long loopbound=a.size - (a.size % (INT_VL * 2));

        for(;i<loopbound;i+=INT_VL*2){
            var vInt1=IntVector.fromMemorySegment(SPECIESINT, a.data, i*INT_BYTES, ORDER);
            var vInt2=IntVector.fromMemorySegment(SPECIESINT, a.data, (i+INT_VL)*INT_BYTES, ORDER);
            var vFloat1=vInt1.convert(VectorOperators.I2F, 0);
            var vFloat2=vInt2.convert(VectorOperators.I2F, 0);
            var VRes1=vFloat1.lanewise(VectorOperators.TAN);
            var VRes2=vFloat2.lanewise(VectorOperators.TAN);
            VRes1.intoMemorySegment(resArray.data, i*FLOAT_BYTES, ORDER);
            VRes2.intoMemorySegment(resArray.data, (i+INT_VL)*FLOAT_BYTES, ORDER);
        }

        loopbound=SPECIESINT.loopBound(a.size);

        for(;i<loopbound;i+=INT_VL){
            var vInt=IntVector.fromMemorySegment(SPECIESINT, a.data, i*INT_BYTES, ORDER);
            var vFloat=vInt.convert(VectorOperators.I2F, 0);
            var VRes=vFloat.lanewise(VectorOperators.TAN);
            VRes.intoMemorySegment(resArray.data, i*FLOAT_BYTES, ORDER);
        }

        for(;i<a.size;i++){
            float val=a.data.get(ValueLayout.JAVA_INT, i*INT_BYTES);
            resArray.data.set(ValueLayout.JAVA_FLOAT,i*FLOAT_BYTES,(float) Math.tan(val));
        }
        return resArray;
    }

    //COT methods

    public static NDArray cotFloat(NDArray a, NDArray resArray){
        long i=0;
        long loopbound=a.size - (a.size % (VL * 2));
        var vOnes=FloatVector.broadcast(SPECIES, 1.0f);

        for(;i<loopbound;i+=VL*2){
            var v1=FloatVector.fromMemorySegment(SPECIES,a.data,i*FLOAT_BYTES,ORDER);
            var v2=FloatVector.fromMemorySegment(SPECIES,a.data,(i+VL)*FLOAT_BYTES,ORDER);
            var vTan1=v1.lanewise(VectorOperators.TAN);
            var vTan2=v2.lanewise(VectorOperators.TAN);
            var VRes1=vOnes.div(vTan1);
            var VRes2=vOnes.div(vTan2);
            VRes1.intoMemorySegment(resArray.data, i*FLOAT_BYTES, ORDER);
            VRes2.intoMemorySegment(resArray.data, (i+VL)*FLOAT_BYTES, ORDER);
        }

        loopbound=SPECIES.loopBound(a.size);

        for(;i<loopbound;i+=VL){
            var v=FloatVector.fromMemorySegment(SPECIES,a.data,i*FLOAT_BYTES,ORDER);
            var vTan=v.lanewise(VectorOperators.TAN);
            var VRes=vOnes.div(vTan);
            VRes.intoMemorySegment(resArray.data, i*FLOAT_BYTES, ORDER);
        }

        for(;i<a.size;i++){
            float val=a.data.get(ValueLayout.JAVA_FLOAT,i*FLOAT_BYTES);
            resArray.data.set(ValueLayout.JAVA_FLOAT,i*FLOAT_BYTES,(float) (1.0 / Math.tan(val)));
        }
        return resArray;
    }

    public static NDArray cotDouble(NDArray a, NDArray resArray){
        long i=0;
        long loopbound=a.size - (a.size % (DB_VL * 2));
        var vOnes=DoubleVector.broadcast(SPECIESDB, 1.0);

        for(;i<loopbound;i+=DB_VL*2){
            var v1=DoubleVector.fromMemorySegment(SPECIESDB,a.data,i*DB_BYTES,ORDER);
            var v2=DoubleVector.fromMemorySegment(SPECIESDB,a.data,(i+DB_VL)*DB_BYTES,ORDER);
            var vTan1=v1.lanewise(VectorOperators.TAN);
            var vTan2=v2.lanewise(VectorOperators.TAN);
            var VRes1=vOnes.div(vTan1);
            var VRes2=vOnes.div(vTan2);
            VRes1.intoMemorySegment(resArray.data,i*DB_BYTES,ORDER);
            VRes2.intoMemorySegment(resArray.data,(i+DB_VL)*DB_BYTES,ORDER);
        }

        loopbound=SPECIESDB.loopBound(a.size);

        for(;i<loopbound;i+=DB_VL){
            var v=DoubleVector.fromMemorySegment(SPECIESDB,a.data,i*DB_BYTES,ORDER);
            var vTan=v.lanewise(VectorOperators.TAN);
            var VRes=vOnes.div(vTan);
            VRes.intoMemorySegment(resArray.data,i*DB_BYTES,ORDER);
        }

        for(;i<a.size;i++){
            double val = a.data.get(ValueLayout.JAVA_DOUBLE, i * DB_BYTES);
            resArray.data.set(ValueLayout.JAVA_DOUBLE, i * DB_BYTES, (1.0 / Math.tan(val)));
        }

        return resArray;
    }

    public static NDArray cotInt(NDArray a, NDArray resArray){
        long i=0;
        long loopbound=a.size - (a.size % (INT_VL * 2));
        var vOnes=FloatVector.broadcast(SPECIES, 1.0f);

        for(;i<loopbound;i+=INT_VL*2){
            var vInt1=IntVector.fromMemorySegment(SPECIESINT, a.data, i*INT_BYTES, ORDER);
            var vInt2=IntVector.fromMemorySegment(SPECIESINT, a.data, (i+INT_VL)*INT_BYTES, ORDER);
            var vFloat1=vInt1.convert(VectorOperators.I2F, 0);
            var vFloat2=vInt2.convert(VectorOperators.I2F, 0);
            var vTan1=vFloat1.lanewise(VectorOperators.TAN);
            var vTan2=vFloat2.lanewise(VectorOperators.TAN);
            var VRes1=vOnes.div(vTan1);
            var VRes2=vOnes.div(vTan2);
            VRes1.intoMemorySegment(resArray.data, i*FLOAT_BYTES, ORDER);
            VRes2.intoMemorySegment(resArray.data, (i+INT_VL)*FLOAT_BYTES, ORDER);
        }

        loopbound=SPECIESINT.loopBound(a.size);

        for(;i<loopbound;i+=INT_VL){
            var vInt=IntVector.fromMemorySegment(SPECIESINT, a.data, i*INT_BYTES, ORDER);
            var vFloat=vInt.convert(VectorOperators.I2F, 0);
            var vTan=vFloat.lanewise(VectorOperators.TAN);
            var VRes=vOnes.div(vTan);
            VRes.intoMemorySegment(resArray.data, i*FLOAT_BYTES, ORDER);
        }

        for(;i<a.size;i++){
            float val=a.data.get(ValueLayout.JAVA_INT, i*INT_BYTES);
            resArray.data.set(ValueLayout.JAVA_FLOAT,i*FLOAT_BYTES,(float) (1.0 / Math.tan(val)));
        }
        return resArray;
    }

    //SINH methods

    public static NDArray sinhFloat(NDArray a,NDArray resArray){
        long i=0;
        long loopbound=a.size - (a.size % (VL * 2));

        for(;i<loopbound;i+=VL*2){
            var v1=FloatVector.fromMemorySegment(SPECIES,a.data,i*FLOAT_BYTES,ORDER);
            var v2=FloatVector.fromMemorySegment(SPECIES,a.data,(i+VL)*FLOAT_BYTES,ORDER);
            var VRes1=v1.lanewise(VectorOperators.SINH);
            var VRes2=v2.lanewise(VectorOperators.SINH);
            VRes1.intoMemorySegment(resArray.data, i*FLOAT_BYTES, ORDER);
            VRes2.intoMemorySegment(resArray.data, (i+VL)*FLOAT_BYTES, ORDER);
        }

        loopbound=SPECIES.loopBound(a.size);

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
        long loopbound=a.size - (a.size % (DB_VL * 2));

        for(;i<loopbound;i+=DB_VL*2){
            var v1=DoubleVector.fromMemorySegment(SPECIESDB,a.data,i*DB_BYTES,ORDER);
            var v2=DoubleVector.fromMemorySegment(SPECIESDB,a.data,(i+DB_VL)*DB_BYTES,ORDER);
            var VRes1=v1.lanewise(VectorOperators.SINH);
            var VRes2=v2.lanewise(VectorOperators.SINH);
            VRes1.intoMemorySegment(resArray.data,i*DB_BYTES,ORDER);
            VRes2.intoMemorySegment(resArray.data,(i+DB_VL)*DB_BYTES,ORDER);
        }

        loopbound=SPECIESDB.loopBound(a.size);

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
        long loopbound=a.size - (a.size % (INT_VL * 2));

        for(;i<loopbound;i+=INT_VL*2){
            var vInt1=IntVector.fromMemorySegment(SPECIESINT, a.data, i*INT_BYTES, ORDER);
            var vInt2=IntVector.fromMemorySegment(SPECIESINT, a.data, (i+INT_VL)*INT_BYTES, ORDER);
            var vFloat1=vInt1.convert(VectorOperators.I2F, 0);
            var vFloat2=vInt2.convert(VectorOperators.I2F, 0);
            var VRes1=vFloat1.lanewise(VectorOperators.SINH);
            var VRes2=vFloat2.lanewise(VectorOperators.SINH);
            VRes1.intoMemorySegment(resArray.data, i*FLOAT_BYTES, ORDER);
            VRes2.intoMemorySegment(resArray.data, (i+INT_VL)*FLOAT_BYTES, ORDER);
        }

        loopbound=SPECIESINT.loopBound(a.size);

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
        long loopbound=a.size - (a.size % (VL * 2));

        for(;i<loopbound;i+=VL*2){
            var v1=FloatVector.fromMemorySegment(SPECIES,a.data,i*FLOAT_BYTES,ORDER);
            var v2=FloatVector.fromMemorySegment(SPECIES,a.data,(i+VL)*FLOAT_BYTES,ORDER);
            var VRes1=v1.lanewise(VectorOperators.COSH);
            var VRes2=v2.lanewise(VectorOperators.COSH);
            VRes1.intoMemorySegment(resArray.data, i*FLOAT_BYTES, ORDER);
            VRes2.intoMemorySegment(resArray.data, (i+VL)*FLOAT_BYTES, ORDER);
        }

        loopbound=SPECIES.loopBound(a.size);

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
        long loopbound=a.size - (a.size % (DB_VL * 2));

        for(;i<loopbound;i+=DB_VL*2){
            var v1=DoubleVector.fromMemorySegment(SPECIESDB,a.data,i*DB_BYTES,ORDER);
            var v2=DoubleVector.fromMemorySegment(SPECIESDB,a.data,(i+DB_VL)*DB_BYTES,ORDER);
            var VRes1=v1.lanewise(VectorOperators.COSH);
            var VRes2=v2.lanewise(VectorOperators.COSH);
            VRes1.intoMemorySegment(resArray.data,i*DB_BYTES,ORDER);
            VRes2.intoMemorySegment(resArray.data,(i+DB_VL)*DB_BYTES,ORDER);
        }

        loopbound=SPECIESDB.loopBound(a.size);

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
        long loopbound=a.size - (a.size % (INT_VL * 2));

        for(;i<loopbound;i+=INT_VL*2){
            var vInt1=IntVector.fromMemorySegment(SPECIESINT, a.data, i*INT_BYTES, ORDER);
            var vInt2=IntVector.fromMemorySegment(SPECIESINT, a.data, (i+INT_VL)*INT_BYTES, ORDER);
            var vFloat1=vInt1.convert(VectorOperators.I2F, 0);
            var vFloat2=vInt2.convert(VectorOperators.I2F, 0);
            var VRes1=vFloat1.lanewise(VectorOperators.COSH);
            var VRes2=vFloat2.lanewise(VectorOperators.COSH);
            VRes1.intoMemorySegment(resArray.data, i*FLOAT_BYTES, ORDER);
            VRes2.intoMemorySegment(resArray.data, (i+INT_VL)*FLOAT_BYTES, ORDER);
        }

        loopbound=SPECIESINT.loopBound(a.size);

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

    //TANH methods

    public static NDArray tanhFloat(NDArray a,NDArray resArray){
        long i=0;
        long loopbound=a.size - (a.size % (VL * 2));

        for(;i<loopbound;i+=VL*2){
            var v1=FloatVector.fromMemorySegment(SPECIES,a.data,i*FLOAT_BYTES,ORDER);
            var v2=FloatVector.fromMemorySegment(SPECIES,a.data,(i+VL)*FLOAT_BYTES,ORDER);
            var VRes1=v1.lanewise(VectorOperators.TANH);
            var VRes2=v2.lanewise(VectorOperators.TANH);
            VRes1.intoMemorySegment(resArray.data, i*FLOAT_BYTES, ORDER);
            VRes2.intoMemorySegment(resArray.data, (i+VL)*FLOAT_BYTES, ORDER);
        }

        loopbound=SPECIES.loopBound(a.size);

        for(;i<loopbound;i+=VL){
            var v=FloatVector.fromMemorySegment(SPECIES,a.data,i*FLOAT_BYTES,ORDER);
            var VRes=v.lanewise(VectorOperators.TANH);
            VRes.intoMemorySegment(resArray.data, i*FLOAT_BYTES, ORDER);
        }

        for(;i<a.size;i++){
            float val=a.data.get(ValueLayout.JAVA_FLOAT,i*FLOAT_BYTES);
            resArray.data.set(ValueLayout.JAVA_FLOAT,i*FLOAT_BYTES,(float) Math.tanh(val));
        }
        return resArray;
    }

    public static NDArray tanhDouble(NDArray a,NDArray resArray){
        long i=0;
        long loopbound=a.size - (a.size % (DB_VL * 2));

        for(;i<loopbound;i+=DB_VL*2){
            var v1=DoubleVector.fromMemorySegment(SPECIESDB,a.data,i*DB_BYTES,ORDER);
            var v2=DoubleVector.fromMemorySegment(SPECIESDB,a.data,(i+DB_VL)*DB_BYTES,ORDER);
            var VRes1=v1.lanewise(VectorOperators.TANH);
            var VRes2=v2.lanewise(VectorOperators.TANH);
            VRes1.intoMemorySegment(resArray.data,i*DB_BYTES,ORDER);
            VRes2.intoMemorySegment(resArray.data,(i+DB_VL)*DB_BYTES,ORDER);
        }

        loopbound=SPECIESDB.loopBound(a.size);

        for(;i<loopbound;i+=DB_VL){
            var v=DoubleVector.fromMemorySegment(SPECIESDB,a.data,i*DB_BYTES,ORDER);
            var VRes=v.lanewise(VectorOperators.TANH);
            VRes.intoMemorySegment(resArray.data,i*DB_BYTES,ORDER);
        }

        for(;i<a.size;i++){
            double val = a.data.get(ValueLayout.JAVA_DOUBLE, i * DB_BYTES);
            resArray.data.set(ValueLayout.JAVA_DOUBLE, i * DB_BYTES, Math.tanh(val));
        }

        return resArray;
    }

    public static NDArray tanhInt(NDArray a,NDArray resArray){
        long i=0;
        long loopbound=a.size - (a.size % (INT_VL * 2));

        for(;i<loopbound;i+=INT_VL*2){
            var vInt1=IntVector.fromMemorySegment(SPECIESINT, a.data, i*INT_BYTES, ORDER);
            var vInt2=IntVector.fromMemorySegment(SPECIESINT, a.data, (i+INT_VL)*INT_BYTES, ORDER);
            var vFloat1=vInt1.convert(VectorOperators.I2F, 0);
            var vFloat2=vInt2.convert(VectorOperators.I2F, 0);
            var VRes1=vFloat1.lanewise(VectorOperators.TANH);
            var VRes2=vFloat2.lanewise(VectorOperators.TANH);
            VRes1.intoMemorySegment(resArray.data, i*FLOAT_BYTES, ORDER);
            VRes2.intoMemorySegment(resArray.data, (i+INT_VL)*FLOAT_BYTES, ORDER);
        }

        loopbound=SPECIESINT.loopBound(a.size);

        for(;i<loopbound;i+=INT_VL){
            var vInt=IntVector.fromMemorySegment(SPECIESINT, a.data, i*INT_BYTES, ORDER);
            var vFloat=vInt.convert(VectorOperators.I2F, 0);
            var VRes=vFloat.lanewise(VectorOperators.TANH);
            VRes.intoMemorySegment(resArray.data, i*FLOAT_BYTES, ORDER);
        }

        for(;i<a.size;i++){
            float val=a.data.get(ValueLayout.JAVA_INT, i*INT_BYTES);
            resArray.data.set(ValueLayout.JAVA_FLOAT,i*FLOAT_BYTES,(float) Math.tanh(val));
        }
        return resArray;
    }

    //COTH methods

    public static NDArray cothFloat(NDArray a, NDArray resArray){
        long i=0;
        long loopbound=a.size - (a.size % (VL * 2));
        var vOnes=FloatVector.broadcast(SPECIES, 1.0f);

        for(;i<loopbound;i+=VL*2){
            var v1=FloatVector.fromMemorySegment(SPECIES,a.data,i*FLOAT_BYTES,ORDER);
            var v2=FloatVector.fromMemorySegment(SPECIES,a.data,(i+VL)*FLOAT_BYTES,ORDER);
            var vTanh1=v1.lanewise(VectorOperators.TANH);
            var vTanh2=v2.lanewise(VectorOperators.TANH);
            var VRes1=vOnes.div(vTanh1);
            var VRes2=vOnes.div(vTanh2);
            VRes1.intoMemorySegment(resArray.data, i*FLOAT_BYTES, ORDER);
            VRes2.intoMemorySegment(resArray.data, (i+VL)*FLOAT_BYTES, ORDER);
        }

        loopbound=SPECIES.loopBound(a.size);
        
        for(;i<loopbound;i+=VL){
            var v=FloatVector.fromMemorySegment(SPECIES,a.data,i*FLOAT_BYTES,ORDER);
            var vTanh=v.lanewise(VectorOperators.TANH);
            var VRes=vOnes.div(vTanh);
            VRes.intoMemorySegment(resArray.data, i*FLOAT_BYTES, ORDER);
        }

        for(;i<a.size;i++){
            float val=a.data.get(ValueLayout.JAVA_FLOAT,i*FLOAT_BYTES);
            resArray.data.set(ValueLayout.JAVA_FLOAT,i*FLOAT_BYTES,(float) (1.0 / Math.tanh(val)));
        }
        return resArray;
    }

    public static NDArray cothDouble(NDArray a, NDArray resArray){
        long i=0;
        long loopbound=a.size - (a.size % (DB_VL * 2));
        var vOnes=DoubleVector.broadcast(SPECIESDB, 1.0);

        for(;i<loopbound;i+=DB_VL*2){
            var v1=DoubleVector.fromMemorySegment(SPECIESDB,a.data,i*DB_BYTES,ORDER);
            var v2=DoubleVector.fromMemorySegment(SPECIESDB,a.data,(i+DB_VL)*DB_BYTES,ORDER);
            var vTanh1=v1.lanewise(VectorOperators.TANH);
            var vTanh2=v2.lanewise(VectorOperators.TANH);
            var VRes1=vOnes.div(vTanh1);
            var VRes2=vOnes.div(vTanh2);
            VRes1.intoMemorySegment(resArray.data,i*DB_BYTES,ORDER);
            VRes2.intoMemorySegment(resArray.data,(i+DB_VL)*DB_BYTES,ORDER);
        }

        loopbound=SPECIESDB.loopBound(a.size);

        for(;i<loopbound;i+=DB_VL){
            var v=DoubleVector.fromMemorySegment(SPECIESDB,a.data,i*DB_BYTES,ORDER);
            var vTanh=v.lanewise(VectorOperators.TANH);
            var VRes=vOnes.div(vTanh);
            VRes.intoMemorySegment(resArray.data,i*DB_BYTES,ORDER);
        }

        for(;i<a.size;i++){
            double val = a.data.get(ValueLayout.JAVA_DOUBLE, i * DB_BYTES);
            resArray.data.set(ValueLayout.JAVA_DOUBLE, i * DB_BYTES, (1.0 / Math.tanh(val)));
        }

        return resArray;
    }

    public static NDArray cothInt(NDArray a, NDArray resArray){
        long i=0;
        long loopbound=a.size - (a.size % (INT_VL * 2));
        var vOnes=FloatVector.broadcast(SPECIES, 1.0f);

        for(;i<loopbound;i+=INT_VL*2){
            var vInt1=IntVector.fromMemorySegment(SPECIESINT, a.data, i*INT_BYTES, ORDER);
            var vInt2=IntVector.fromMemorySegment(SPECIESINT, a.data, (i+INT_VL)*INT_BYTES, ORDER);
            var vFloat1=vInt1.convert(VectorOperators.I2F, 0);
            var vFloat2=vInt2.convert(VectorOperators.I2F, 0);
            var vTanh1=vFloat1.lanewise(VectorOperators.TANH);
            var vTanh2=vFloat2.lanewise(VectorOperators.TANH);
            var VRes1=vOnes.div(vTanh1);
            var VRes2=vOnes.div(vTanh2);
            VRes1.intoMemorySegment(resArray.data, i*FLOAT_BYTES, ORDER);
            VRes2.intoMemorySegment(resArray.data, (i+INT_VL)*FLOAT_BYTES, ORDER);
        }

        loopbound=SPECIESINT.loopBound(a.size);

        for(;i<loopbound;i+=INT_VL){
            var vInt=IntVector.fromMemorySegment(SPECIESINT, a.data, i*INT_BYTES, ORDER);
            var vFloat=vInt.convert(VectorOperators.I2F, 0);
            var vTanh=vFloat.lanewise(VectorOperators.TANH);
            var VRes=vOnes.div(vTanh);
            VRes.intoMemorySegment(resArray.data, i*FLOAT_BYTES, ORDER);
        }

        for(;i<a.size;i++){
            float val=a.data.get(ValueLayout.JAVA_INT, i*INT_BYTES);
            resArray.data.set(ValueLayout.JAVA_FLOAT,i*FLOAT_BYTES,(float) (1.0 / Math.tanh(val)));
        }
        return resArray;
    }

    //
}
