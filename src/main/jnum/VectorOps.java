package jnum;


import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;
import jdk.incubator.vector.VectorOperators;

class VectorOps{
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

    private VectorOps(){}


    public static NDArray addFloat(NDArray a,NDArray b,NDArray resArray){        
        long i=0;
        long loopbound=SPECIES.loopBound(a.size);

        for(;i<loopbound;i+=VL){
            var v1=FloatVector.fromMemorySegment(SPECIES,a.data,i*FLOAT_BYTES,ORDER);
            var v2=FloatVector.fromMemorySegment(SPECIES,b.data,i*FLOAT_BYTES,ORDER);
            var vRes=v1.add(v2);
            vRes.intoMemorySegment(resArray.data,i*FLOAT_BYTES,ORDER);
        }

        for (; i < a.size; i++) {
            var val1 = a.data.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            var val2 = b.data.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            resArray.data.setAtIndex(ValueLayout.JAVA_FLOAT, i, val1 + val2);
        }
        return resArray;
    }

    public static NDArray addInt(NDArray a,NDArray b,NDArray resArray){        
        long i=0;
        long loopbound=SPECIES.loopBound(a.size);

        for(;i<loopbound;i+=INT_VL){
            var v1=IntVector.fromMemorySegment(SPECIESINT,a.data,i*INT_BYTES,ORDER);
            var v2=IntVector.fromMemorySegment(SPECIESINT,b.data,i*INT_BYTES,ORDER);
            var vRes=v1.add(v2);
            vRes.intoMemorySegment(resArray.data,i*INT_BYTES,ORDER);
        }

        for (; i < a.size; i++) {
            var val1 = a.data.getAtIndex(ValueLayout.JAVA_INT, i);
            var val2 = b.data.getAtIndex(ValueLayout.JAVA_INT, i);
            resArray.data.setAtIndex(ValueLayout.JAVA_INT, i, val1 + val2);
        }
        return resArray;
    }

    public static NDArray addDouble(NDArray a,NDArray b,NDArray resArray){        
        long i=0;
        long loopbound=SPECIES.loopBound(a.size);

        for(;i<loopbound;i+=DB_VL){
            var v1=DoubleVector.fromMemorySegment(SPECIESDB,a.data,i*DB_BYTES,ORDER);
            var v2=DoubleVector.fromMemorySegment(SPECIESDB,b.data,i*DB_BYTES,ORDER);
            var vRes=v1.add(v2);
            vRes.intoMemorySegment(resArray.data,i*DB_BYTES,ORDER);
        }

        for (; i < a.size; i++) {
            var val1 = a.data.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
            var val2 = b.data.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
            resArray.data.setAtIndex(ValueLayout.JAVA_DOUBLE, i, val1 + val2);
        }
        return resArray;
    }

    public static NDArray subFloat(NDArray a,NDArray b,NDArray resArray){        
        long i=0;
        long loopbound=SPECIES.loopBound(a.size);

        for(;i<loopbound;i+=VL){
            var v1=FloatVector.fromMemorySegment(SPECIES,a.data,i*FLOAT_BYTES,ORDER);
            var v2=FloatVector.fromMemorySegment(SPECIES,b.data,i*FLOAT_BYTES,ORDER);
            var vRes=v1.sub(v2);
            vRes.intoMemorySegment(resArray.data,i*FLOAT_BYTES,ORDER);
        }

        for (; i < a.size; i++) {
            var val1 = a.data.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            var val2 = b.data.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            resArray.data.setAtIndex(ValueLayout.JAVA_FLOAT, i, val1 - val2);
        }
        return resArray;
    }

    public static NDArray subInt(NDArray a,NDArray b,NDArray resArray){        
        long i=0;
        long loopbound=SPECIES.loopBound(a.size);

        for(;i<loopbound;i+=INT_VL){
            var v1=IntVector.fromMemorySegment(SPECIESINT,a.data,i*INT_BYTES,ORDER);
            var v2=IntVector.fromMemorySegment(SPECIESINT,b.data,i*INT_BYTES,ORDER);
            var vRes=v1.sub(v2);
            vRes.intoMemorySegment(resArray.data,i*INT_BYTES,ORDER);
        }

        for (; i < a.size; i++) {
            var val1 = a.data.getAtIndex(ValueLayout.JAVA_INT, i);
            var val2 = b.data.getAtIndex(ValueLayout.JAVA_INT, i);
            resArray.data.setAtIndex(ValueLayout.JAVA_INT, i, val1 - val2);
        }
        return resArray;
    }

    public static NDArray subDouble(NDArray a,NDArray b,NDArray resArray){        
        long i=0;
        long loopbound=SPECIES.loopBound(a.size);

        for(;i<loopbound;i+=DB_VL){
            var v1=DoubleVector.fromMemorySegment(SPECIESDB,a.data,i*DB_BYTES,ORDER);
            var v2=DoubleVector.fromMemorySegment(SPECIESDB,b.data,i*DB_BYTES,ORDER);
            var vRes=v1.sub(v2);
            vRes.intoMemorySegment(resArray.data,i*DB_BYTES,ORDER);
        }

        for (; i < a.size; i++) {
            var val1 = a.data.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
            var val2 = b.data.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
            resArray.data.setAtIndex(ValueLayout.JAVA_DOUBLE, i, val1 - val2);
        }
        return resArray;
    }

    public static NDArray mulFloat(NDArray a,NDArray b,NDArray resArray){        
        long i=0;
        long loopbound=SPECIES.loopBound(a.size);

        for(;i<loopbound;i+=VL){
            var v1=FloatVector.fromMemorySegment(SPECIES,a.data,i*FLOAT_BYTES,ORDER);
            var v2=FloatVector.fromMemorySegment(SPECIES,b.data,i*FLOAT_BYTES,ORDER);
            var vRes=v1.mul(v2);
            vRes.intoMemorySegment(resArray.data,i*FLOAT_BYTES,ORDER);
        }

        for (; i < a.size; i++) {
            var val1 = a.data.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            var val2 = b.data.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            resArray.data.setAtIndex(ValueLayout.JAVA_FLOAT, i, val1 * val2);
        }
        return resArray;
    }

    public static NDArray mulInt(NDArray a,NDArray b,NDArray resArray){        
        long i=0;
        long loopbound=SPECIES.loopBound(a.size);

        for(;i<loopbound;i+=INT_VL){
            var v1=IntVector.fromMemorySegment(SPECIESINT,a.data,i*INT_BYTES,ORDER);
            var v2=IntVector.fromMemorySegment(SPECIESINT,b.data,i*INT_BYTES,ORDER);
            var vRes=v1.mul(v2);
            vRes.intoMemorySegment(resArray.data,i*INT_BYTES,ORDER);
        }

        for (; i < a.size; i++) {
            var val1 = a.data.getAtIndex(ValueLayout.JAVA_INT, i);
            var val2 = b.data.getAtIndex(ValueLayout.JAVA_INT, i);
            resArray.data.setAtIndex(ValueLayout.JAVA_INT, i, val1 * val2);
        }
        return resArray;
    }

    public static NDArray mulDouble(NDArray a,NDArray b,NDArray resArray){        
        long i=0;
        long loopbound=SPECIES.loopBound(a.size);

        for(;i<loopbound;i+=DB_VL){
            var v1=DoubleVector.fromMemorySegment(SPECIESDB,a.data,i*DB_BYTES,ORDER);
            var v2=DoubleVector.fromMemorySegment(SPECIESDB,b.data,i*DB_BYTES,ORDER);
            var vRes=v1.mul(v2);
            vRes.intoMemorySegment(resArray.data,i*DB_BYTES,ORDER);
        }

        for (; i < a.size; i++) {
            var val1 = a.data.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
            var val2 = b.data.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
            resArray.data.setAtIndex(ValueLayout.JAVA_DOUBLE, i, val1 * val2);
        }
        return resArray;
    }

    public static NDArray divFloat(NDArray a,NDArray b,NDArray resArray){        
        long i=0;
        long loopbound=SPECIES.loopBound(a.size);

        for(;i<loopbound;i+=VL){
            var v1=FloatVector.fromMemorySegment(SPECIES,a.data,i*FLOAT_BYTES,ORDER);
            var v2=FloatVector.fromMemorySegment(SPECIES,b.data,i*FLOAT_BYTES,ORDER);
            var vRes=v1.div(v2);
            vRes.intoMemorySegment(resArray.data,i*FLOAT_BYTES,ORDER);
        }

        for (; i < a.size; i++) {
            var val1 = a.data.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            var val2 = b.data.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            resArray.data.setAtIndex(ValueLayout.JAVA_FLOAT, i, val1 / val2);
        }
        return resArray;
    }

    public static NDArray divInt(NDArray a,NDArray b,NDArray resArray){        
        long i=0;
        long loopbound=SPECIES.loopBound(a.size);

        for(;i<loopbound;i+=INT_VL){
            var v1=IntVector.fromMemorySegment(SPECIESINT,a.data,i*INT_BYTES,ORDER);
            var v2=IntVector.fromMemorySegment(SPECIESINT,b.data,i*INT_BYTES,ORDER);
            var vRes=v1.div(v2);
            vRes.intoMemorySegment(resArray.data,i*INT_BYTES,ORDER);
        }

        for (; i < a.size; i++) {
            var val1 = a.data.getAtIndex(ValueLayout.JAVA_INT, i);
            var val2 = b.data.getAtIndex(ValueLayout.JAVA_INT, i);
            resArray.data.setAtIndex(ValueLayout.JAVA_INT, i, val1 / val2);
        }
        return resArray;
    }

    public static NDArray divDouble(NDArray a,NDArray b,NDArray resArray){        
        long i=0;
        long loopbound=SPECIES.loopBound(a.size);

        for(;i<loopbound;i+=DB_VL){
            var v1=DoubleVector.fromMemorySegment(SPECIESDB,a.data,i*DB_BYTES,ORDER);
            var v2=DoubleVector.fromMemorySegment(SPECIESDB,b.data,i*DB_BYTES,ORDER);
            var vRes=v1.div(v2);
            vRes.intoMemorySegment(resArray.data,i*DB_BYTES,ORDER);
        }

        for (; i < a.size; i++) {
            var val1 = a.data.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
            var val2 = b.data.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
            resArray.data.setAtIndex(ValueLayout.JAVA_DOUBLE, i, val1 / val2);
        }
        return resArray;
    }

    public static NDArray addFloat(NDArray a,float b,NDArray resArray){
        long i=0;
        long loopbound=SPECIES.loopBound(a.size);

        for(;i<loopbound;i+=VL){
            var v1=FloatVector.fromMemorySegment(SPECIES,a.data,i*FLOAT_BYTES,ORDER);
            var vRes=v1.add(b);
            vRes.intoMemorySegment(resArray.data,i*FLOAT_BYTES,ORDER);
        }

        for (; i < a.size; i++) {
            var val1 = a.data.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            resArray.data.setAtIndex(ValueLayout.JAVA_FLOAT, i, val1 + b);
        }
        return resArray;
    }

    public static NDArray addInt(NDArray a,int b,NDArray resArray){
        long i=0;
        long loopbound=SPECIES.loopBound(a.size);

        for(;i<loopbound;i+=INT_VL){
            var v1=IntVector.fromMemorySegment(SPECIESINT,a.data,i*INT_BYTES,ORDER);
            var vRes=v1.add(b);
            vRes.intoMemorySegment(resArray.data,i*INT_BYTES,ORDER);
        }

        for (; i < a.size; i++) {
            var val1 = a.data.getAtIndex(ValueLayout.JAVA_INT, i);
            resArray.data.setAtIndex(ValueLayout.JAVA_INT, i, val1 + b);
        }
        return resArray;
    }

    public static NDArray addDouble(NDArray a,double b,NDArray resArray){
        long i=0;
        long loopbound=SPECIES.loopBound(a.size);

        for(;i<loopbound;i+=DB_VL){
            var v1=DoubleVector.fromMemorySegment(SPECIESDB,a.data,i*DB_BYTES,ORDER);
            var vRes=v1.add(b);
            vRes.intoMemorySegment(resArray.data,i*DB_BYTES,ORDER);
        }

        for (; i < a.size; i++) {
            var val1 = a.data.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
            resArray.data.setAtIndex(ValueLayout.JAVA_DOUBLE, i, val1 + b);
        }
        return resArray;
    }

    public static NDArray subFloat(NDArray a,float b,NDArray resArray){
        long i=0;
        long loopbound=SPECIES.loopBound(a.size);

        for(;i<loopbound;i+=VL){
            var v1=FloatVector.fromMemorySegment(SPECIES,a.data,i*FLOAT_BYTES,ORDER);
            var vRes=v1.sub(b);
            vRes.intoMemorySegment(resArray.data,i*FLOAT_BYTES,ORDER);
        }

        for (; i < a.size; i++) {
            var val1 = a.data.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            resArray.data.setAtIndex(ValueLayout.JAVA_FLOAT, i, val1 - b);
        }
        return resArray;
    }

    public static NDArray subInt(NDArray a,int b,NDArray resArray){
        long i=0;
        long loopbound=SPECIES.loopBound(a.size);

        for(;i<loopbound;i+=INT_VL){
            var v1=IntVector.fromMemorySegment(SPECIESINT,a.data,i*INT_BYTES,ORDER);
            var vRes=v1.sub(b);
            vRes.intoMemorySegment(resArray.data,i*INT_BYTES,ORDER);
        }

        for (; i < a.size; i++) {
            var val1 = a.data.getAtIndex(ValueLayout.JAVA_INT, i);
            resArray.data.setAtIndex(ValueLayout.JAVA_INT, i, val1 - b);
        }
        return resArray;
    }

    public static NDArray subDouble(NDArray a,double b,NDArray resArray){
        long i=0;
        long loopbound=SPECIES.loopBound(a.size);

        for(;i<loopbound;i+=DB_VL){
            var v1=DoubleVector.fromMemorySegment(SPECIESDB,a.data,i*DB_BYTES,ORDER);
            var vRes=v1.sub(b);
            vRes.intoMemorySegment(resArray.data,i*DB_BYTES,ORDER);
        }

        for (; i < a.size; i++) {
            var val1 = a.data.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
            resArray.data.setAtIndex(ValueLayout.JAVA_DOUBLE, i, val1 - b);
        }
        return resArray;
    }

    public static NDArray mulFloat(NDArray a,float b,NDArray resArray){
        long i=0;
        long loopbound=SPECIES.loopBound(a.size);

        for(;i<loopbound;i+=VL){
            var v1=FloatVector.fromMemorySegment(SPECIES,a.data,i*FLOAT_BYTES,ORDER);
            var vRes=v1.mul(b);
            vRes.intoMemorySegment(resArray.data,i*FLOAT_BYTES,ORDER);
        }

        for (; i < a.size; i++) {
            var val1 = a.data.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            resArray.data.setAtIndex(ValueLayout.JAVA_FLOAT, i, val1 * b);
        }
        return resArray;
    }

    public static NDArray mulInt(NDArray a,int b,NDArray resArray){
        long i=0;
        long loopbound=SPECIES.loopBound(a.size);

        for(;i<loopbound;i+=INT_VL){
            var v1=IntVector.fromMemorySegment(SPECIESINT,a.data,i*INT_BYTES,ORDER);
            var vRes=v1.mul(b);
            vRes.intoMemorySegment(resArray.data,i*INT_BYTES,ORDER);
        }

        for (; i < a.size; i++) {
            var val1 = a.data.getAtIndex(ValueLayout.JAVA_INT, i);
            resArray.data.setAtIndex(ValueLayout.JAVA_INT, i, val1 * b);
        }
        return resArray;
    }

    public static NDArray mulDouble(NDArray a,double b,NDArray resArray){
        long i=0;
        long loopbound=SPECIES.loopBound(a.size);

        for(;i<loopbound;i+=DB_VL){
            var v1=DoubleVector.fromMemorySegment(SPECIESDB,a.data,i*DB_BYTES,ORDER);
            var vRes=v1.mul(b);
            vRes.intoMemorySegment(resArray.data,i*DB_BYTES,ORDER);
        }

        for (; i < a.size; i++) {
            var val1 = a.data.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
            resArray.data.setAtIndex(ValueLayout.JAVA_DOUBLE, i, val1 * b);
        }
        return resArray;
    }

    public static NDArray divFloat(NDArray a,float b,NDArray resArray){
        long i=0;
        long loopbound=SPECIES.loopBound(a.size);

        for(;i<loopbound;i+=VL){
            var v1=FloatVector.fromMemorySegment(SPECIES,a.data,i*FLOAT_BYTES,ORDER);
            var vRes=v1.div(b);
            vRes.intoMemorySegment(resArray.data,i*FLOAT_BYTES,ORDER);
        }

        for (; i < a.size; i++) {
            var val1 = a.data.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            resArray.data.setAtIndex(ValueLayout.JAVA_FLOAT, i, val1 / b);
        }
        return resArray;
    }

    public static NDArray divInt(NDArray a,int b,NDArray resArray){
        long i=0;
        long loopbound=SPECIES.loopBound(a.size);

        for(;i<loopbound;i+=INT_VL){
            var v1=IntVector.fromMemorySegment(SPECIESINT,a.data,i*INT_BYTES,ORDER);
            var vRes=v1.div(b);
            vRes.intoMemorySegment(resArray.data,i*INT_BYTES,ORDER);
        }

        for (; i < a.size; i++) {
            var val1 = a.data.getAtIndex(ValueLayout.JAVA_INT, i);
            resArray.data.setAtIndex(ValueLayout.JAVA_INT, i, val1 / b);
        }
        return resArray;
    }

    public static NDArray divDouble(NDArray a,double b,NDArray resArray){
        long i=0;
        long loopbound=SPECIES.loopBound(a.size);

        for(;i<loopbound;i+=DB_VL){
            var v1=DoubleVector.fromMemorySegment(SPECIESDB,a.data,i*DB_BYTES,ORDER);
            var vRes=v1.div(b);
            vRes.intoMemorySegment(resArray.data,i*DB_BYTES,ORDER);
        }

        for (; i < a.size; i++) {
            var val1 = a.data.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
            resArray.data.setAtIndex(ValueLayout.JAVA_DOUBLE, i, val1 / b);
        }
        return resArray;
    }

    public static float sum(NDArray a){
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
        return total;
    }

    public static float max(NDArray a){
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
        return finalMax;
    }

    public static float min(NDArray a){
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
        return finalMin;
    }

}