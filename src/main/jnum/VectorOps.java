package jnum;


import java.lang.classfile.constantpool.DoubleEntry;
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
    //this is utility class does not require instantability 
    private VectorOps(){
        throw new AssertionError();
    }


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
        long loopbound=SPECIESINT.loopBound(a.size);

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
        long loopbound=SPECIESDB.loopBound(a.size);

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
        long loopbound=SPECIESINT.loopBound(a.size);

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
        long loopbound=SPECIESDB.loopBound(a.size);

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
        long loopbound=SPECIESINT.loopBound(a.size);

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
        long loopbound=SPECIESDB.loopBound(a.size);

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
        long loopbound=SPECIESINT.loopBound(a.size);

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
        long loopbound=SPECIESDB.loopBound(a.size);

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
        long loopbound=SPECIESINT.loopBound(a.size);

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
        long loopbound=SPECIESDB.loopBound(a.size);

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
        long loopbound=SPECIESINT.loopBound(a.size);

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
        long loopbound=SPECIESDB.loopBound(a.size);

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
        long loopbound=SPECIESINT.loopBound(a.size);

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
        long loopbound=SPECIESDB.loopBound(a.size);

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
        long loopbound=SPECIESINT.loopBound(a.size);

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
        long loopbound=SPECIESDB.loopBound(a.size);

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

}