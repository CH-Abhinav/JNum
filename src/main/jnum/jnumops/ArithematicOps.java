package jnum.jnumops;

import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;
import jnum.NDArray;
import jdk.incubator.vector.VectorOperators;
public class ArithematicOps {
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
    private ArithematicOps(){
        throw new AssertionError();
    }


    public static NDArray addFloat(NDArray a,NDArray b,NDArray resArray){
        if(a.isContiguous() && b.isContiguous() && resArray.isContiguous()) return addFloatSIMD(a, b, resArray);
        else return addFloatStrides(a, b, resArray);
    }

    private static NDArray addFloatSIMD(NDArray a,NDArray b,NDArray resArray){
        long i=0;
        long loopbound=a.size - (a.size % (VL * 2));

        for(;i<loopbound;i+=VL*2){
            var va1=FloatVector.fromMemorySegment(SPECIES,a.data,i*FLOAT_BYTES,ORDER);
            var vb1=FloatVector.fromMemorySegment(SPECIES,b.data,i*FLOAT_BYTES,ORDER);
            var va2=FloatVector.fromMemorySegment(SPECIES, a.data, (i+VL)*FLOAT_BYTES, ORDER);
            var vb2=FloatVector.fromMemorySegment(SPECIES, b.data, (i+VL)*FLOAT_BYTES, ORDER);
            var vRes1=va1.add(vb1);
            var VRes2=va2.add(vb2);
            vRes1.intoMemorySegment(resArray.data,i*FLOAT_BYTES,ORDER);
            VRes2.intoMemorySegment(resArray.data, (i+VL)*FLOAT_BYTES, ORDER);
        }

        loopbound=SPECIES.loopBound(a.size);

        for(;i<loopbound;i+=VL){
            var va1=FloatVector.fromMemorySegment(SPECIES,a.data,i*FLOAT_BYTES,ORDER);
            var vb1=FloatVector.fromMemorySegment(SPECIES,b.data,i*FLOAT_BYTES,ORDER);
            var vRes=va1.add(vb1);
            vRes.intoMemorySegment(resArray.data,i*FLOAT_BYTES,ORDER);
        }

        for (; i < a.size; i++) {
            var val1 = a.data.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            var val2 = b.data.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            resArray.data.setAtIndex(ValueLayout.JAVA_FLOAT, i, val1 + val2);
        }
        return resArray;
    }

    private static NDArray addFloatStrides(NDArray a,NDArray b,NDArray resArray){
        for(long i=0;i<resArray.size;i++){
            long tempIndex = i;
            long[] coords = new long[resArray.shape.length];
            for (int d = resArray.shape.length - 1; d >= 0; d--) {
                coords[d] = tempIndex % resArray.shape[d];
                tempIndex = tempIndex / resArray.shape[d];
            }
            long flatA = 0, flatB = 0, flatRes = 0;
            for (int d = 0; d < resArray.shape.length; d++) {
                flatA += coords[d] * (long) a.strides[d];
                flatB += coords[d] * (long) b.strides[d];
                flatRes += coords[d] * (long) resArray.strides[d];
            }
            float valA = a.data.getAtIndex(ValueLayout.JAVA_FLOAT, flatA);
            float valB = b.data.getAtIndex(ValueLayout.JAVA_FLOAT, flatB);
            resArray.data.setAtIndex(ValueLayout.JAVA_FLOAT, flatRes, valA + valB);
        }
        return resArray;
    }

    public static NDArray addInt(NDArray a,NDArray b,NDArray resArray){
        if(a.isContiguous() && b.isContiguous() && resArray.isContiguous()) return addIntSIMD(a, b, resArray);
        else return addIntStrides(a, b, resArray);
    }

    private static NDArray addIntSIMD(NDArray a,NDArray b,NDArray resArray){
        long i=0;
        long loopbound=a.size - (a.size % (INT_VL * 2));

        for(;i<loopbound;i+=INT_VL*2){
            var v1a=IntVector.fromMemorySegment(SPECIESINT,a.data,i*INT_BYTES,ORDER);
            var v1b=IntVector.fromMemorySegment(SPECIESINT,b.data,i*INT_BYTES,ORDER);
            var v2a=IntVector.fromMemorySegment(SPECIESINT,a.data,(i+INT_VL)*INT_BYTES,ORDER);
            var v2b=IntVector.fromMemorySegment(SPECIESINT,b.data,(i+INT_VL)*INT_BYTES,ORDER);
            var vRes1=v1a.add(v1b);
            var vRes2=v2a.add(v2b);
            vRes1.intoMemorySegment(resArray.data,i*INT_BYTES,ORDER);
            vRes2.intoMemorySegment(resArray.data,(i+INT_VL)*INT_BYTES,ORDER);
        }

        loopbound=SPECIESINT.loopBound(a.size);

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

    private static NDArray addIntStrides(NDArray a,NDArray b,NDArray resArray){
        for(long i=0;i<resArray.size;i++){
            long tempIndex = i;
            long[] coords = new long[resArray.shape.length];
            for (int d = resArray.shape.length - 1; d >= 0; d--) {
                coords[d] = tempIndex % resArray.shape[d];
                tempIndex = tempIndex / resArray.shape[d];
            }
            long flatA = 0, flatB = 0, flatRes = 0;
            for (int d = 0; d < resArray.shape.length; d++) {
                flatA += coords[d] * (long) a.strides[d];
                flatB += coords[d] * (long) b.strides[d];
                flatRes += coords[d] * (long) resArray.strides[d];
            }
            int valA = a.data.getAtIndex(ValueLayout.JAVA_INT, flatA);
            int valB = b.data.getAtIndex(ValueLayout.JAVA_INT, flatB);
            resArray.data.setAtIndex(ValueLayout.JAVA_INT, flatRes, valA + valB);
        }
        return resArray;
    }

    public static NDArray addDouble(NDArray a,NDArray b,NDArray resArray){
        if(a.isContiguous() && b.isContiguous() && resArray.isContiguous()) return addDoubleSIMD(a, b, resArray);
        else return addDoubleStrides(a, b, resArray);
    }

    private static NDArray addDoubleSIMD(NDArray a,NDArray b,NDArray resArray){
        long i=0;
        long loopbound=a.size - (a.size % (DB_VL * 2));

        for(;i<loopbound;i+=DB_VL*2){
            var v1a=DoubleVector.fromMemorySegment(SPECIESDB,a.data,i*DB_BYTES,ORDER);
            var v1b=DoubleVector.fromMemorySegment(SPECIESDB,b.data,i*DB_BYTES,ORDER);
            var v2a=DoubleVector.fromMemorySegment(SPECIESDB,a.data,(i+DB_VL)*DB_BYTES,ORDER);
            var v2b=DoubleVector.fromMemorySegment(SPECIESDB,b.data,(i+DB_VL)*DB_BYTES,ORDER);
            var vRes1=v1a.add(v1b);
            var vRes2=v2a.add(v2b);
            vRes1.intoMemorySegment(resArray.data,i*DB_BYTES,ORDER);
            vRes2.intoMemorySegment(resArray.data,(i+DB_VL)*DB_BYTES,ORDER);
        }

        loopbound=SPECIESDB.loopBound(a.size);

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

    private static NDArray addDoubleStrides(NDArray a,NDArray b,NDArray resArray){
        for(long i=0;i<resArray.size;i++){
            long tempIndex = i;
            long[] coords = new long[resArray.shape.length];
            for (int d = resArray.shape.length - 1; d >= 0; d--) {
                coords[d] = tempIndex % resArray.shape[d];
                tempIndex = tempIndex / resArray.shape[d];
            }
            long flatA = 0, flatB = 0, flatRes = 0;
            for (int d = 0; d < resArray.shape.length; d++) {
                flatA += coords[d] * (long) a.strides[d];
                flatB += coords[d] * (long) b.strides[d];
                flatRes += coords[d] * (long) resArray.strides[d];
            }
            double valA = a.data.getAtIndex(ValueLayout.JAVA_DOUBLE, flatA);
            double valB = b.data.getAtIndex(ValueLayout.JAVA_DOUBLE, flatB);
            resArray.data.setAtIndex(ValueLayout.JAVA_DOUBLE, flatRes, valA + valB);
        }
        return resArray;
    }

    public static NDArray subFloat(NDArray a,NDArray b,NDArray resArray){       
        if(a.isContiguous() && b.isContiguous() && resArray.isContiguous()) return subFloatSIMD(a, b, resArray);
        else return subFloatStrides(a, b, resArray);
    }

    private static NDArray subFloatSIMD(NDArray a,NDArray b,NDArray resArray){
        long i=0;
        long loopbound=a.size - (a.size % (VL * 2));

        for(;i<loopbound;i+=VL*2){
            var v1a=FloatVector.fromMemorySegment(SPECIES,a.data,i*FLOAT_BYTES,ORDER);
            var v1b=FloatVector.fromMemorySegment(SPECIES,b.data,i*FLOAT_BYTES,ORDER);
            var v2a=FloatVector.fromMemorySegment(SPECIES,a.data,(i+VL)*FLOAT_BYTES,ORDER);
            var v2b=FloatVector.fromMemorySegment(SPECIES,b.data,(i+VL)*FLOAT_BYTES,ORDER);
            var vRes1=v1a.sub(v1b);
            var vRes2=v2a.sub(v2b);
            vRes1.intoMemorySegment(resArray.data,i*FLOAT_BYTES,ORDER);
            vRes2.intoMemorySegment(resArray.data,(i+VL)*FLOAT_BYTES,ORDER);
        }

        loopbound=SPECIES.loopBound(a.size);

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

    private static NDArray subFloatStrides(NDArray a,NDArray b,NDArray resArray){
        for(long i=0;i<resArray.size;i++){
            long tempIndex = i;
            long[] coords = new long[resArray.shape.length];
            for (int d = resArray.shape.length - 1; d >= 0; d--) {
                coords[d] = tempIndex % resArray.shape[d];
                tempIndex = tempIndex / resArray.shape[d];
            }
            long flatA = 0, flatB = 0, flatRes = 0;
            for (int d = 0; d < resArray.shape.length; d++) {
                flatA += coords[d] * (long) a.strides[d];
                flatB += coords[d] * (long) b.strides[d];
                flatRes += coords[d] * (long) resArray.strides[d];
            }
            float valA = a.data.getAtIndex(ValueLayout.JAVA_FLOAT, flatA);
            float valB = b.data.getAtIndex(ValueLayout.JAVA_FLOAT, flatB);
            resArray.data.setAtIndex(ValueLayout.JAVA_FLOAT, flatRes, valA - valB);
        }
        return resArray;
    }

    public static NDArray subInt(NDArray a,NDArray b,NDArray resArray){      
        if(a.isContiguous() && b.isContiguous() && resArray.isContiguous()) return subIntSIMD(a, b, resArray);
        else return subIntStrides(a, b, resArray);
    }

    private static NDArray subIntSIMD(NDArray a,NDArray b,NDArray resArray){
        long i=0;
        long loopbound=a.size - (a.size % (INT_VL * 2));

        for(;i<loopbound;i+=INT_VL*2){
            var v1a=IntVector.fromMemorySegment(SPECIESINT,a.data,i*INT_BYTES,ORDER);
            var v1b=IntVector.fromMemorySegment(SPECIESINT,b.data,i*INT_BYTES,ORDER);
            var v2a=IntVector.fromMemorySegment(SPECIESINT,a.data,(i+INT_VL)*INT_BYTES,ORDER);
            var v2b=IntVector.fromMemorySegment(SPECIESINT,b.data,(i+INT_VL)*INT_BYTES,ORDER);
            var vRes1=v1a.sub(v1b);
            var vRes2=v2a.sub(v2b);
            vRes1.intoMemorySegment(resArray.data,i*INT_BYTES,ORDER);
            vRes2.intoMemorySegment(resArray.data,(i+INT_VL)*INT_BYTES,ORDER);
        }

        loopbound=SPECIESINT.loopBound(a.size);

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

    private static NDArray subIntStrides(NDArray a,NDArray b,NDArray resArray){
        for(long i=0;i<resArray.size;i++){
            long tempIndex = i;
            long[] coords = new long[resArray.shape.length];
            for (int d = resArray.shape.length - 1; d >= 0; d--) {
                coords[d] = tempIndex % resArray.shape[d];
                tempIndex = tempIndex / resArray.shape[d];
            }
            long flatA = 0, flatB = 0, flatRes = 0;
            for (int d = 0; d < resArray.shape.length; d++) {
                flatA += coords[d] * (long) a.strides[d];
                flatB += coords[d] * (long) b.strides[d];
                flatRes += coords[d] * (long) resArray.strides[d];
            }
            int valA = a.data.getAtIndex(ValueLayout.JAVA_INT, flatA);
            int valB = b.data.getAtIndex(ValueLayout.JAVA_INT, flatB);
            resArray.data.setAtIndex(ValueLayout.JAVA_INT, flatRes, valA - valB);
        }
        return resArray;
    }

    public static NDArray subDouble(NDArray a,NDArray b,NDArray resArray){    
        if(a.isContiguous() && b.isContiguous() && resArray.isContiguous()) return subDoubleSIMD(a, b, resArray);
        else return subDoubleStrides(a, b, resArray);
    }

    private static NDArray subDoubleSIMD(NDArray a,NDArray b,NDArray resArray){
        long i=0;
        long loopbound=a.size - (a.size % (DB_VL * 2));

        for(;i<loopbound;i+=DB_VL*2){
            var v1a=DoubleVector.fromMemorySegment(SPECIESDB,a.data,i*DB_BYTES,ORDER);
            var v1b=DoubleVector.fromMemorySegment(SPECIESDB,b.data,i*DB_BYTES,ORDER);
            var v2a=DoubleVector.fromMemorySegment(SPECIESDB,a.data,(i+DB_VL)*DB_BYTES,ORDER);
            var v2b=DoubleVector.fromMemorySegment(SPECIESDB,b.data,(i+DB_VL)*DB_BYTES,ORDER);
            var vRes1=v1a.sub(v1b);
            var vRes2=v2a.sub(v2b);
            vRes1.intoMemorySegment(resArray.data,i*DB_BYTES,ORDER);
            vRes2.intoMemorySegment(resArray.data,(i+DB_VL)*DB_BYTES,ORDER);
        }

        loopbound=SPECIESDB.loopBound(a.size);

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

    private static NDArray subDoubleStrides(NDArray a,NDArray b,NDArray resArray){
        for(long i=0;i<resArray.size;i++){
            long tempIndex = i;
            long[] coords = new long[resArray.shape.length];
            for (int d = resArray.shape.length - 1; d >= 0; d--) {
                coords[d] = tempIndex % resArray.shape[d];
                tempIndex = tempIndex / resArray.shape[d];
            }
            long flatA = 0, flatB = 0, flatRes = 0;
            for (int d = 0; d < resArray.shape.length; d++) {
                flatA += coords[d] * (long) a.strides[d];
                flatB += coords[d] * (long) b.strides[d];
                flatRes += coords[d] * (long) resArray.strides[d];
            }
            double valA = a.data.getAtIndex(ValueLayout.JAVA_DOUBLE, flatA);
            double valB = b.data.getAtIndex(ValueLayout.JAVA_DOUBLE, flatB);
            resArray.data.setAtIndex(ValueLayout.JAVA_DOUBLE, flatRes, valA - valB);
        }
        return resArray;
    }

    public static NDArray mulFloat(NDArray a,NDArray b,NDArray resArray){     
        if(a.isContiguous() && b.isContiguous() && resArray.isContiguous()) return mulFloatSIMD(a, b, resArray);
        else return mulFloatStrides(a, b, resArray);
    }

    private static NDArray mulFloatSIMD(NDArray a,NDArray b,NDArray resArray){
        long i=0;
        long loopbound=a.size - (a.size % (VL * 2));

        for(;i<loopbound;i+=VL*2){
            var v1a=FloatVector.fromMemorySegment(SPECIES,a.data,i*FLOAT_BYTES,ORDER);
            var v1b=FloatVector.fromMemorySegment(SPECIES,b.data,i*FLOAT_BYTES,ORDER);
            var v2a=FloatVector.fromMemorySegment(SPECIES,a.data,(i+VL)*FLOAT_BYTES,ORDER);
            var v2b=FloatVector.fromMemorySegment(SPECIES,b.data,(i+VL)*FLOAT_BYTES,ORDER);
            var vRes1=v1a.mul(v1b);
            var vRes2=v2a.mul(v2b);
            vRes1.intoMemorySegment(resArray.data,i*FLOAT_BYTES,ORDER);
            vRes2.intoMemorySegment(resArray.data,(i+VL)*FLOAT_BYTES,ORDER);
        }

        loopbound=SPECIES.loopBound(a.size);

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

    private static NDArray mulFloatStrides(NDArray a,NDArray b,NDArray resArray){
        for(long i=0;i<resArray.size;i++){
            long tempIndex = i;
            long[] coords = new long[resArray.shape.length];
            for (int d = resArray.shape.length - 1; d >= 0; d--) {
                coords[d] = tempIndex % resArray.shape[d];
                tempIndex = tempIndex / resArray.shape[d];
            }
            long flatA = 0, flatB = 0, flatRes = 0;
            for (int d = 0; d < resArray.shape.length; d++) {
                flatA += coords[d] * (long) a.strides[d];
                flatB += coords[d] * (long) b.strides[d];
                flatRes += coords[d] * (long) resArray.strides[d];
            }
            float valA = a.data.getAtIndex(ValueLayout.JAVA_FLOAT, flatA);
            float valB = b.data.getAtIndex(ValueLayout.JAVA_FLOAT, flatB);
            resArray.data.setAtIndex(ValueLayout.JAVA_FLOAT, flatRes, valA * valB);
        }
        return resArray;
    }

    public static NDArray mulInt(NDArray a,NDArray b,NDArray resArray){  
        if(a.isContiguous() && b.isContiguous() && resArray.isContiguous()) return mulIntSIMD(a, b, resArray);
        else return mulIntStrides(a, b, resArray);
    }

    private static NDArray mulIntSIMD(NDArray a,NDArray b,NDArray resArray){
        long i=0;
        long loopbound=a.size - (a.size % (INT_VL * 2));

        for(;i<loopbound;i+=INT_VL*2){
            var v1a=IntVector.fromMemorySegment(SPECIESINT,a.data,i*INT_BYTES,ORDER);
            var v1b=IntVector.fromMemorySegment(SPECIESINT,b.data,i*INT_BYTES,ORDER);
            var v2a=IntVector.fromMemorySegment(SPECIESINT,a.data,(i+INT_VL)*INT_BYTES,ORDER);
            var v2b=IntVector.fromMemorySegment(SPECIESINT,b.data,(i+INT_VL)*INT_BYTES,ORDER);
            var vRes1=v1a.mul(v1b);
            var vRes2=v2a.mul(v2b);
            vRes1.intoMemorySegment(resArray.data,i*INT_BYTES,ORDER);
            vRes2.intoMemorySegment(resArray.data,(i+INT_VL)*INT_BYTES,ORDER);
        }

        loopbound=SPECIESINT.loopBound(a.size);

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

    private static NDArray mulIntStrides(NDArray a,NDArray b,NDArray resArray){
        for(long i=0;i<resArray.size;i++){
            long tempIndex = i;
            long[] coords = new long[resArray.shape.length];
            for (int d = resArray.shape.length - 1; d >= 0; d--) {
                coords[d] = tempIndex % resArray.shape[d];
                tempIndex = tempIndex / resArray.shape[d];
            }
            long flatA = 0, flatB = 0, flatRes = 0;
            for (int d = 0; d < resArray.shape.length; d++) {
                flatA += coords[d] * (long) a.strides[d];
                flatB += coords[d] * (long) b.strides[d];
                flatRes += coords[d] * (long) resArray.strides[d];
            }
            int valA = a.data.getAtIndex(ValueLayout.JAVA_INT, flatA);
            int valB = b.data.getAtIndex(ValueLayout.JAVA_INT, flatB);
            resArray.data.setAtIndex(ValueLayout.JAVA_INT, flatRes, valA * valB);
        }
        return resArray;
    }

    public static NDArray mulDouble(NDArray a,NDArray b,NDArray resArray){        
        if(a.isContiguous() && b.isContiguous() && resArray.isContiguous()) return mulDoubleSIMD(a, b, resArray);
        else return mulDoubleStrides(a, b, resArray);
    }

    private static NDArray mulDoubleSIMD(NDArray a,NDArray b,NDArray resArray){
        long i=0;
        long loopbound=a.size - (a.size % (DB_VL * 2));

        for(;i<loopbound;i+=DB_VL*2){
            var v1a=DoubleVector.fromMemorySegment(SPECIESDB,a.data,i*DB_BYTES,ORDER);
            var v1b=DoubleVector.fromMemorySegment(SPECIESDB,b.data,i*DB_BYTES,ORDER);
            var v2a=DoubleVector.fromMemorySegment(SPECIESDB,a.data,(i+DB_VL)*DB_BYTES,ORDER);
            var v2b=DoubleVector.fromMemorySegment(SPECIESDB,b.data,(i+DB_VL)*DB_BYTES,ORDER);
            var vRes1=v1a.mul(v1b);
            var vRes2=v2a.mul(v2b);
            vRes1.intoMemorySegment(resArray.data,i*DB_BYTES,ORDER);
            vRes2.intoMemorySegment(resArray.data,(i+DB_VL)*DB_BYTES,ORDER);
        }

        loopbound=SPECIESDB.loopBound(a.size);

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

    private static NDArray mulDoubleStrides(NDArray a,NDArray b,NDArray resArray){
        for(long i=0;i<resArray.size;i++){
            long tempIndex = i;
            long[] coords = new long[resArray.shape.length];
            for (int d = resArray.shape.length - 1; d >= 0; d--) {
                coords[d] = tempIndex % resArray.shape[d];
                tempIndex = tempIndex / resArray.shape[d];
            }
            long flatA = 0, flatB = 0, flatRes = 0;
            for (int d = 0; d < resArray.shape.length; d++) {
                flatA += coords[d] * (long) a.strides[d];
                flatB += coords[d] * (long) b.strides[d];
                flatRes += coords[d] * (long) resArray.strides[d];
            }
            double valA = a.data.getAtIndex(ValueLayout.JAVA_DOUBLE, flatA);
            double valB = b.data.getAtIndex(ValueLayout.JAVA_DOUBLE, flatB);
            resArray.data.setAtIndex(ValueLayout.JAVA_DOUBLE, flatRes, valA * valB);
        }
        return resArray;
    }

    public static NDArray divFloat(NDArray a,NDArray b,NDArray resArray){       
        if(a.isContiguous() && b.isContiguous() && resArray.isContiguous()) return divFloatSIMD(a, b, resArray);
        else return divFloatStrides(a, b, resArray);
    }

    private static NDArray divFloatSIMD(NDArray a,NDArray b,NDArray resArray){
        long i=0;
        long loopbound=a.size - (a.size % (VL * 2));

        for(;i<loopbound;i+=VL*2){
            var v1a=FloatVector.fromMemorySegment(SPECIES,a.data,i*FLOAT_BYTES,ORDER);
            var v1b=FloatVector.fromMemorySegment(SPECIES,b.data,i*FLOAT_BYTES,ORDER);
            var v2a=FloatVector.fromMemorySegment(SPECIES,a.data,(i+VL)*FLOAT_BYTES,ORDER);
            var v2b=FloatVector.fromMemorySegment(SPECIES,b.data,(i+VL)*FLOAT_BYTES,ORDER);
            var vRes1=v1a.div(v1b);
            var vRes2=v2a.div(v2b);
            vRes1.intoMemorySegment(resArray.data,i*FLOAT_BYTES,ORDER);
            vRes2.intoMemorySegment(resArray.data,(i+VL)*FLOAT_BYTES,ORDER);
        }

        loopbound=SPECIES.loopBound(a.size);

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

    private static NDArray divFloatStrides(NDArray a,NDArray b,NDArray resArray){
        for(long i=0;i<resArray.size;i++){
            long tempIndex = i;
            long[] coords = new long[resArray.shape.length];
            for (int d = resArray.shape.length - 1; d >= 0; d--) {
                coords[d] = tempIndex % resArray.shape[d];
                tempIndex = tempIndex / resArray.shape[d];
            }
            long flatA = 0, flatB = 0, flatRes = 0;
            for (int d = 0; d < resArray.shape.length; d++) {
                flatA += coords[d] * (long) a.strides[d];
                flatB += coords[d] * (long) b.strides[d];
                flatRes += coords[d] * (long) resArray.strides[d];
            }
            float valA = a.data.getAtIndex(ValueLayout.JAVA_FLOAT, flatA);
            float valB = b.data.getAtIndex(ValueLayout.JAVA_FLOAT, flatB);
            resArray.data.setAtIndex(ValueLayout.JAVA_FLOAT, flatRes, valA / valB);
        }
        return resArray;
    }

    public static NDArray divInt(NDArray a,NDArray b,NDArray resArray){       
        if(a.isContiguous() && b.isContiguous() && resArray.isContiguous()) return divIntSIMD(a, b, resArray);
        else return divIntStrides(a, b, resArray);
    }

    private static NDArray divIntSIMD(NDArray a,NDArray b,NDArray resArray){
        long i=0;
        long loopbound=a.size - (a.size % (INT_VL * 2));

        for(;i<loopbound;i+=INT_VL*2){
            var v1a=IntVector.fromMemorySegment(SPECIESINT,a.data,i*INT_BYTES,ORDER);
            var v1b=IntVector.fromMemorySegment(SPECIESINT,b.data,i*INT_BYTES,ORDER);
            var v2a=IntVector.fromMemorySegment(SPECIESINT,a.data,(i+INT_VL)*INT_BYTES,ORDER);
            var v2b=IntVector.fromMemorySegment(SPECIESINT,b.data,(i+INT_VL)*INT_BYTES,ORDER);
            var vRes1=v1a.div(v1b);
            var vRes2=v2a.div(v2b);
            vRes1.intoMemorySegment(resArray.data,i*INT_BYTES,ORDER);
            vRes2.intoMemorySegment(resArray.data,(i+INT_VL)*INT_BYTES,ORDER);
        }

        loopbound=SPECIESINT.loopBound(a.size);

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

    private static NDArray divIntStrides(NDArray a,NDArray b,NDArray resArray){
        for(long i=0;i<resArray.size;i++){
            long tempIndex = i;
            long[] coords = new long[resArray.shape.length];
            for (int d = resArray.shape.length - 1; d >= 0; d--) {
                coords[d] = tempIndex % resArray.shape[d];
                tempIndex = tempIndex / resArray.shape[d];
            }
            long flatA = 0, flatB = 0, flatRes = 0;
            for (int d = 0; d < resArray.shape.length; d++) {
                flatA += coords[d] * (long) a.strides[d];
                flatB += coords[d] * (long) b.strides[d];
                flatRes += coords[d] * (long) resArray.strides[d];
            }
            int valA = a.data.getAtIndex(ValueLayout.JAVA_INT, flatA);
            int valB = b.data.getAtIndex(ValueLayout.JAVA_INT, flatB);
            resArray.data.setAtIndex(ValueLayout.JAVA_INT, flatRes, valA / valB);
        }
        return resArray;
    }

    public static NDArray divDouble(NDArray a,NDArray b,NDArray resArray){      
        if(a.isContiguous() && b.isContiguous() && resArray.isContiguous()) return divDoubleSIMD(a, b, resArray);
        else return divDoubleStrides(a, b, resArray);
    }

    private static NDArray divDoubleSIMD(NDArray a,NDArray b,NDArray resArray){
        long i=0;
        long loopbound=a.size - (a.size % (DB_VL * 2));

        for(;i<loopbound;i+=DB_VL*2){
            var v1a=DoubleVector.fromMemorySegment(SPECIESDB,a.data,i*DB_BYTES,ORDER);
            var v1b=DoubleVector.fromMemorySegment(SPECIESDB,b.data,i*DB_BYTES,ORDER);
            var v2a=DoubleVector.fromMemorySegment(SPECIESDB,a.data,(i+DB_VL)*DB_BYTES,ORDER);
            var v2b=DoubleVector.fromMemorySegment(SPECIESDB,b.data,(i+DB_VL)*DB_BYTES,ORDER);
            var vRes1=v1a.div(v1b);
            var vRes2=v2a.div(v2b);
            vRes1.intoMemorySegment(resArray.data,i*DB_BYTES,ORDER);
            vRes2.intoMemorySegment(resArray.data,(i+DB_VL)*DB_BYTES,ORDER);
        }

        loopbound=SPECIESDB.loopBound(a.size);

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

    private static NDArray divDoubleStrides(NDArray a,NDArray b,NDArray resArray){
        for(long i=0;i<resArray.size;i++){
            long tempIndex = i;
            long[] coords = new long[resArray.shape.length];
            for (int d = resArray.shape.length - 1; d >= 0; d--) {
                coords[d] = tempIndex % resArray.shape[d];
                tempIndex = tempIndex / resArray.shape[d];
            }
            long flatA = 0, flatB = 0, flatRes = 0;
            for (int d = 0; d < resArray.shape.length; d++) {
                flatA += coords[d] * (long) a.strides[d];
                flatB += coords[d] * (long) b.strides[d];
                flatRes += coords[d] * (long) resArray.strides[d];
            }
            double valA = a.data.getAtIndex(ValueLayout.JAVA_DOUBLE, flatA);
            double valB = b.data.getAtIndex(ValueLayout.JAVA_DOUBLE, flatB);
            resArray.data.setAtIndex(ValueLayout.JAVA_DOUBLE, flatRes, valA / valB);
        }
        return resArray;
    }

    public static NDArray addFloat(NDArray a,float b,NDArray resArray){
        if(a.isContiguous() && resArray.isContiguous()) return addFloatSIMD(a, b, resArray);
        else return addFloatStrides(a, b, resArray);
    }

    private static NDArray addFloatSIMD(NDArray a,float b,NDArray resArray){
        long i=0;
        long loopbound=a.size - (a.size % (VL * 2));

        for(;i<loopbound;i+=VL*2){
            var v1=FloatVector.fromMemorySegment(SPECIES,a.data,i*FLOAT_BYTES,ORDER);
            var v2=FloatVector.fromMemorySegment(SPECIES,a.data,(i+VL)*FLOAT_BYTES,ORDER);
            var vRes1=v1.add(b);
            var vRes2=v2.add(b);
            vRes1.intoMemorySegment(resArray.data,i*FLOAT_BYTES,ORDER);
            vRes2.intoMemorySegment(resArray.data,(i+VL)*FLOAT_BYTES,ORDER);
        }

        loopbound=SPECIES.loopBound(a.size);

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

    private static NDArray addFloatStrides(NDArray a,float b,NDArray resArray){
        for(long i=0;i<resArray.size;i++){
            long tempIndex = i;
            long[] coords = new long[resArray.shape.length];
            for (int d = resArray.shape.length - 1; d >= 0; d--) {
                coords[d] = tempIndex % resArray.shape[d];
                tempIndex = tempIndex / resArray.shape[d];
            }
            long flatA = 0, flatRes = 0;
            for (int d = 0; d < resArray.shape.length; d++) {
                flatA += coords[d] * (long) a.strides[d];
                flatRes += coords[d] * (long) resArray.strides[d];
            }
            float valA = a.data.getAtIndex(ValueLayout.JAVA_FLOAT, flatA);
            resArray.data.setAtIndex(ValueLayout.JAVA_FLOAT, flatRes, valA + b);
        }
        return resArray;
    }

    public static NDArray addInt(NDArray a,int b,NDArray resArray){
        if(a.isContiguous() && resArray.isContiguous()) return addIntSIMD(a, b, resArray);
        else return addIntStrides(a, b, resArray);
    }

    private static NDArray addIntSIMD(NDArray a,int b,NDArray resArray){
        long i=0;
        long loopbound=a.size - (a.size % (INT_VL * 2));

        for(;i<loopbound;i+=INT_VL*2){
            var v1=IntVector.fromMemorySegment(SPECIESINT,a.data,i*INT_BYTES,ORDER);
            var v2=IntVector.fromMemorySegment(SPECIESINT,a.data,(i+INT_VL)*INT_BYTES,ORDER);
            var vRes1=v1.add(b);
            var vRes2=v2.add(b);
            vRes1.intoMemorySegment(resArray.data,i*INT_BYTES,ORDER);
            vRes2.intoMemorySegment(resArray.data,(i+INT_VL)*INT_BYTES,ORDER);
        }

        loopbound=SPECIESINT.loopBound(a.size);

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

    private static NDArray addIntStrides(NDArray a,int b,NDArray resArray){
        for(long i=0;i<resArray.size;i++){
            long tempIndex = i;
            long[] coords = new long[resArray.shape.length];
            for (int d = resArray.shape.length - 1; d >= 0; d--) {
                coords[d] = tempIndex % resArray.shape[d];
                tempIndex = tempIndex / resArray.shape[d];
            }
            long flatA = 0, flatRes = 0;
            for (int d = 0; d < resArray.shape.length; d++) {
                flatA += coords[d] * (long) a.strides[d];
                flatRes += coords[d] * (long) resArray.strides[d];
            }
            int valA = a.data.getAtIndex(ValueLayout.JAVA_INT, flatA);
            resArray.data.setAtIndex(ValueLayout.JAVA_INT, flatRes, valA + b);
        }
        return resArray;
    }

    public static NDArray addDouble(NDArray a,double b,NDArray resArray){
        if(a.isContiguous() && resArray.isContiguous()) return addDoubleSIMD(a, b, resArray);
        else return addDoubleStrides(a, b, resArray);
    }

    private static NDArray addDoubleSIMD(NDArray a,double b,NDArray resArray){
        long i=0;
        long loopbound=a.size - (a.size % (DB_VL * 2));

        for(;i<loopbound;i+=DB_VL*2){
            var v1=DoubleVector.fromMemorySegment(SPECIESDB,a.data,i*DB_BYTES,ORDER);
            var v2=DoubleVector.fromMemorySegment(SPECIESDB,a.data,(i+DB_VL)*DB_BYTES,ORDER);
            var vRes1=v1.add(b);
            var vRes2=v2.add(b);
            vRes1.intoMemorySegment(resArray.data,i*DB_BYTES,ORDER);
            vRes2.intoMemorySegment(resArray.data,(i+DB_VL)*DB_BYTES,ORDER);
        }

        loopbound=SPECIESDB.loopBound(a.size);

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

    private static NDArray addDoubleStrides(NDArray a,double b,NDArray resArray){
        for(long i=0;i<resArray.size;i++){
            long tempIndex = i;
            long[] coords = new long[resArray.shape.length];
            for (int d = resArray.shape.length - 1; d >= 0; d--) {
                coords[d] = tempIndex % resArray.shape[d];
                tempIndex = tempIndex / resArray.shape[d];
            }
            long flatA = 0, flatRes = 0;
            for (int d = 0; d < resArray.shape.length; d++) {
                flatA += coords[d] * (long) a.strides[d];
                flatRes += coords[d] * (long) resArray.strides[d];
            }
            double valA = a.data.getAtIndex(ValueLayout.JAVA_DOUBLE, flatA);
            resArray.data.setAtIndex(ValueLayout.JAVA_DOUBLE, flatRes, valA + b);
        }
        return resArray;
    }

    public static NDArray subFloat(NDArray a,float b,NDArray resArray){
        if(a.isContiguous() && resArray.isContiguous()) return subFloatSIMD(a, b, resArray);
        else return subFloatStrides(a, b, resArray);
    }

    private static NDArray subFloatSIMD(NDArray a,float b,NDArray resArray){
        long i=0;
        long loopbound=a.size - (a.size % (VL * 2));

        for(;i<loopbound;i+=VL*2){
            var v1=FloatVector.fromMemorySegment(SPECIES,a.data,i*FLOAT_BYTES,ORDER);
            var v2=FloatVector.fromMemorySegment(SPECIES,a.data,(i+VL)*FLOAT_BYTES,ORDER);
            var vRes1=v1.sub(b);
            var vRes2=v2.sub(b);
            vRes1.intoMemorySegment(resArray.data,i*FLOAT_BYTES,ORDER);
            vRes2.intoMemorySegment(resArray.data,(i+VL)*FLOAT_BYTES,ORDER);
        }

        loopbound=SPECIES.loopBound(a.size);

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

    private static NDArray subFloatStrides(NDArray a,float b,NDArray resArray){
        for(long i=0;i<resArray.size;i++){
            long tempIndex = i;
            long[] coords = new long[resArray.shape.length];
            for (int d = resArray.shape.length - 1; d >= 0; d--) {
                coords[d] = tempIndex % resArray.shape[d];
                tempIndex = tempIndex / resArray.shape[d];
            }
            long flatA = 0, flatRes = 0;
            for (int d = 0; d < resArray.shape.length; d++) {
                flatA += coords[d] * (long) a.strides[d];
                flatRes += coords[d] * (long) resArray.strides[d];
            }
            float valA = a.data.getAtIndex(ValueLayout.JAVA_FLOAT, flatA);
            resArray.data.setAtIndex(ValueLayout.JAVA_FLOAT, flatRes, valA - b);
        }
        return resArray;
    }

    public static NDArray subInt(NDArray a,int b,NDArray resArray){
        if(a.isContiguous() && resArray.isContiguous()) return subIntSIMD(a, b, resArray);
        else return subIntStrides(a, b, resArray);
    }

    private static NDArray subIntSIMD(NDArray a,int b,NDArray resArray){
        long i=0;
        long loopbound=a.size - (a.size % (INT_VL * 2));

        for(;i<loopbound;i+=INT_VL*2){
            var v1=IntVector.fromMemorySegment(SPECIESINT,a.data,i*INT_BYTES,ORDER);
            var v2=IntVector.fromMemorySegment(SPECIESINT,a.data,(i+INT_VL)*INT_BYTES,ORDER);
            var vRes1=v1.sub(b);
            var vRes2=v2.sub(b);
            vRes1.intoMemorySegment(resArray.data,i*INT_BYTES,ORDER);
            vRes2.intoMemorySegment(resArray.data,(i+INT_VL)*INT_BYTES,ORDER);
        }

        loopbound=SPECIESINT.loopBound(a.size);

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

    private static NDArray subIntStrides(NDArray a,int b,NDArray resArray){
        for(long i=0;i<resArray.size;i++){
            long tempIndex = i;
            long[] coords = new long[resArray.shape.length];
            for (int d = resArray.shape.length - 1; d >= 0; d--) {
                coords[d] = tempIndex % resArray.shape[d];
                tempIndex = tempIndex / resArray.shape[d];
            }
            long flatA = 0, flatRes = 0;
            for (int d = 0; d < resArray.shape.length; d++) {
                flatA += coords[d] * (long) a.strides[d];
                flatRes += coords[d] * (long) resArray.strides[d];
            }
            int valA = a.data.getAtIndex(ValueLayout.JAVA_INT, flatA);
            resArray.data.setAtIndex(ValueLayout.JAVA_INT, flatRes, valA - b);
        }
        return resArray;
    }

    public static NDArray subDouble(NDArray a,double b,NDArray resArray){
        if(a.isContiguous() && resArray.isContiguous()) return subDoubleSIMD(a, b, resArray);
        else return subDoubleStrides(a, b, resArray);
    }

    private static NDArray subDoubleSIMD(NDArray a,double b,NDArray resArray){
        long i=0;
        long loopbound=a.size - (a.size % (DB_VL * 2));

        for(;i<loopbound;i+=DB_VL*2){
            var v1=DoubleVector.fromMemorySegment(SPECIESDB,a.data,i*DB_BYTES,ORDER);
            var v2=DoubleVector.fromMemorySegment(SPECIESDB,a.data,(i+DB_VL)*DB_BYTES,ORDER);
            var vRes1=v1.sub(b);
            var vRes2=v2.sub(b);
            vRes1.intoMemorySegment(resArray.data,i*DB_BYTES,ORDER);
            vRes2.intoMemorySegment(resArray.data,(i+DB_VL)*DB_BYTES,ORDER);
        }

        loopbound=SPECIESDB.loopBound(a.size);

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

    private static NDArray subDoubleStrides(NDArray a,double b,NDArray resArray){
        for(long i=0;i<resArray.size;i++){
            long tempIndex = i;
            long[] coords = new long[resArray.shape.length];
            for (int d = resArray.shape.length - 1; d >= 0; d--) {
                coords[d] = tempIndex % resArray.shape[d];
                tempIndex = tempIndex / resArray.shape[d];
            }
            long flatA = 0, flatRes = 0;
            for (int d = 0; d < resArray.shape.length; d++) {
                flatA += coords[d] * (long) a.strides[d];
                flatRes += coords[d] * (long) resArray.strides[d];
            }
            double valA = a.data.getAtIndex(ValueLayout.JAVA_DOUBLE, flatA);
            resArray.data.setAtIndex(ValueLayout.JAVA_DOUBLE, flatRes, valA - b);
        }
        return resArray;
    }

    public static NDArray mulFloat(NDArray a,float b,NDArray resArray){
        if(a.isContiguous() && resArray.isContiguous()) return mulFloatSIMD(a, b, resArray);
        else return mulFloatStrides(a, b, resArray);
    }

    private static NDArray mulFloatSIMD(NDArray a,float b,NDArray resArray){
        long i=0;
        long loopbound=a.size - (a.size % (VL * 2));

        for(;i<loopbound;i+=VL*2){
            var v1=FloatVector.fromMemorySegment(SPECIES,a.data,i*FLOAT_BYTES,ORDER);
            var v2=FloatVector.fromMemorySegment(SPECIES,a.data,(i+VL)*FLOAT_BYTES,ORDER);
            var vRes1=v1.mul(b);
            var vRes2=v2.mul(b);
            vRes1.intoMemorySegment(resArray.data,i*FLOAT_BYTES,ORDER);
            vRes2.intoMemorySegment(resArray.data,(i+VL)*FLOAT_BYTES,ORDER);
        }

        loopbound=SPECIES.loopBound(a.size);

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

    private static NDArray mulFloatStrides(NDArray a,float b,NDArray resArray){
        for(long i=0;i<resArray.size;i++){
            long tempIndex = i;
            long[] coords = new long[resArray.shape.length];
            for (int d = resArray.shape.length - 1; d >= 0; d--) {
                coords[d] = tempIndex % resArray.shape[d];
                tempIndex = tempIndex / resArray.shape[d];
            }
            long flatA = 0, flatRes = 0;
            for (int d = 0; d < resArray.shape.length; d++) {
                flatA += coords[d] * (long) a.strides[d];
                flatRes += coords[d] * (long) resArray.strides[d];
            }
            float valA = a.data.getAtIndex(ValueLayout.JAVA_FLOAT, flatA);
            resArray.data.setAtIndex(ValueLayout.JAVA_FLOAT, flatRes, valA * b);
        }
        return resArray;
    }

    public static NDArray mulInt(NDArray a,int b,NDArray resArray){
        if(a.isContiguous() && resArray.isContiguous()) return mulIntSIMD(a, b, resArray);
        else return mulIntStrides(a, b, resArray);
    }

    private static NDArray mulIntSIMD(NDArray a,int b,NDArray resArray){
        long i=0;
        long loopbound=a.size - (a.size % (INT_VL * 2));

        for(;i<loopbound;i+=INT_VL*2){
            var v1=IntVector.fromMemorySegment(SPECIESINT,a.data,i*INT_BYTES,ORDER);
            var v2=IntVector.fromMemorySegment(SPECIESINT,a.data,(i+INT_VL)*INT_BYTES,ORDER);
            var vRes1=v1.mul(b);
            var vRes2=v2.mul(b);
            vRes1.intoMemorySegment(resArray.data,i*INT_BYTES,ORDER);
            vRes2.intoMemorySegment(resArray.data,(i+INT_VL)*INT_BYTES,ORDER);
        }

        loopbound=SPECIESINT.loopBound(a.size);

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

    private static NDArray mulIntStrides(NDArray a,int b,NDArray resArray){
        for(long i=0;i<resArray.size;i++){
            long tempIndex = i;
            long[] coords = new long[resArray.shape.length];
            for (int d = resArray.shape.length - 1; d >= 0; d--) {
                coords[d] = tempIndex % resArray.shape[d];
                tempIndex = tempIndex / resArray.shape[d];
            }
            long flatA = 0, flatRes = 0;
            for (int d = 0; d < resArray.shape.length; d++) {
                flatA += coords[d] * (long) a.strides[d];
                flatRes += coords[d] * (long) resArray.strides[d];
            }
            int valA = a.data.getAtIndex(ValueLayout.JAVA_INT, flatA);
            resArray.data.setAtIndex(ValueLayout.JAVA_INT, flatRes, valA * b);
        }
        return resArray;
    }

    public static NDArray mulDouble(NDArray a,double b,NDArray resArray){
        if(a.isContiguous() && resArray.isContiguous()) return mulDoubleSIMD(a, b, resArray);
        else return mulDoubleStrides(a, b, resArray);
    }

    private static NDArray mulDoubleSIMD(NDArray a,double b,NDArray resArray){
        long i=0;
        long loopbound=a.size - (a.size % (DB_VL * 2));

        for(;i<loopbound;i+=DB_VL*2){
            var v1=DoubleVector.fromMemorySegment(SPECIESDB,a.data,i*DB_BYTES,ORDER);
            var v2=DoubleVector.fromMemorySegment(SPECIESDB,a.data,(i+DB_VL)*DB_BYTES,ORDER);
            var vRes1=v1.mul(b);
            var vRes2=v2.mul(b);
            vRes1.intoMemorySegment(resArray.data,i*DB_BYTES,ORDER);
            vRes2.intoMemorySegment(resArray.data,(i+DB_VL)*DB_BYTES,ORDER);
        }

        loopbound=SPECIESDB.loopBound(a.size);

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

    private static NDArray mulDoubleStrides(NDArray a,double b,NDArray resArray){
        for(long i=0;i<resArray.size;i++){
            long tempIndex = i;
            long[] coords = new long[resArray.shape.length];
            for (int d = resArray.shape.length - 1; d >= 0; d--) {
                coords[d] = tempIndex % resArray.shape[d];
                tempIndex = tempIndex / resArray.shape[d];
            }
            long flatA = 0, flatRes = 0;
            for (int d = 0; d < resArray.shape.length; d++) {
                flatA += coords[d] * (long) a.strides[d];
                flatRes += coords[d] * (long) resArray.strides[d];
            }
            double valA = a.data.getAtIndex(ValueLayout.JAVA_DOUBLE, flatA);
            resArray.data.setAtIndex(ValueLayout.JAVA_DOUBLE, flatRes, valA * b);
        }
        return resArray;
    }

    public static NDArray divFloat(NDArray a,float b,NDArray resArray){
        if(a.isContiguous() && resArray.isContiguous()) return divFloatSIMD(a, b, resArray);
        else return divFloatStrides(a, b, resArray);
    }

    private static NDArray divFloatSIMD(NDArray a,float b,NDArray resArray){
        long i=0;
        long loopbound=a.size - (a.size % (VL * 2));

        for(;i<loopbound;i+=VL*2){
            var v1=FloatVector.fromMemorySegment(SPECIES,a.data,i*FLOAT_BYTES,ORDER);
            var v2=FloatVector.fromMemorySegment(SPECIES,a.data,(i+VL)*FLOAT_BYTES,ORDER);
            var vRes1=v1.div(b);
            var vRes2=v2.div(b);
            vRes1.intoMemorySegment(resArray.data,i*FLOAT_BYTES,ORDER);
            vRes2.intoMemorySegment(resArray.data,(i+VL)*FLOAT_BYTES,ORDER);
        }

        loopbound=SPECIES.loopBound(a.size);

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

    private static NDArray divFloatStrides(NDArray a,float b,NDArray resArray){
        for(long i=0;i<resArray.size;i++){
            long tempIndex = i;
            long[] coords = new long[resArray.shape.length];
            for (int d = resArray.shape.length - 1; d >= 0; d--) {
                coords[d] = tempIndex % resArray.shape[d];
                tempIndex = tempIndex / resArray.shape[d];
            }
            long flatA = 0, flatRes = 0;
            for (int d = 0; d < resArray.shape.length; d++) {
                flatA += coords[d] * (long) a.strides[d];
                flatRes += coords[d] * (long) resArray.strides[d];
            }
            float valA = a.data.getAtIndex(ValueLayout.JAVA_FLOAT, flatA);
            resArray.data.setAtIndex(ValueLayout.JAVA_FLOAT, flatRes, valA / b);
        }
        return resArray;
    }

    public static NDArray divInt(NDArray a,int b,NDArray resArray){
        if(a.isContiguous() && resArray.isContiguous()) return divIntSIMD(a, b, resArray);
        else return divIntStrides(a, b, resArray);
    }

    private static NDArray divIntSIMD(NDArray a,int b,NDArray resArray){
        long i=0;
        long loopbound=a.size - (a.size % (INT_VL * 2));

        for(;i<loopbound;i+=INT_VL*2){
            var v1=IntVector.fromMemorySegment(SPECIESINT,a.data,i*INT_BYTES,ORDER);
            var v2=IntVector.fromMemorySegment(SPECIESINT,a.data,(i+INT_VL)*INT_BYTES,ORDER);
            var vRes1=v1.div(b);
            var vRes2=v2.div(b);
            vRes1.intoMemorySegment(resArray.data,i*INT_BYTES,ORDER);
            vRes2.intoMemorySegment(resArray.data,(i+INT_VL)*INT_BYTES,ORDER);
        }

        loopbound=SPECIESINT.loopBound(a.size);

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

    private static NDArray divIntStrides(NDArray a,int b,NDArray resArray){
        for(long i=0;i<resArray.size;i++){
            long tempIndex = i;
            long[] coords = new long[resArray.shape.length];
            for (int d = resArray.shape.length - 1; d >= 0; d--) {
                coords[d] = tempIndex % resArray.shape[d];
                tempIndex = tempIndex / resArray.shape[d];
            }
            long flatA = 0, flatRes = 0;
            for (int d = 0; d < resArray.shape.length; d++) {
                flatA += coords[d] * (long) a.strides[d];
                flatRes += coords[d] * (long) resArray.strides[d];
            }
            int valA = a.data.getAtIndex(ValueLayout.JAVA_INT, flatA);
            resArray.data.setAtIndex(ValueLayout.JAVA_INT, flatRes, valA / b);
        }
        return resArray;
    }

    public static NDArray divDouble(NDArray a,double b,NDArray resArray){
        if(a.isContiguous() && resArray.isContiguous()) return divDoubleSIMD(a, b, resArray);
        else return divDoubleStrides(a, b, resArray);
    }

    private static NDArray divDoubleSIMD(NDArray a,double b,NDArray resArray){
        long i=0;
        long loopbound=a.size - (a.size % (DB_VL * 2));

        for(;i<loopbound;i+=DB_VL*2){
            var v1=DoubleVector.fromMemorySegment(SPECIESDB,a.data,i*DB_BYTES,ORDER);
            var v2=DoubleVector.fromMemorySegment(SPECIESDB,a.data,(i+DB_VL)*DB_BYTES,ORDER);
            var vRes1=v1.div(b);
            var vRes2=v2.div(b);
            vRes1.intoMemorySegment(resArray.data,i*DB_BYTES,ORDER);
            vRes2.intoMemorySegment(resArray.data,(i+DB_VL)*DB_BYTES,ORDER);
        }

        loopbound=SPECIESDB.loopBound(a.size);

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

    private static NDArray divDoubleStrides(NDArray a,double b,NDArray resArray){
        for(long i=0;i<resArray.size;i++){
            long tempIndex = i;
            long[] coords = new long[resArray.shape.length];
            for (int d = resArray.shape.length - 1; d >= 0; d--) {
                coords[d] = tempIndex % resArray.shape[d];
                tempIndex = tempIndex / resArray.shape[d];
            }
            long flatA = 0, flatRes = 0;
            for (int d = 0; d < resArray.shape.length; d++) {
                flatA += coords[d] * (long) a.strides[d];
                flatRes += coords[d] * (long) resArray.strides[d];
            }
            double valA = a.data.getAtIndex(ValueLayout.JAVA_DOUBLE, flatA);
            resArray.data.setAtIndex(ValueLayout.JAVA_DOUBLE, flatRes, valA / b);
        }
        return resArray;
    }
}
