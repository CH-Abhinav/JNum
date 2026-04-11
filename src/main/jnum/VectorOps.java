package jnum;

import com.sun.jdi.FloatValue;
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

    //non instatitable utility class    
    private VectorOps(){
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
        for(int i=0;i<resArray.size;i++){
            int tempIndex = i;
            int[] coords = new int[resArray.shape.length];
            for (int d = resArray.shape.length - 1; d >= 0; d--) {
                coords[d] = tempIndex % resArray.shape[d];
                tempIndex = tempIndex / resArray.shape[d];
            }
            long flatA = 0, flatB = 0, flatRes = 0;
            for (int d = 0; d < resArray.shape.length; d++) {
                flatA += coords[d] * a.strides[d];
                flatB += coords[d] * b.strides[d];
                flatRes += coords[d] * resArray.strides[d];
            }
            float valA = a.data.getAtIndex(ValueLayout.JAVA_FLOAT, flatA);
            float valB = b.data.getAtIndex(ValueLayout.JAVA_FLOAT, flatB);
            resArray.data.setAtIndex(ValueLayout.JAVA_FLOAT, flatRes, valA + valB);
        }
        return resArray;
    }

    public static NDArray addInt(NDArray a,NDArray b,NDArray resArray){
        NDArray safeA = a.isContiguous() ? a : a.contiguous();
        NDArray safeB = b.isContiguous() ? b : b.contiguous();    
        long i=0;
        long loopbound=SPECIESINT.loopBound(safeA.size);

        for(;i<loopbound;i+=INT_VL){
            var v1=IntVector.fromMemorySegment(SPECIESINT,safeA.data,i*INT_BYTES,ORDER);
            var v2=IntVector.fromMemorySegment(SPECIESINT,safeB.data,i*INT_BYTES,ORDER);
            var vRes=v1.add(v2);
            vRes.intoMemorySegment(resArray.data,i*INT_BYTES,ORDER);
        }

        for (; i < safeA.size; i++) {
            var val1 = safeA.data.getAtIndex(ValueLayout.JAVA_INT, i);
            var val2 = safeB.data.getAtIndex(ValueLayout.JAVA_INT, i);
            resArray.data.setAtIndex(ValueLayout.JAVA_INT, i, val1 + val2);
        }
        return resArray;
    }

    public static NDArray addDouble(NDArray a,NDArray b,NDArray resArray){
        NDArray safeA = a.isContiguous() ? a : a.contiguous();
        NDArray safeB = b.isContiguous() ? b : b.contiguous();   
        long i=0;
        long loopbound=SPECIESDB.loopBound(safeA.size);

        for(;i<loopbound;i+=DB_VL){
            var v1=DoubleVector.fromMemorySegment(SPECIESDB,safeA.data,i*DB_BYTES,ORDER);
            var v2=DoubleVector.fromMemorySegment(SPECIESDB,safeB.data,i*DB_BYTES,ORDER);
            var vRes=v1.add(v2);
            vRes.intoMemorySegment(resArray.data,i*DB_BYTES,ORDER);
        }

        for (; i < safeA.size; i++) {
            var val1 = safeA.data.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
            var val2 = safeB.data.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
            resArray.data.setAtIndex(ValueLayout.JAVA_DOUBLE, i, val1 + val2);
        }
        return resArray;
    }

    public static NDArray subFloat(NDArray a,NDArray b,NDArray resArray){       
        if(a.isContiguous() && b.isContiguous() && resArray.isContiguous()) return subFloatSIMD(a, b, resArray);
        else return subFloatStrides(a, b, resArray);
    }

    private static NDArray subFloatSIMD(NDArray a,NDArray b,NDArray resArray){
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

    private static NDArray subFloatStrides(NDArray a,NDArray b,NDArray resArray){
        for(int i=0;i<resArray.size;i++){
            int tempIndex = i;
            int[] coords = new int[resArray.shape.length];
            for (int d = resArray.shape.length - 1; d >= 0; d--) {
                coords[d] = tempIndex % resArray.shape[d];
                tempIndex = tempIndex / resArray.shape[d];
            }
            long flatA = 0, flatB = 0, flatRes = 0;
            for (int d = 0; d < resArray.shape.length; d++) {
                flatA += coords[d] * a.strides[d];
                flatB += coords[d] * b.strides[d];
                flatRes += coords[d] * resArray.strides[d];
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

    private static NDArray subIntStrides(NDArray a,NDArray b,NDArray resArray){
        for(int i=0;i<resArray.size;i++){
            int tempIndex = i;
            int[] coords = new int[resArray.shape.length];
            for (int d = resArray.shape.length - 1; d >= 0; d--) {
                coords[d] = tempIndex % resArray.shape[d];
                tempIndex = tempIndex / resArray.shape[d];
            }
            long flatA = 0, flatB = 0, flatRes = 0;
            for (int d = 0; d < resArray.shape.length; d++) {
                flatA += coords[d] * a.strides[d];
                flatB += coords[d] * b.strides[d];
                flatRes += coords[d] * resArray.strides[d];
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

    private static NDArray subDoubleStrides(NDArray a,NDArray b,NDArray resArray){
        for(int i=0;i<resArray.size;i++){
            int tempIndex = i;
            int[] coords = new int[resArray.shape.length];
            for (int d = resArray.shape.length - 1; d >= 0; d--) {
                coords[d] = tempIndex % resArray.shape[d];
                tempIndex = tempIndex / resArray.shape[d];
            }
            long flatA = 0, flatB = 0, flatRes = 0;
            for (int d = 0; d < resArray.shape.length; d++) {
                flatA += coords[d] * a.strides[d];
                flatB += coords[d] * b.strides[d];
                flatRes += coords[d] * resArray.strides[d];
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

    private static NDArray mulFloatStrides(NDArray a,NDArray b,NDArray resArray){
        for(int i=0;i<resArray.size;i++){
            int tempIndex = i;
            int[] coords = new int[resArray.shape.length];
            for (int d = resArray.shape.length - 1; d >= 0; d--) {
                coords[d] = tempIndex % resArray.shape[d];
                tempIndex = tempIndex / resArray.shape[d];
            }
            long flatA = 0, flatB = 0, flatRes = 0;
            for (int d = 0; d < resArray.shape.length; d++) {
                flatA += coords[d] * a.strides[d];
                flatB += coords[d] * b.strides[d];
                flatRes += coords[d] * resArray.strides[d];
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

    private static NDArray mulIntStrides(NDArray a,NDArray b,NDArray resArray){
        for(int i=0;i<resArray.size;i++){
            int tempIndex = i;
            int[] coords = new int[resArray.shape.length];
            for (int d = resArray.shape.length - 1; d >= 0; d--) {
                coords[d] = tempIndex % resArray.shape[d];
                tempIndex = tempIndex / resArray.shape[d];
            }
            long flatA = 0, flatB = 0, flatRes = 0;
            for (int d = 0; d < resArray.shape.length; d++) {
                flatA += coords[d] * a.strides[d];
                flatB += coords[d] * b.strides[d];
                flatRes += coords[d] * resArray.strides[d];
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

    private static NDArray mulDoubleStrides(NDArray a,NDArray b,NDArray resArray){
        for(int i=0;i<resArray.size;i++){
            int tempIndex = i;
            int[] coords = new int[resArray.shape.length];
            for (int d = resArray.shape.length - 1; d >= 0; d--) {
                coords[d] = tempIndex % resArray.shape[d];
                tempIndex = tempIndex / resArray.shape[d];
            }
            long flatA = 0, flatB = 0, flatRes = 0;
            for (int d = 0; d < resArray.shape.length; d++) {
                flatA += coords[d] * a.strides[d];
                flatB += coords[d] * b.strides[d];
                flatRes += coords[d] * resArray.strides[d];
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

    private static NDArray divFloatStrides(NDArray a,NDArray b,NDArray resArray){
        for(int i=0;i<resArray.size;i++){
            int tempIndex = i;
            int[] coords = new int[resArray.shape.length];
            for (int d = resArray.shape.length - 1; d >= 0; d--) {
                coords[d] = tempIndex % resArray.shape[d];
                tempIndex = tempIndex / resArray.shape[d];
            }
            long flatA = 0, flatB = 0, flatRes = 0;
            for (int d = 0; d < resArray.shape.length; d++) {
                flatA += coords[d] * a.strides[d];
                flatB += coords[d] * b.strides[d];
                flatRes += coords[d] * resArray.strides[d];
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

    private static NDArray divIntStrides(NDArray a,NDArray b,NDArray resArray){
        for(int i=0;i<resArray.size;i++){
            int tempIndex = i;
            int[] coords = new int[resArray.shape.length];
            for (int d = resArray.shape.length - 1; d >= 0; d--) {
                coords[d] = tempIndex % resArray.shape[d];
                tempIndex = tempIndex / resArray.shape[d];
            }
            long flatA = 0, flatB = 0, flatRes = 0;
            for (int d = 0; d < resArray.shape.length; d++) {
                flatA += coords[d] * a.strides[d];
                flatB += coords[d] * b.strides[d];
                flatRes += coords[d] * resArray.strides[d];
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

    private static NDArray divDoubleStrides(NDArray a,NDArray b,NDArray resArray){
        for(int i=0;i<resArray.size;i++){
            int tempIndex = i;
            int[] coords = new int[resArray.shape.length];
            for (int d = resArray.shape.length - 1; d >= 0; d--) {
                coords[d] = tempIndex % resArray.shape[d];
                tempIndex = tempIndex / resArray.shape[d];
            }
            long flatA = 0, flatB = 0, flatRes = 0;
            for (int d = 0; d < resArray.shape.length; d++) {
                flatA += coords[d] * a.strides[d];
                flatB += coords[d] * b.strides[d];
                flatRes += coords[d] * resArray.strides[d];
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

    private static NDArray addFloatStrides(NDArray a,float b,NDArray resArray){
        for(int i=0;i<resArray.size;i++){
            int tempIndex = i;
            int[] coords = new int[resArray.shape.length];
            for (int d = resArray.shape.length - 1; d >= 0; d--) {
                coords[d] = tempIndex % resArray.shape[d];
                tempIndex = tempIndex / resArray.shape[d];
            }
            long flatA = 0, flatRes = 0;
            for (int d = 0; d < resArray.shape.length; d++) {
                flatA += coords[d] * a.strides[d];
                flatRes += coords[d] * resArray.strides[d];
            }
            float valA = a.data.getAtIndex(ValueLayout.JAVA_FLOAT, flatA);
            resArray.data.setAtIndex(ValueLayout.JAVA_FLOAT, flatRes, valA + b);
        }
        return resArray;
    }

    public static NDArray addInt(NDArray a,int b,NDArray resArray){
        NDArray safeA = a.isContiguous() ? a : a.contiguous();
        long i=0;
        long loopbound=SPECIESINT.loopBound(safeA.size);

        for(;i<loopbound;i+=INT_VL){
            var v1=IntVector.fromMemorySegment(SPECIESINT,safeA.data,i*INT_BYTES,ORDER);
            var vRes=v1.add(b);
            vRes.intoMemorySegment(resArray.data,i*INT_BYTES,ORDER);
        }

        for (; i < safeA.size; i++) {
            var val1 = safeA.data.getAtIndex(ValueLayout.JAVA_INT, i);
            resArray.data.setAtIndex(ValueLayout.JAVA_INT, i, val1 + b);
        }
        return resArray;
    }

    public static NDArray addDouble(NDArray a,double b,NDArray resArray){
        NDArray safeA = a.isContiguous() ? a : a.contiguous();
        long i=0;
        long loopbound=SPECIESDB.loopBound(safeA.size);

        for(;i<loopbound;i+=DB_VL){
            var v1=DoubleVector.fromMemorySegment(SPECIESDB,safeA.data,i*DB_BYTES,ORDER);
            var vRes=v1.add(b);
            vRes.intoMemorySegment(resArray.data,i*DB_BYTES,ORDER);
        }

        for (; i < safeA.size; i++) {
            var val1 = safeA.data.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
            resArray.data.setAtIndex(ValueLayout.JAVA_DOUBLE, i, val1 + b);
        }
        return resArray;
    }

    public static NDArray subFloat(NDArray a,float b,NDArray resArray){
        if(a.isContiguous() && resArray.isContiguous()) return subFloatSIMD(a, b, resArray);
        else return subFloatStrides(a, b, resArray);
    }

    private static NDArray subFloatSIMD(NDArray a,float b,NDArray resArray){
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

    private static NDArray subFloatStrides(NDArray a,float b,NDArray resArray){
        for(int i=0;i<resArray.size;i++){
            int tempIndex = i;
            int[] coords = new int[resArray.shape.length];
            for (int d = resArray.shape.length - 1; d >= 0; d--) {
                coords[d] = tempIndex % resArray.shape[d];
                tempIndex = tempIndex / resArray.shape[d];
            }
            long flatA = 0, flatRes = 0;
            for (int d = 0; d < resArray.shape.length; d++) {
                flatA += coords[d] * a.strides[d];
                flatRes += coords[d] * resArray.strides[d];
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

    private static NDArray subIntStrides(NDArray a,int b,NDArray resArray){
        for(int i=0;i<resArray.size;i++){
            int tempIndex = i;
            int[] coords = new int[resArray.shape.length];
            for (int d = resArray.shape.length - 1; d >= 0; d--) {
                coords[d] = tempIndex % resArray.shape[d];
                tempIndex = tempIndex / resArray.shape[d];
            }
            long flatA = 0, flatRes = 0;
            for (int d = 0; d < resArray.shape.length; d++) {
                flatA += coords[d] * a.strides[d];
                flatRes += coords[d] * resArray.strides[d];
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

    private static NDArray subDoubleStrides(NDArray a,double b,NDArray resArray){
        for(int i=0;i<resArray.size;i++){
            int tempIndex = i;
            int[] coords = new int[resArray.shape.length];
            for (int d = resArray.shape.length - 1; d >= 0; d--) {
                coords[d] = tempIndex % resArray.shape[d];
                tempIndex = tempIndex / resArray.shape[d];
            }
            long flatA = 0, flatRes = 0;
            for (int d = 0; d < resArray.shape.length; d++) {
                flatA += coords[d] * a.strides[d];
                flatRes += coords[d] * resArray.strides[d];
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

    private static NDArray mulFloatStrides(NDArray a,float b,NDArray resArray){
        for(int i=0;i<resArray.size;i++){
            int tempIndex = i;
            int[] coords = new int[resArray.shape.length];
            for (int d = resArray.shape.length - 1; d >= 0; d--) {
                coords[d] = tempIndex % resArray.shape[d];
                tempIndex = tempIndex / resArray.shape[d];
            }
            long flatA = 0, flatRes = 0;
            for (int d = 0; d < resArray.shape.length; d++) {
                flatA += coords[d] * a.strides[d];
                flatRes += coords[d] * resArray.strides[d];
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

    private static NDArray mulIntStrides(NDArray a,int b,NDArray resArray){
        for(int i=0;i<resArray.size;i++){
            int tempIndex = i;
            int[] coords = new int[resArray.shape.length];
            for (int d = resArray.shape.length - 1; d >= 0; d--) {
                coords[d] = tempIndex % resArray.shape[d];
                tempIndex = tempIndex / resArray.shape[d];
            }
            long flatA = 0, flatRes = 0;
            for (int d = 0; d < resArray.shape.length; d++) {
                flatA += coords[d] * a.strides[d];
                flatRes += coords[d] * resArray.strides[d];
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

    private static NDArray mulDoubleStrides(NDArray a,double b,NDArray resArray){
        for(int i=0;i<resArray.size;i++){
            int tempIndex = i;
            int[] coords = new int[resArray.shape.length];
            for (int d = resArray.shape.length - 1; d >= 0; d--) {
                coords[d] = tempIndex % resArray.shape[d];
                tempIndex = tempIndex / resArray.shape[d];
            }
            long flatA = 0, flatRes = 0;
            for (int d = 0; d < resArray.shape.length; d++) {
                flatA += coords[d] * a.strides[d];
                flatRes += coords[d] * resArray.strides[d];
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

    private static NDArray divFloatStrides(NDArray a,float b,NDArray resArray){
        for(int i=0;i<resArray.size;i++){
            int tempIndex = i;
            int[] coords = new int[resArray.shape.length];
            for (int d = resArray.shape.length - 1; d >= 0; d--) {
                coords[d] = tempIndex % resArray.shape[d];
                tempIndex = tempIndex / resArray.shape[d];
            }
            long flatA = 0, flatRes = 0;
            for (int d = 0; d < resArray.shape.length; d++) {
                flatA += coords[d] * a.strides[d];
                flatRes += coords[d] * resArray.strides[d];
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

    private static NDArray divIntStrides(NDArray a,int b,NDArray resArray){
        for(int i=0;i<resArray.size;i++){
            int tempIndex = i;
            int[] coords = new int[resArray.shape.length];
            for (int d = resArray.shape.length - 1; d >= 0; d--) {
                coords[d] = tempIndex % resArray.shape[d];
                tempIndex = tempIndex / resArray.shape[d];
            }
            long flatA = 0, flatRes = 0;
            for (int d = 0; d < resArray.shape.length; d++) {
                flatA += coords[d] * a.strides[d];
                flatRes += coords[d] * resArray.strides[d];
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

    private static NDArray divDoubleStrides(NDArray a,double b,NDArray resArray){
        for(int i=0;i<resArray.size;i++){
            int tempIndex = i;
            int[] coords = new int[resArray.shape.length];
            for (int d = resArray.shape.length - 1; d >= 0; d--) {
                coords[d] = tempIndex % resArray.shape[d];
                tempIndex = tempIndex / resArray.shape[d];
            }
            long flatA = 0, flatRes = 0;
            for (int d = 0; d < resArray.shape.length; d++) {
                flatA += coords[d] * a.strides[d];
                flatRes += coords[d] * resArray.strides[d];
            }
            double valA = a.data.getAtIndex(ValueLayout.JAVA_DOUBLE, flatA);
            resArray.data.setAtIndex(ValueLayout.JAVA_DOUBLE, flatRes, valA / b);
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
