package jnum;


import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Arrays;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
//import java.nio.ByteOrder;


public class NDArray{
    final MemorySegment data;
    final int[] shape;
    final int[] strides;
    final long size;
    //private static final VectorSpecies<Float> SPECIES= FloatVector.SPECIES_PREFERRED;
    //private static final long FLOAT_BYTES = ValueLayout.JAVA_FLOAT.byteSize();
    //private static final ByteOrder ORDER = ByteOrder.nativeOrder();
    //private static final int VL = SPECIES.length();

    private NDArray(MemorySegment data,int[] shape,int[] strides){
        this.data=data;
        this.shape=shape.clone();
        this.strides=strides.clone();
        long CalcSize=1;
        for (int dim : shape) CalcSize *= dim;
        this.size = CalcSize;
    }

    public static NDArray zeros(int... shape){
        long Size=1;
        for(int dim:shape) Size*=dim;
        Arena arena=Arena.ofAuto();
        MemorySegment segment=arena.allocate(ValueLayout.JAVA_FLOAT,Size);
        return new NDArray(segment,shape,calculateDefaultStrides(shape));
    }

    public static NDArray ones(int... shape){
        long Size=1;
        for(int dim:shape) Size*=dim;
        Arena arena=Arena.ofAuto();
        MemorySegment segment=arena.allocate(ValueLayout.JAVA_FLOAT,Size);
        for(long i=0;i<Size;i++){
            segment.setAtIndex(ValueLayout.JAVA_FLOAT, i, 1.0f);
        }
        return new NDArray(segment, shape, calculateDefaultStrides(shape));
    }

    private static int[] calculateDefaultStrides(int[] shape){
        int[] strides=new int[shape.length];
        int currentstride=1;
        for(int i=shape.length-1;i>=0;i--){
            strides[i]=currentstride;
            currentstride*=shape[i];
        }
        return strides;
    }

    public static NDArray from(float[] data,int... shape){
        long CalcSize=1;
        for(int dim:shape) CalcSize*=dim;
        if(CalcSize!=data.length){
            throw new IllegalArgumentException("Shape dimensions do not match data size.");
        }
        Arena arena = Arena.ofAuto();
        MemorySegment segment = arena.allocate(ValueLayout.JAVA_FLOAT, CalcSize);
        MemorySegment.copy(data, 0,segment, ValueLayout.JAVA_FLOAT , 0, data.length);

        return new NDArray(segment,shape,calculateDefaultStrides(shape));
    }

    public NDArray reshape(int... newShape) {
        long newCalcSize = 1;
        for (int dim : newShape) newCalcSize *= dim;
        if (newCalcSize != this.size) {
            throw new IllegalArgumentException("Cannot reshape array of size " + this.size + " into shape " + Arrays.toString(newShape));
        }
        return new NDArray(this.data, newShape, calculateDefaultStrides(newShape));
    }

    public String shapeString() {
        return Arrays.toString(shape).replace("[", "(").replace("]", ")");
    }

    public float getFlat(long index) {
        return data.getAtIndex(ValueLayout.JAVA_FLOAT, index);
    }

    public float get(int... indices){
        if(indices.length!=shape.length){
            throw new IllegalArgumentException("illegal indices :"+indices.length+" does not match with shape "+shape.length);
        }
        long flatIndex=0;
        for(int i=0;i<indices.length;i++){
            if (indices[i] < 0 || indices[i] >= shape[i]) {
                throw new IndexOutOfBoundsException("Index " + indices[i] + " is out of bounds for dimension " + i + " with size " + shape[i]);
            }
            flatIndex+=(long)indices[i]*strides[i];
        }
        return getFlat(flatIndex);
    }

    @Override
    public String toString(){
        StringBuilder sb = new StringBuilder("NDArray" + shapeString() + " [");
        int maxPrint = 6;
        for (long i = 0; i < size; i++) {
            if (i == maxPrint / 2 && size > maxPrint) {
                sb.append("..., ");
                i = size - (maxPrint / 2) - 1;
                continue;
            }
            sb.append(data.getAtIndex(ValueLayout.JAVA_FLOAT, i));
            if (i < size - 1) sb.append(", ");
        }
        return sb.append("]").toString();
    }

    public float max(){
        return VectorOps.max(this);
    }

    public float min(){
        return VectorOps.min(this);
    }

    public float sum(){
        return VectorOps.sum(this);
    }

    public float avg(){
        return VectorOps.sum(this)/(float)size;
    }

    public NDArray add(NDArray b){
        NDArray resArray = NDArray.zeros(this.shape);
        return VectorOps.add(this, b, resArray);
    }

    public NDArray add(NDArray b,NDArray resArray){
        return VectorOps.add(this,b,resArray);
    }

    public NDArray add(float b){
        NDArray resArray = NDArray.zeros(this.shape);
        return VectorOps.add(this, b, resArray);
    }

    public NDArray add(float b,NDArray resArray){
        return VectorOps.add(this,b,resArray);
    }

    public NDArray sub(NDArray b){
        NDArray resArray = NDArray.zeros(this.shape);
        return VectorOps.sub(this, b, resArray);
    }

    public NDArray sub(NDArray b,NDArray resArray){
        return VectorOps.sub(this,b,resArray);
    }

    public NDArray sub(float b){
        NDArray resArray = NDArray.zeros(this.shape);
        return VectorOps.sub(this, b, resArray);
    }

    public NDArray sub(float b,NDArray resArray){
        return VectorOps.sub(this,b,resArray);
    }

    public NDArray mul(NDArray b){
        NDArray resArray = NDArray.zeros(this.shape);
        return VectorOps.mul(this, b, resArray);
    }

    public NDArray mul(NDArray b,NDArray resArray){
        return VectorOps.mul(this,b,resArray);
    }

    public NDArray mul(float b){
        NDArray resArray = NDArray.zeros(this.shape);
        return VectorOps.mul(this, b, resArray);
    }

    public NDArray mul(float b,NDArray resArray){
        return VectorOps.mul(this,b,resArray);
    }

    public NDArray div(NDArray b){
        NDArray resArray = NDArray.zeros(this.shape);
        return VectorOps.div(this, b, resArray);
    }

    public NDArray div(NDArray b,NDArray resArray){
        return VectorOps.div(this,b,resArray);
    }

    public NDArray div(float b){
        NDArray resArray = NDArray.zeros(this.shape);
        return VectorOps.div(this, b, resArray);
    }

    public NDArray div(float b,NDArray resArray){
        return VectorOps.div(this,b,resArray);
    }


}