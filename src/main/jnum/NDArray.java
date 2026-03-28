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
    public final DType dtype;
    //private static final VectorSpecies<Float> SPECIES= FloatVector.SPECIES_PREFERRED;
    //private static final long FLOAT_BYTES = ValueLayout.JAVA_FLOAT.byteSize();
    //private static final ByteOrder ORDER = ByteOrder.nativeOrder();
    //private static final int VL = SPECIES.length();

    private static void validArguments(NDArray a,NDArray b){
        if (!Arrays.equals(a.shape, b.shape)) {
            throw new IllegalArgumentException("Shape mismatch: " + a.shapeString() + " vs " + b.shapeString()+" cannot compute");
        }
        if(a.dtype!=b.dtype) throw new IllegalArgumentException("Types are not aliged: object type "+a.dtype+" is not same as operand type "+b.dtype);
    }

    private NDArray(MemorySegment data,int[] shape,int[] strides,DType dType){
        this.data=data;
        this.shape=shape.clone();
        this.strides=strides.clone();
        long CalcSize=1;
        for (int dim : shape) CalcSize *= dim;
        this.size = CalcSize;
        this.dtype=dType;
    }

    public static NDArray zeros(int... shape){
        return zeros(DType.FLOAT, shape);
    }

    public static NDArray zeros(DType dType,int... shape){
        long Size=1;
        for(int dim:shape) Size*=dim;
        Arena arena=Arena.ofAuto();
        MemorySegment segment=arena.allocate(dType.layout,Size);
        return new NDArray(segment,shape,calculateDefaultStrides(shape),dType);
    }

    public static NDArray ones(int...shape){
        return ones(DType.FLOAT, shape);
    }

    public static NDArray ones(DType dType,int... shape){
        long Size=1;
        for(int dim:shape) Size*=dim;
        Arena arena=Arena.ofAuto();
        MemorySegment segment=arena.allocate(dType.layout,Size);
        switch (dType) {
            case INTEGER -> {
            for(long i=0;i<Size;i++){
            segment.setAtIndex(ValueLayout.JAVA_INT, i, 1);
            }
            }
            case FLOAT ->{
                for(long i=0;i<Size;i++){
            segment.setAtIndex(ValueLayout.JAVA_FLOAT, i, 1.0f);
            }
            }
            case DOUBLE -> {
                for(long i=0;i<Size;i++){
            segment.setAtIndex(ValueLayout.JAVA_DOUBLE, i, 1.0);
            }
            }
            default -> throw new AssertionError();
        }
        return new NDArray(segment, shape, calculateDefaultStrides(shape),dType);
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
        DType dType=DType.FLOAT;
        Arena arena = Arena.ofAuto();
        MemorySegment segment = arena.allocate(dType.layout, CalcSize);
        MemorySegment.copy(data, 0,segment, dType.layout , 0, data.length);

        return new NDArray(segment,shape,calculateDefaultStrides(shape),dType);
    }

    public static NDArray from(int[] data,int... shape){
        long CalcSize=1;
        for(int dim:shape) CalcSize*=dim;
        if(CalcSize!=data.length){
            throw new IllegalArgumentException("Shape dimensions do not match data size.");
        }
        DType dType=DType.INTEGER;
        Arena arena = Arena.ofAuto();
        MemorySegment segment = arena.allocate(dType.layout, CalcSize);
        MemorySegment.copy(data, 0,segment, dType.layout , 0, data.length);

        return new NDArray(segment,shape,calculateDefaultStrides(shape),dType);
    }

    public static NDArray from(double [] data,int... shape){
        long CalcSize=1;
        for(int dim:shape) CalcSize*=dim;
        if(CalcSize!=data.length){
            throw new IllegalArgumentException("Shape dimensions do not match data size.");
        }
        DType dType=DType.DOUBLE;
        Arena arena = Arena.ofAuto();
        MemorySegment segment = arena.allocate(dType.layout, CalcSize);
        MemorySegment.copy(data, 0,segment, dType.layout , 0, data.length);

        return new NDArray(segment,shape,calculateDefaultStrides(shape),dType);
    }

    public NDArray reshape(int... newShape) {
        long newCalcSize = 1;
        for (int dim : newShape) newCalcSize *= dim;
        if (newCalcSize != this.size) {
            throw new IllegalArgumentException("Cannot reshape array of size " + this.size + " into shape " + Arrays.toString(newShape));
        }
        return new NDArray(this.data, newShape, calculateDefaultStrides(newShape),this.dtype);
    }

    public NDArray reshape(DType dType,int... newShape) {
        long newCalcSize = 1;
        for (int dim : newShape) newCalcSize *= dim;
        if (newCalcSize != this.size) {
            throw new IllegalArgumentException("Cannot reshape array of size " + this.size + " into shape " + Arrays.toString(newShape));
        }
        return new NDArray(this.data, newShape, calculateDefaultStrides(newShape),dType);
    }

    public String shapeString() {
        return Arrays.toString(shape).replace("[", "(").replace("]", ")");
    }

    public double getFlat(long index){
        return switch(this.dtype){
            case FLOAT ->data.getAtIndex(ValueLayout.JAVA_FLOAT, index);
            case INTEGER ->data.getAtIndex(ValueLayout.JAVA_INT, index);
            case DOUBLE ->data.getAtIndex(ValueLayout.JAVA_DOUBLE, index);
            default -> throw new AssertionError();
        };
    }

    public float getFlatFloat(long index) {
        return data.getAtIndex(ValueLayout.JAVA_FLOAT, index);
    }

    public int getFlatInt(long index){
        return data.getAtIndex(ValueLayout.JAVA_INT, index);
    }

    public double getFlatDouble(long index){
        return data.getAtIndex(ValueLayout.JAVA_DOUBLE, index);
    }

    public double get(int... indices){
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

    public int getInt(int... indices){
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
        return getFlatInt(flatIndex);
    }

    public float getFloat(int... indices){
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
        return getFlatFloat(flatIndex);
    }

    public int[] shape() {
        return shape.clone();
    }

    public int ndim() {
        return shape.length;
    }

    public long size() {
        return size;
    }

    public DType getDType(){
        return this.dtype;
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

    public double max() {
        return switch(this.dtype) {
            case FLOAT -> VectorOps.maxFloat(this);
            case DOUBLE -> VectorOps.maxDouble(this);
            case INTEGER -> VectorOps.maxInt(this);
        };
    }

    public double min() {
        return switch(this.dtype) {
            case FLOAT -> VectorOps.minFloat(this);
            case DOUBLE -> VectorOps.minDouble(this);
            case INTEGER -> VectorOps.minInt(this);
        };
    }

    public double sum() {
        return switch(this.dtype) {
            case FLOAT -> VectorOps.sumFloat(this);
            case DOUBLE -> VectorOps.sumDouble(this);
            case INTEGER -> VectorOps.sumInt(this);
        };
    }

    public double avg() {
        return this.sum() / (double) this.size;
    }

    public NDArray add(NDArray b){
        validArguments(this,b);
        NDArray resArray = NDArray.zeros(this.dtype,this.shape);
        return switch(this.dtype) {
        case FLOAT -> VectorOps.addFloat(this, b, resArray);
        case DOUBLE -> VectorOps.addDouble(this, b, resArray);
        case INTEGER -> VectorOps.addInt(this, b, resArray);
    };
    }

    public NDArray add(NDArray b,NDArray resArray){
        validArguments(this,b);
        return switch(this.dtype) {
        case FLOAT -> VectorOps.addFloat(this, b, resArray);
        case DOUBLE -> VectorOps.addDouble(this, b, resArray);
        case INTEGER -> VectorOps.addInt(this, b, resArray);
    };
    }

    public NDArray add(float b){
        if(this.dtype==DType.INTEGER) throw new IllegalArgumentException();
        NDArray resArray = NDArray.zeros(this.dtype,this.shape);
        return VectorOps.addFloat(this, b, resArray);
    }

    public NDArray add(int b){
        NDArray resArray = NDArray.zeros(this.dtype,this.shape);
        return VectorOps.addInt(this, b, resArray);
    }

    public NDArray add(double b){
        if(this.dtype==DType.INTEGER || this.dtype==DType.FLOAT) throw new IllegalArgumentException();
        NDArray resArray = NDArray.zeros(this.dtype,this.shape);
        return VectorOps.addDouble(this, b, resArray);
    }

    public NDArray add(float b,NDArray resArray){
        validArguments(this,resArray);
        if(this.dtype==DType.INTEGER) throw new IllegalArgumentException();
        if(resArray.dtype==DType.INTEGER) throw new IllegalArgumentException();
        return VectorOps.addFloat(this,b,resArray);
    }

    public NDArray add(int b,NDArray resArray){
        validArguments(this,resArray);
        return VectorOps.addInt(this,b,resArray);
    }

    public NDArray add(double b,NDArray resArray){
        validArguments(this,resArray);
        if(this.dtype==DType.INTEGER || this.dtype==DType.FLOAT) throw new IllegalArgumentException();
        if(resArray.dtype==DType.INTEGER || resArray.dtype==DType.FLOAT) throw new IllegalArgumentException();
        return VectorOps.addDouble(this,b,resArray);
    }

    public NDArray sub(NDArray b){
        validArguments(this,b);
        NDArray resArray=NDArray.zeros(this.shape);
        return switch(this.dtype) {
        case FLOAT -> VectorOps.subFloat(this, b, resArray);
        case DOUBLE -> VectorOps.subDouble(this, b, resArray);
        case INTEGER -> VectorOps.subInt(this, b, resArray);
    };
    }

    public NDArray sub(NDArray b,NDArray resArray){
        validArguments(this,b);
        return switch(this.dtype) {
        case FLOAT -> VectorOps.subFloat(this, b, resArray);
        case DOUBLE -> VectorOps.subDouble(this, b, resArray);
        case INTEGER -> VectorOps.subInt(this, b, resArray);
    };
    }

    public NDArray sub(float b){
        if(this.dtype==DType.INTEGER) throw new IllegalArgumentException();
        NDArray resArray = NDArray.zeros(this.dtype,this.shape);
        return VectorOps.subFloat(this, b, resArray);
    }

    public NDArray sub(int b){
        NDArray resArray = NDArray.zeros(this.dtype,this.shape);
        return VectorOps.subInt(this, b, resArray);
    }

    public NDArray sub(double b){
        if(this.dtype==DType.INTEGER || this.dtype==DType.FLOAT) throw new IllegalArgumentException();
        NDArray resArray = NDArray.zeros(this.dtype,this.shape);
        return VectorOps.subDouble(this, b, resArray);
    }

    public NDArray sub(float b,NDArray resArray){
        validArguments(this,resArray);
        if(this.dtype==DType.INTEGER) throw new IllegalArgumentException();
        if(resArray.dtype==DType.INTEGER) throw new IllegalArgumentException();
        return VectorOps.subFloat(this, b, resArray);
    }

    public NDArray sub(int b,NDArray resArray){
        validArguments(this,resArray);
        return VectorOps.subInt(this, b, resArray);
    }

    public NDArray sub(double b,NDArray resArray){
        validArguments(this,resArray);
        if(this.dtype==DType.INTEGER || this.dtype==DType.FLOAT) throw new IllegalArgumentException();
        if(resArray.dtype==DType.INTEGER || resArray.dtype==DType.FLOAT) throw new IllegalArgumentException();
        return VectorOps.subDouble(this, b, resArray);
    }

    public NDArray mul(NDArray b){
        validArguments(this, b);
        NDArray resArray = NDArray.zeros(this.dtype, this.shape);
        return switch(this.dtype) {
            case FLOAT -> VectorOps.mulFloat(this, b, resArray);
            case DOUBLE -> VectorOps.mulDouble(this, b, resArray);
            case INTEGER -> VectorOps.mulInt(this, b, resArray);
        };
    }

    public NDArray mul(NDArray b, NDArray resArray){
        validArguments(this, b);
        return switch(this.dtype) {
            case FLOAT -> VectorOps.mulFloat(this, b, resArray);
            case DOUBLE -> VectorOps.mulDouble(this, b, resArray);
            case INTEGER -> VectorOps.mulInt(this, b, resArray);
        };
    }

    public NDArray mul(float b){
        if(this.dtype==DType.INTEGER) throw new IllegalArgumentException();
        NDArray resArray = NDArray.zeros(this.shape);
        return VectorOps.mulFloat(this, b, resArray);
    }

    public NDArray mul(int b){
        NDArray resArray = NDArray.zeros(this.dtype, this.shape);
        return VectorOps.mulInt(this, b, resArray);
    }

    public NDArray mul(double b){
        if(this.dtype==DType.INTEGER || this.dtype==DType.FLOAT) throw new IllegalArgumentException();
        NDArray resArray = NDArray.zeros(this.dtype, this.shape);
        return VectorOps.mulDouble(this, b, resArray);
    }

    public NDArray mul(float b,NDArray resArray){
        validArguments(this,resArray);
        if(this.dtype==DType.INTEGER) throw new IllegalArgumentException();
        if(resArray.dtype==DType.INTEGER) throw new IllegalArgumentException();
        return VectorOps.mulFloat(this, b, resArray);
    }

    public NDArray mul(int b,NDArray resArray){
        validArguments(this,resArray);
        return VectorOps.mulInt(this, b, resArray);
    }

    public NDArray mul(double b,NDArray resArray){
        validArguments(this,resArray);
        if(this.dtype==DType.INTEGER || this.dtype==DType.FLOAT) throw new IllegalArgumentException();
        if(resArray.dtype==DType.INTEGER || resArray.dtype==DType.FLOAT) throw new IllegalArgumentException();
        return VectorOps.mulDouble(this, b, resArray);
    }

    public NDArray div(NDArray b){
        validArguments(this, b);
        NDArray resArray = NDArray.zeros(this.dtype, this.shape);
        return switch(this.dtype) {
            case FLOAT -> VectorOps.divFloat(this, b, resArray);
            case DOUBLE -> VectorOps.divDouble(this, b, resArray);
            case INTEGER -> VectorOps.divInt(this, b, resArray);
        };
    }

    public NDArray div(NDArray b, NDArray resArray){
        validArguments(this, b);
        return switch(this.dtype) {
            case FLOAT -> VectorOps.divFloat(this, b, resArray);
            case DOUBLE -> VectorOps.divDouble(this, b, resArray);
            case INTEGER -> VectorOps.divInt(this, b, resArray);
        };
    }

    public NDArray div(float b){
        if(this.dtype==DType.INTEGER) throw new IllegalArgumentException();
        NDArray resArray = NDArray.zeros(this.shape);
        return VectorOps.divFloat(this, b, resArray);
    }

    public NDArray div(int b){
        NDArray resArray = NDArray.zeros(this.dtype, this.shape);
        return VectorOps.divInt(this, b, resArray);
    }

    public NDArray div(double b){
        if(this.dtype==DType.INTEGER || this.dtype==DType.FLOAT) throw new IllegalArgumentException();
        NDArray resArray = NDArray.zeros(this.dtype, this.shape);
        return VectorOps.divDouble(this, b, resArray);
    }

    public NDArray div(float b,NDArray resArray){
        validArguments(this,resArray);
        if(this.dtype==DType.INTEGER) throw new IllegalArgumentException();
        if(resArray.dtype==DType.INTEGER) throw new IllegalArgumentException();
        return VectorOps.divFloat(this, b, resArray);
    }

    public NDArray div(int b,NDArray resArray){
        validArguments(this,resArray);
        return VectorOps.divInt(this, b, resArray);
    }

    public NDArray div(double b,NDArray resArray){
        validArguments(this,resArray);
        if(this.dtype==DType.INTEGER || this.dtype==DType.FLOAT) throw new IllegalArgumentException();
        if(resArray.dtype==DType.INTEGER || resArray.dtype==DType.FLOAT) throw new IllegalArgumentException();
        return VectorOps.divDouble(this, b, resArray);
    }

}