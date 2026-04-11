package jnum;


import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Arrays;
import java.util.NoSuchElementException;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;


public class NDArray{
    final MemorySegment data;
    final int[] shape;
    final int[] strides;
    final long size;
    public final DType dtype;

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
        return zeros(Arena.ofAuto(),DType.FLOAT,shape);
    }

    public static NDArray zeros(Arena arena,int... shape){
        return zeros(arena,DType.FLOAT, shape);
    }

    public static NDArray zeros(DType dType,int... shape){
        return zeros(Arena.ofAuto(),dType,shape);
    }

    public static NDArray zeros(Arena arena,DType dType,int... shape){
        long Size=1;
        for(int dim:shape) Size*=dim;
        MemorySegment segment=arena.allocate(dType.layout,Size);
        return new NDArray(segment,shape,calculateDefaultStrides(shape),dType);
    }

    public static NDArray ones(int... shape){
        return ones(Arena.ofAuto(),DType.FLOAT,shape);
    }

    public static NDArray ones(DType dType,int... shape){
        return ones(Arena.ofAuto(),dType,shape);
    }

    public static NDArray ones(Arena arena,int...shape){
        return ones(arena,DType.FLOAT, shape);
    }

    public static NDArray ones(Arena arena,DType dType,int... shape){
        long Size=1;
        for(int dim:shape) Size*=dim;
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

    //float array from methods
    public static NDArray from(float[] data, int... shape) {
        return from(Arena.ofAuto(), data, shape);
    }

    public static NDArray from(Arena arena, float[] data, int... shape) {
        long CalcSize = 1;
        for (int dim : shape) CalcSize *= dim;
        if (CalcSize != data.length) {
            throw new IllegalArgumentException("Shape dimensions do not match data size.");
        }
        DType dType = DType.FLOAT;
        MemorySegment segment = arena.allocate(dType.layout, CalcSize);
        MemorySegment.copy(data, 0, segment, dType.layout, 0, data.length);
        return new NDArray(segment, shape, calculateDefaultStrides(shape), dType);
    }

    //int array from methods
    public static NDArray from(int[] data, int... shape) {
        return from(Arena.ofAuto(), data, shape);
    }

    public static NDArray from(Arena arena, int[] data, int... shape) {
        long CalcSize = 1;
        for (int dim : shape) CalcSize *= dim;
        if (CalcSize != data.length) {
            throw new IllegalArgumentException("Shape dimensions do not match data size.");
        }
        DType dType = DType.INTEGER;
        MemorySegment segment = arena.allocate(dType.layout, CalcSize);
        MemorySegment.copy(data, 0, segment, dType.layout, 0, data.length);
        return new NDArray(segment, shape, calculateDefaultStrides(shape), dType);
    }

    // DOUBLE array from method
    public static NDArray from(double[] data, int... shape) {
        return from(Arena.ofAuto(), data, shape);
    }

    public static NDArray from(Arena arena, double[] data, int... shape) {
        long CalcSize = 1;
        for (int dim : shape) CalcSize *= dim;
        if (CalcSize != data.length) {
            throw new IllegalArgumentException("Shape dimensions do not match data size.");
        }
        DType dType = DType.DOUBLE;
        MemorySegment segment = arena.allocate(dType.layout, CalcSize);
        MemorySegment.copy(data, 0, segment, dType.layout, 0, data.length);
        return new NDArray(segment, shape, calculateDefaultStrides(shape), dType);
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

    public NDArray transpose(){
        if(ndim()<2) return this;
        var newShape=new int[this.shape.length];
        var newStrides=new int[this.strides.length];
        for(int i=0;i<this.shape.length;i++){
            newShape[i]=this.shape[(this.shape.length-1-i)];
            newStrides[i]=this.strides[(this.strides.length-1-i)];
        }
        return new NDArray(this.data,newShape,newStrides,this.dtype);
    }

    public boolean isContiguous(){
        var expStride=1;
        for(int i=this.shape.length-1;i>=0;i--){
            if(this.strides[i]!=expStride) return false;
            expStride*=shape[i];
        }
        return true;
    }

    public NDArray contiguous(){
        return contiguous(Arena.ofAuto());
    }

    public double dot(NDArray b){
        if (this.ndim() != 1 || b.ndim() != 1) {
            throw new IllegalArgumentException("Dot product requires 1D vectors. Shapes: " + this.shapeString() + ", " + b.shapeString());
        }
        if (this.size != b.size) {
            throw new IllegalArgumentException("Vector sizes must match for dot product.");
        }
        return this.mul(b).sum();
    }

    public NDArray contiguous(Arena arena){
        if(this.isContiguous()) return this;
        var segment=arena.allocate(this.dtype.layout,this.size);
        var newStrides=calculateDefaultStrides(this.shape);
        for(int i=0;i<this.size;i++){
            var tempindex=i;
            var coord=new int[this.shape.length];
            for(int j=this.shape.length-1;j>=0;j--){
                coord[j]=tempindex%this.shape[j];
                tempindex=tempindex/this.shape[j];
            }
            long oldFlatIndex = 0;
            for(int d = 0; d < this.shape.length; d++){
                oldFlatIndex += coord[d] * this.strides[d];
            }
            switch(dtype){
                case FLOAT->{
                    var val=this.data.getAtIndex(ValueLayout.JAVA_FLOAT, oldFlatIndex);
                    segment.setAtIndex(ValueLayout.JAVA_FLOAT, i, val);
                }
                case INTEGER->{
                    var val=this.data.getAtIndex(ValueLayout.JAVA_INT, oldFlatIndex);
                    segment.setAtIndex(ValueLayout.JAVA_INT, i, val);
                }
                case DOUBLE->{
                    var val=this.data.getAtIndex(ValueLayout.JAVA_DOUBLE, oldFlatIndex);
                    segment.setAtIndex(ValueLayout.JAVA_DOUBLE, i, val);
                }
            }
        }
        return new NDArray(segment, this.shape, newStrides, dtype);
    }


    public String shapeString() {
        return Arrays.toString(shape).replace("[", "(").replace("]", ")");
    }

    public NDArray copy(){
        NDArray dups=NDArray.zeros(this.dtype, this.shape);
        MemorySegment.copy(this.data, 0, dups.data, 0, this.data.byteSize());
        return dups;
    }

    public boolean equals(NDArray a){
        if (!Arrays.equals(this.shape, a.shape)) return false;
        if(this.dtype!=a.dtype) return false;
        long mismatch=this.data.mismatch(a.data);
        return mismatch==-1;
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

    public int[] indexOf(double b){
        int[] indices=new int[this.shape.length];
        switch(dtype){
            case FLOAT->{
                var c= (float)b;
            for(long i=0;i<this.size;i++){
                if(c==this.data.getAtIndex(ValueLayout.JAVA_FLOAT,i)){
                    for (int d = 0; d < this.shape.length; d++) {
                        indices[d] = (int) ((i / this.strides[d]) % this.shape[d]);
                    }
                    return indices;
                }
            }
        }
        case INTEGER->{
                var c= (int)b;
            for(long i=0;i<this.size;i++){
                if(c==this.data.getAtIndex(ValueLayout.JAVA_INT,i)){
                    for (int d = 0; d < this.shape.length; d++) {
                        indices[d] = (int) ((i / this.strides[d]) % this.shape[d]);
                    }
                    return indices;
                }
            }
        }
        case DOUBLE->{
            
                var c= (double)b;
            for(long i=0;i<this.size;i++){
                if(c==this.data.getAtIndex(ValueLayout.JAVA_DOUBLE,i)){
                    for (int d = 0; d < this.shape.length; d++) {
                        indices[d] = (int) ((i / this.strides[d]) % this.shape[d]);
                    }
                    return indices;
                    }
                }
            }
        }
    throw new NoSuchElementException();    
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

    public MemorySegment data() {
        return this.data;
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

    //VectorOps 

    //addition operation

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

    //subtract operations

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

    //multiplication operations 

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

    //division operations

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

    //IN PLACE operations of VectorOps

    // addinplace() methods

    public NDArray addi(NDArray b){
        return this.add(b,this);
    }

    public NDArray addi(float b){
        return this.add(b,this);
    }

    public NDArray addi(int b){
        return this.add(b,this);
    }

    public NDArray addi(double b){
        return this.add(b,this);
    }

    //subinplace() methods

    public NDArray subi(NDArray b){
        return this.sub(b,this);
    }

    public NDArray subi(float b){
        return this.sub(b,this);
    }

    public NDArray subi(int b){
        return this.sub(b,this);
    }

    public NDArray subi(double b){
        return this.sub(b,this);
    }

    //mulinplace() methods

    public NDArray muli(NDArray b){
        return this.mul(b,this);
    }

    public NDArray muli(float b){
        return this.mul(b,this);
    }

    public NDArray muli(int b){
        return this.mul(b,this);
    }

    public NDArray muli(double b){
        return this.mul(b,this);
    }

    //divinplace() methods

    public NDArray divi(NDArray b){
        return this.div(b,this);
    }

    public NDArray divi(float b){
        return this.div(b,this);
    }

    public NDArray divi(int b){
        return this.div(b,this);
    }

    public NDArray divi(double b){
        return this.div(b,this);
    }

    //MATH OPERATIONS FROM MathOps.java 

    public NDArray sqrt(){
        return switch(this.dtype){
            case FLOAT-> MathOps.sqrtFloat(this, NDArray.zeros(DType.FLOAT,this.shape));
            case DOUBLE-> MathOps.sqrtDouble(this, NDArray.zeros(DType.DOUBLE,this.shape));
            case INTEGER -> MathOps.sqrtInt(this, NDArray.zeros(DType.FLOAT, this.shape));
        };
    }

    public NDArray abs(){
        return switch(this.dtype){
            case FLOAT-> MathOps.absFloat(this, NDArray.zeros(DType.FLOAT,this.shape));
            case DOUBLE-> MathOps.absDouble(this, NDArray.zeros(DType.DOUBLE,this.shape));
            case INTEGER -> MathOps.absInt(this, NDArray.zeros(DType.INTEGER, this.shape));
        };
    }

    public NDArray exp(){
        return switch(this.dtype){
            case FLOAT-> MathOps.expFloat(this, NDArray.zeros(DType.FLOAT,this.shape));
            case DOUBLE-> MathOps.expDouble(this, NDArray.zeros(DType.DOUBLE,this.shape));
            case INTEGER -> MathOps.expInt(this, NDArray.zeros(DType.FLOAT, this.shape));
        };
    }

    public NDArray log(){
        return switch(this.dtype){
            case FLOAT-> MathOps.logFloat(this, NDArray.zeros(DType.FLOAT,this.shape));
            case DOUBLE-> MathOps.logDouble(this, NDArray.zeros(DType.DOUBLE,this.shape));
            case INTEGER -> MathOps.logInt(this, NDArray.zeros(DType.FLOAT, this.shape));
        };
    }

    public NDArray log10(){
        return switch(this.dtype){
            case FLOAT-> MathOps.log10Float(this, NDArray.zeros(DType.FLOAT,this.shape));
            case DOUBLE-> MathOps.log10Double(this, NDArray.zeros(DType.DOUBLE,this.shape));
            case INTEGER -> MathOps.log10Int(this, NDArray.zeros(DType.FLOAT, this.shape));
        };
    }

    public NDArray sin(){
        NDArray safeThis = this.isContiguous() ? this : this.contiguous();
        
        return switch(this.dtype){
            case FLOAT-> MathOps.sinFloat(safeThis, NDArray.zeros(DType.FLOAT,this.shape));
            case DOUBLE-> MathOps.sinDouble(safeThis, NDArray.zeros(DType.DOUBLE,this.shape));
            case INTEGER -> MathOps.sinInt(safeThis, NDArray.zeros(DType.FLOAT, this.shape));
        };
    }

    public NDArray cos(){
        NDArray safeThis = this.isContiguous() ? this : this.contiguous();
        return switch(this.dtype){
            case FLOAT-> MathOps.cosFloat(safeThis, NDArray.zeros(DType.FLOAT,this.shape));
            case DOUBLE-> MathOps.cosDouble(safeThis, NDArray.zeros(DType.DOUBLE,this.shape));
            case INTEGER -> MathOps.cosInt(safeThis, NDArray.zeros(DType.FLOAT, this.shape));
        };
    }

    public NDArray tan(){
        NDArray safeThis = this.isContiguous() ? this : this.contiguous();
        
        return switch(this.dtype){
            case FLOAT-> MathOps.tanFloat(safeThis, NDArray.zeros(DType.FLOAT,this.shape));
            case DOUBLE-> MathOps.tanDouble(safeThis, NDArray.zeros(DType.DOUBLE,this.shape));
            case INTEGER -> MathOps.tanInt(safeThis, NDArray.zeros(DType.FLOAT, this.shape));
        };
    }

    public NDArray cot(){
        NDArray safeThis = this.isContiguous() ? this : this.contiguous();
        
        return switch(this.dtype){
            case FLOAT-> MathOps.cotFloat(safeThis, NDArray.zeros(DType.FLOAT,this.shape));
            case DOUBLE-> MathOps.cotDouble(safeThis, NDArray.zeros(DType.DOUBLE,this.shape));
            case INTEGER -> MathOps.cotInt(safeThis, NDArray.zeros(DType.FLOAT, this.shape));
        };
    }

    public NDArray sinh(){
        NDArray safeThis = this.isContiguous() ? this : this.contiguous();
        return switch(this.dtype){
            case FLOAT-> MathOps.sinhFloat(safeThis, NDArray.zeros(DType.FLOAT,this.shape));
            case DOUBLE-> MathOps.sinhDouble(safeThis, NDArray.zeros(DType.DOUBLE,this.shape));
            case INTEGER -> MathOps.sinhInt(safeThis, NDArray.zeros(DType.FLOAT, this.shape));
        };
    }

    public NDArray cosh(){
        NDArray safeThis = this.isContiguous() ? this : this.contiguous();
        return switch(this.dtype){
            case FLOAT-> MathOps.coshFloat(safeThis, NDArray.zeros(DType.FLOAT,this.shape));
            case DOUBLE-> MathOps.coshDouble(safeThis, NDArray.zeros(DType.DOUBLE,this.shape));
            case INTEGER -> MathOps.coshInt(safeThis, NDArray.zeros(DType.FLOAT, this.shape));
        };
    }

    public NDArray tanh(){
        NDArray safeThis = this.isContiguous() ? this : this.contiguous();
        
        return switch(this.dtype){
            case FLOAT-> MathOps.tanhFloat(safeThis, NDArray.zeros(DType.FLOAT,this.shape));
            case DOUBLE-> MathOps.tanhDouble(safeThis, NDArray.zeros(DType.DOUBLE,this.shape));
            case INTEGER -> MathOps.tanhInt(safeThis, NDArray.zeros(DType.FLOAT, this.shape));
        };
    }




}