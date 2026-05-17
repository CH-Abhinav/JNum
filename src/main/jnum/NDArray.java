package jnum;


import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Arrays;
import java.util.NoSuchElementException;
import java.util.concurrent.ThreadLocalRandom;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import jnum.jnumops.ArithmaticOps;
import jnum.jnumops.ReduceOps;
import jnum.jnumops.TrigOps;
import jnum.jnumops.ExpOps;
import jnum.jnumops.MatMulOps;

public class NDArray{
    public final MemorySegment data;
    public final int[] shape;
    public final int[] strides;
    public final long size;
    public final DType dtype;

    private NDArray(MemorySegment data,int[] shape,int[] strides,DType dType){
        this.data=data;
        this.shape=shape.clone();
        this.strides=strides.clone();
        long CalcSize=1;
        for (int dim : shape) CalcSize *= dim;
        this.size = CalcSize;
        this.dtype=dType;
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

    public static NDArray rand(int... shape){
        NDArray resArray=NDArray.zeros(shape);
        for(long i=0;i<resArray.size;i++){
            resArray.data.setAtIndex(ValueLayout.JAVA_FLOAT, i, ThreadLocalRandom.current().nextFloat());
        }
        return resArray;
    }

    public static NDArray rand(DType dType,int... shape){
        NDArray resArray=NDArray.zeros(dType, shape);
        switch(dType){
            case FLOAT->{for(long i=0;i<resArray.size;i++){
                resArray.data.setAtIndex(ValueLayout.JAVA_FLOAT, i, ThreadLocalRandom.current().nextFloat());
            }}
            case INTEGER->{for(long i=0;i<resArray.size;i++){
                resArray.data.setAtIndex(ValueLayout.JAVA_INT, i, ThreadLocalRandom.current().nextInt());
            }}
            case DOUBLE->{for(long i=0;i<resArray.size;i++){
                resArray.data.setAtIndex(ValueLayout.JAVA_DOUBLE, i, ThreadLocalRandom.current().nextDouble());
            }}
        }
        return resArray;
    }

    public static NDArray rand(Arena arena,DType dType,int... shape){
        NDArray resArray=NDArray.zeros(arena,dType, shape);
        switch(dType){
            case FLOAT->{for(long i=0;i<resArray.size;i++){
                resArray.data.setAtIndex(ValueLayout.JAVA_FLOAT, i, ThreadLocalRandom.current().nextFloat());
            }}
            case INTEGER->{for(long i=0;i<resArray.size;i++){
                resArray.data.setAtIndex(ValueLayout.JAVA_INT, i, ThreadLocalRandom.current().nextInt());
            }}
            case DOUBLE->{for(long i=0;i<resArray.size;i++){
                resArray.data.setAtIndex(ValueLayout.JAVA_DOUBLE, i, ThreadLocalRandom.current().nextDouble());
            }}
        }
        return resArray;
    }

    public static NDArray rand(int max,DType dType,int... shape){
        NDArray resArray=NDArray.zeros(dType, shape);
        switch(dType){
            case FLOAT->{for(long i=0;i<resArray.size;i++){
                resArray.data.setAtIndex(ValueLayout.JAVA_FLOAT, i, ThreadLocalRandom.current().nextFloat(max));
            }}
            case INTEGER->{for(long i=0;i<resArray.size;i++){
                resArray.data.setAtIndex(ValueLayout.JAVA_INT, i, ThreadLocalRandom.current().nextInt(max));
            }}
            case DOUBLE->{for(long i=0;i<resArray.size;i++){
                resArray.data.setAtIndex(ValueLayout.JAVA_DOUBLE, i, ThreadLocalRandom.current().nextDouble(max));
            }}
        }
        return resArray;
    }

    public static NDArray rand(float max,DType dType,int... shape){
        NDArray resArray=NDArray.zeros(dType, shape);
        switch(dType){
            case FLOAT->{for(long i=0;i<resArray.size;i++){
                resArray.data.setAtIndex(ValueLayout.JAVA_FLOAT, i, ThreadLocalRandom.current().nextFloat(max));
            }}
            case INTEGER->{for(long i=0;i<resArray.size;i++){
                throw new IllegalArgumentException();
            }}
            case DOUBLE->{for(long i=0;i<resArray.size;i++){
                resArray.data.setAtIndex(ValueLayout.JAVA_DOUBLE, i, ThreadLocalRandom.current().nextDouble(max));
            }}
        }
        return resArray;
    }

    public static NDArray rand(double max,DType dType,int... shape){
        NDArray resArray=NDArray.zeros(dType, shape);
        switch(dType){
            case FLOAT->{for(long i=0;i<resArray.size;i++){
                throw new IllegalArgumentException();
            }}
            case INTEGER->{for(long i=0;i<resArray.size;i++){
                throw new IllegalArgumentException();
            }}
            case DOUBLE->{for(long i=0;i<resArray.size;i++){
                resArray.data.setAtIndex(ValueLayout.JAVA_DOUBLE, i, ThreadLocalRandom.current().nextDouble(max));
            }}
        }
        return resArray;
    }

    public static NDArray rand(int min,int max,DType dType,int... shape){
        NDArray resArray=NDArray.zeros(dType, shape);
        switch(dType){
            case FLOAT->{for(long i=0;i<resArray.size;i++){
                resArray.data.setAtIndex(ValueLayout.JAVA_FLOAT, i, ThreadLocalRandom.current().nextFloat(min,max));
            }}
            case INTEGER->{for(long i=0;i<resArray.size;i++){
                resArray.data.setAtIndex(ValueLayout.JAVA_INT, i, ThreadLocalRandom.current().nextInt(min, max));
            }}
            case DOUBLE->{for(long i=0;i<resArray.size;i++){
                resArray.data.setAtIndex(ValueLayout.JAVA_DOUBLE, i, ThreadLocalRandom.current().nextDouble(min,max));
            }}
        }
        return resArray;
    }

    public static NDArray rand(float min,float max,DType dType,int... shape){
        NDArray resArray=NDArray.zeros(dType, shape);
        switch(dType){
            case FLOAT->{for(long i=0;i<resArray.size;i++){
                resArray.data.setAtIndex(ValueLayout.JAVA_FLOAT, i, ThreadLocalRandom.current().nextFloat(min,max));
            }}
            case INTEGER->throw new IllegalArgumentException();
            case DOUBLE->{for(long i=0;i<resArray.size;i++){
                resArray.data.setAtIndex(ValueLayout.JAVA_DOUBLE, i, ThreadLocalRandom.current().nextDouble(min,max));
            }}
        }
        return resArray;
    }

    public static NDArray rand(double min,double max,DType dType,int... shape){
        NDArray resArray=NDArray.zeros(dType, shape);
        switch(dType){
            case FLOAT->throw new IllegalArgumentException();
            case INTEGER->throw new IllegalArgumentException();
            case DOUBLE->{for(long i=0;i<resArray.size;i++){
                resArray.data.setAtIndex(ValueLayout.JAVA_DOUBLE, i, ThreadLocalRandom.current().nextDouble(min,max));
            }}
        }
        return resArray;
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

    public NDArray broadcastTo(int ... shape){
        if(Arrays.equals(this.shape,shape)) return this;
        int ndim=shape.length;
        if(ndim<this.ndim()){
            throw new IllegalArgumentException();
        }
        int[] newStrides=new int[ndim];
        int[] paddedShape=new int[ndim];
        int[] paddedStrides=new int[ndim];
        int offset=ndim=this.ndim();
        for(int i=0;i<ndim;i++){
            if(i<offset){
                paddedShape[i]=1;
                paddedStrides[i]=0;
            }
            else{
                paddedShape[i]=this.shape[i-offset];
                paddedStrides[i]=this.strides[i-offset];
            }
        }

        for(int i=0;i<ndim;i++){
            if(paddedShape[i]==shape[i]) newStrides[i]=paddedStrides[i];
            else if(paddedShape[i]==1) newStrides[i]=0;
            else throw new IllegalArgumentException();
        }

        return new NDArray(this.data, shape, newStrides, this.dtype);
    }

    public static int[] calculateBroadcastShape(int[] shapeA, int[] shapeB){
        int maxDims=Math.max(shapeA.length, shapeB.length);
        int[] result=new int[maxDims];
        for (int i = 1; i <= maxDims; i++) {
            int dimA = (shapeA.length - i >= 0) ? shapeA[shapeA.length - i] : 1;
            int dimB = (shapeB.length - i >= 0) ? shapeB[shapeB.length - i] : 1;
            
            if (dimA == dimB) {
                result[maxDims - i] = dimA;
            } else if (dimA == 1) {
                result[maxDims - i] = dimB;
            } else if (dimB == 1) {
                result[maxDims - i] = dimA;
            } else {
                throw new IllegalArgumentException("Shapes " + Arrays.toString(shapeA) + " and " + Arrays.toString(shapeB) + " are not broadcastable.");
            }
        }
        return result;
    }

    public static DType promoteTypes(DType a, DType b) {
        if (a == DType.DOUBLE || b == DType.DOUBLE) return DType.DOUBLE;
        if (a == DType.FLOAT || b == DType.FLOAT) return DType.FLOAT;
        return DType.INTEGER;
    }

    private static DType scalarType(int value) {
        return DType.INTEGER;
    }

    private static DType scalarType(float value) {
        return DType.FLOAT;
    }

    private static DType scalarType(double value) {
        return DType.DOUBLE;
    }

    private static NDArray prepareBroadcastOperand(NDArray array, int[] targetShape, DType targetType) {
        return array.broadcastTo(targetShape).cast(targetType);
    }

    private static NDArray validateResultArray(NDArray resArray, DType targetType, int[] targetShape) {
        if (resArray.dtype != targetType) {
            throw new IllegalArgumentException("Result dtype must be " + targetType + " but was " + resArray.dtype);
        }
        if (!Arrays.equals(resArray.shape, targetShape)) {
            throw new IllegalArgumentException("Result shape must be " + Arrays.toString(targetShape) + " but was " + Arrays.toString(resArray.shape));
        }
        return resArray;
    }

    private static void validateMatmulInputs(NDArray a, NDArray b) {
        if (a.ndim() != 2 || b.ndim() != 2) {
            throw new IllegalArgumentException();
        }
        if (a.shape[1] != b.shape[0]) {
            throw new IllegalArgumentException();
        }
    }

    private static int[] calculateReductionShape(int[] shape, int axis) {
        if (axis < 0 || axis >= shape.length) {
            throw new IllegalArgumentException("Axis " + axis + " is out of bounds for shape " + Arrays.toString(shape));
        }
        int[] reducedShape = new int[shape.length - 1];
        for (int i = 0, j = 0; i < shape.length; i++) {
            if (i != axis) {
                reducedShape[j++] = shape[i];
            }
        }
        return reducedShape;
    }

    public NDArray cast(DType target){
        if (this.dtype == target) {
            return this;
        }

        NDArray safeThis = this.isContiguous() ? this : this.contiguous();
        NDArray res = NDArray.zeros(target, safeThis.shape);
        for(long i=0;i <safeThis.size;i++){
            double val=switch (safeThis.dtype){
                case FLOAT -> safeThis.data.getAtIndex(ValueLayout.JAVA_FLOAT, i);
                case DOUBLE -> safeThis.data.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
                case INTEGER -> safeThis.data.getAtIndex(ValueLayout.JAVA_INT, i);
            };
            switch (target) {
                case FLOAT -> res.data.setAtIndex(ValueLayout.JAVA_FLOAT, i, (float) val);
                case DOUBLE -> res.data.setAtIndex(ValueLayout.JAVA_DOUBLE, i, val);
                case INTEGER -> res.data.setAtIndex(ValueLayout.JAVA_INT, i, (int) val);
            }
        }
        return res;
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
            var epsilon=1e-6f;
            for(long i=0;i<this.size;i++){
                float val = this.data.getAtIndex(ValueLayout.JAVA_FLOAT, i);
                if(Math.abs(c - val) < epsilon){
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
            var c= b;
            var epsilon=1e-12;
            for(long i=0;i<this.size;i++){
                var val=this.data.getAtIndex(ValueLayout.JAVA_DOUBLE,i);
                if(Math.abs(c - val) < epsilon){
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
            switch(this.dtype){
                case INTEGER->sb.append(data.getAtIndex(ValueLayout.JAVA_INT, i));
                case FLOAT->sb.append(data.getAtIndex(ValueLayout.JAVA_FLOAT, i));
                case DOUBLE->sb.append(data.getAtIndex(ValueLayout.JAVA_DOUBLE, i));
            }
            if (i < size - 1) sb.append(", ");
        }
        return sb.append("]").toString();
    }

    public double max() {
        return switch(this.dtype) {
            case FLOAT -> ReduceOps.maxFloat(this);
            case DOUBLE -> ReduceOps.maxDouble(this);
            case INTEGER -> ReduceOps.maxInt(this);
        };
    }

    public double min() {
        return switch(this.dtype) {
            case FLOAT -> ReduceOps.minFloat(this);
            case DOUBLE -> ReduceOps.minDouble(this);
            case INTEGER -> ReduceOps.minInt(this);
        };
    }

    public double sum() {
        return switch(this.dtype) {
            case FLOAT -> ReduceOps.sumFloat(this);
            case DOUBLE -> ReduceOps.sumDouble(this);
            case INTEGER -> ReduceOps.sumInt(this);
        };
    }

    public NDArray sum(int axis){
        int[] reducedShape = calculateReductionShape(this.shape, axis);
        NDArray resArray = NDArray.zeros(this.dtype, reducedShape);
        return switch(this.dtype) {
            case FLOAT -> ReduceOps.sumFloatAxis(this,axis,resArray);
            case DOUBLE -> ReduceOps.sumDoubleAxis(this,axis,resArray);
            case INTEGER -> ReduceOps.sumIntAxis(this,axis,resArray);
        };
    }

    public NDArray max(int axis) {
        int[] reducedShape = calculateReductionShape(this.shape, axis);
        NDArray resArray = NDArray.zeros(this.dtype, reducedShape);
        return switch(this.dtype) {
            case FLOAT -> ReduceOps.maxFloatAxis(this, axis, resArray);
            case DOUBLE -> ReduceOps.maxDoubleAxis(this, axis, resArray);
            case INTEGER -> ReduceOps.maxIntAxis(this, axis, resArray);
        };
    }

    public NDArray min(int axis) {
        int[] reducedShape = calculateReductionShape(this.shape, axis);
        NDArray resArray = NDArray.zeros(this.dtype, reducedShape);
        return switch(this.dtype) {
            case FLOAT -> ReduceOps.minFloatAxis(this, axis, resArray);
            case DOUBLE -> ReduceOps.minDoubleAxis(this, axis, resArray);
            case INTEGER -> ReduceOps.minIntAxis(this, axis, resArray);
        };
    }

    public double dot(NDArray b){
        if (this.ndim() != 1 || b.ndim() != 1) {
            throw new IllegalArgumentException("Dot product requires 1D vectors. Shapes: " + this.shapeString() + ", " + b.shapeString());
        }
        if (this.size != b.size) {
            throw new IllegalArgumentException("Vector sizes must match for dot product.");
        }
        return switch(this.dtype){
            case FLOAT->ReduceOps.dotFloat(this, b);
            case INTEGER->ReduceOps.dotInt(this, b);
            case DOUBLE->ReduceOps.dotDouble(this, b);
        };
    }

    public double avg() {
        return this.sum() / (double) this.size;
    }

    //ArithmaticOps.java 

    //addition operation

    public NDArray add(NDArray b){
        DType targetType = promoteTypes(this.dtype, b.dtype);
        int[] targetShape = calculateBroadcastShape(this.shape, b.shape);
        NDArray A = prepareBroadcastOperand(this, targetShape, targetType);
        NDArray B = prepareBroadcastOperand(b, targetShape, targetType);
        NDArray resArray = NDArray.zeros(targetType, targetShape);
        return switch(targetType) {
            case FLOAT -> ArithmaticOps.addFloat(A, B, resArray);
            case DOUBLE -> ArithmaticOps.addDouble(A, B, resArray);
            case INTEGER -> ArithmaticOps.addInt(A, B, resArray);
        };
    }
    
    public NDArray add(NDArray b,NDArray resArray){
        DType targetType = promoteTypes(this.dtype, b.dtype);
        int[] targetShape = calculateBroadcastShape(this.shape, b.shape);
        NDArray A = prepareBroadcastOperand(this, targetShape, targetType);
        NDArray B = prepareBroadcastOperand(b, targetShape, targetType);
        NDArray targetRes = validateResultArray(resArray, targetType, targetShape);
        return switch(targetType) {
            case FLOAT -> ArithmaticOps.addFloat(A, B, targetRes);
            case DOUBLE -> ArithmaticOps.addDouble(A, B, targetRes);
            case INTEGER -> ArithmaticOps.addInt(A, B, targetRes);
        };
    }

    public NDArray add(float b){
        DType targetType = promoteTypes(this.dtype, scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray resArray = NDArray.zeros(targetType, this.shape);
        return switch (targetType) {
            case FLOAT -> ArithmaticOps.addFloat(A, b, resArray);
            case DOUBLE -> ArithmaticOps.addDouble(A, b, resArray);
            case INTEGER -> ArithmaticOps.addInt(A, (int) b, resArray);
        };
    }

    public NDArray add(int b){
        DType targetType = promoteTypes(this.dtype, scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray resArray = NDArray.zeros(targetType, this.shape);
        return switch (targetType) {
            case FLOAT -> ArithmaticOps.addFloat(A, b, resArray);
            case DOUBLE -> ArithmaticOps.addDouble(A, b, resArray);
            case INTEGER -> ArithmaticOps.addInt(A, b, resArray);
        };
    }

    public NDArray add(double b){
        DType targetType = promoteTypes(this.dtype, scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray resArray = NDArray.zeros(targetType, this.shape);
        return switch (targetType) {
            case FLOAT -> ArithmaticOps.addFloat(A, (float) b, resArray);
            case DOUBLE -> ArithmaticOps.addDouble(A, b, resArray);
            case INTEGER -> ArithmaticOps.addInt(A, (int) b, resArray);
        };
    }

    public NDArray add(float b,NDArray resArray){
        DType targetType = promoteTypes(this.dtype, scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray targetRes = validateResultArray(resArray, targetType, this.shape);
        return switch (targetType) {
            case FLOAT -> ArithmaticOps.addFloat(A, b, targetRes);
            case DOUBLE -> ArithmaticOps.addDouble(A, b, targetRes);
            case INTEGER -> ArithmaticOps.addInt(A, (int) b, targetRes);
        };
    }

    public NDArray add(int b,NDArray resArray){
        DType targetType = promoteTypes(this.dtype, scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray targetRes = validateResultArray(resArray, targetType, this.shape);
        return switch (targetType) {
            case FLOAT -> ArithmaticOps.addFloat(A, b, targetRes);
            case DOUBLE -> ArithmaticOps.addDouble(A, b, targetRes);
            case INTEGER -> ArithmaticOps.addInt(A, b, targetRes);
        };
    }

    public NDArray add(double b,NDArray resArray){
        DType targetType = promoteTypes(this.dtype, scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray targetRes = validateResultArray(resArray, targetType, this.shape);
        return switch (targetType) {
            case FLOAT -> ArithmaticOps.addFloat(A, (float) b, targetRes);
            case DOUBLE -> ArithmaticOps.addDouble(A, b, targetRes);
            case INTEGER -> ArithmaticOps.addInt(A, (int) b, targetRes);
        };
    }

    //subtract operations

    public NDArray sub(NDArray b){
        DType targetType = promoteTypes(this.dtype, b.dtype);
        int[] targetShape = calculateBroadcastShape(this.shape, b.shape);
        NDArray A = prepareBroadcastOperand(this, targetShape, targetType);
        NDArray B = prepareBroadcastOperand(b, targetShape, targetType);
        NDArray resArray = NDArray.zeros(targetType, targetShape);
        return switch(targetType) {
            case FLOAT -> ArithmaticOps.subFloat(A, B, resArray);
            case DOUBLE -> ArithmaticOps.subDouble(A, B, resArray);
            case INTEGER -> ArithmaticOps.subInt(A, B, resArray);
        };
    }

    public NDArray sub(NDArray b,NDArray resArray){
        DType targetType = promoteTypes(this.dtype, b.dtype);
        int[] targetShape = calculateBroadcastShape(this.shape, b.shape);
        NDArray A = prepareBroadcastOperand(this, targetShape, targetType);
        NDArray B = prepareBroadcastOperand(b, targetShape, targetType);
        NDArray targetRes = validateResultArray(resArray, targetType, targetShape);
        return switch(targetType) {
            case FLOAT -> ArithmaticOps.subFloat(A, B, targetRes);
            case DOUBLE -> ArithmaticOps.subDouble(A, B, targetRes);
            case INTEGER -> ArithmaticOps.subInt(A, B, targetRes);
        };
    }

    public NDArray sub(float b){
        DType targetType = promoteTypes(this.dtype, scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray resArray = NDArray.zeros(targetType, this.shape);
        return switch (targetType) {
            case FLOAT -> ArithmaticOps.subFloat(A, b, resArray);
            case DOUBLE -> ArithmaticOps.subDouble(A, b, resArray);
            case INTEGER -> ArithmaticOps.subInt(A, (int) b, resArray);
        };
    }

    public NDArray sub(int b){
        DType targetType = promoteTypes(this.dtype, scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray resArray = NDArray.zeros(targetType, this.shape);
        return switch (targetType) {
            case FLOAT -> ArithmaticOps.subFloat(A, b, resArray);
            case DOUBLE -> ArithmaticOps.subDouble(A, b, resArray);
            case INTEGER -> ArithmaticOps.subInt(A, b, resArray);
        };
    }

    public NDArray sub(double b){
        DType targetType = promoteTypes(this.dtype, scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray resArray = NDArray.zeros(targetType, this.shape);
        return switch (targetType) {
            case FLOAT -> ArithmaticOps.subFloat(A, (float) b, resArray);
            case DOUBLE -> ArithmaticOps.subDouble(A, b, resArray);
            case INTEGER -> ArithmaticOps.subInt(A, (int) b, resArray);
        };
    }

    public NDArray sub(float b,NDArray resArray){
        DType targetType = promoteTypes(this.dtype, scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray targetRes = validateResultArray(resArray, targetType, this.shape);
        return switch (targetType) {
            case FLOAT -> ArithmaticOps.subFloat(A, b, targetRes);
            case DOUBLE -> ArithmaticOps.subDouble(A, b, targetRes);
            case INTEGER -> ArithmaticOps.subInt(A, (int) b, targetRes);
        };
    }

    public NDArray sub(int b,NDArray resArray){
        DType targetType = promoteTypes(this.dtype, scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray targetRes = validateResultArray(resArray, targetType, this.shape);
        return switch (targetType) {
            case FLOAT -> ArithmaticOps.subFloat(A, b, targetRes);
            case DOUBLE -> ArithmaticOps.subDouble(A, b, targetRes);
            case INTEGER -> ArithmaticOps.subInt(A, b, targetRes);
        };
    }

    public NDArray sub(double b,NDArray resArray){
        DType targetType = promoteTypes(this.dtype, scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray targetRes = validateResultArray(resArray, targetType, this.shape);
        return switch (targetType) {
            case FLOAT -> ArithmaticOps.subFloat(A, (float) b, targetRes);
            case DOUBLE -> ArithmaticOps.subDouble(A, b, targetRes);
            case INTEGER -> ArithmaticOps.subInt(A, (int) b, targetRes);
        };
    }

    //multiplication operations 

    public NDArray mul(NDArray b){
        DType targetType = promoteTypes(this.dtype, b.dtype);
        int[] targetShape = calculateBroadcastShape(this.shape, b.shape);
        NDArray A = prepareBroadcastOperand(this, targetShape, targetType);
        NDArray B = prepareBroadcastOperand(b, targetShape, targetType);
        NDArray resArray = NDArray.zeros(targetType, targetShape);
        return switch(targetType) {
            case FLOAT -> ArithmaticOps.mulFloat(A, B, resArray);
            case DOUBLE -> ArithmaticOps.mulDouble(A, B, resArray);
            case INTEGER -> ArithmaticOps.mulInt(A, B, resArray);
        };
    }

    public NDArray mul(NDArray b, NDArray resArray){
        DType targetType = promoteTypes(this.dtype, b.dtype);
        int[] targetShape = calculateBroadcastShape(this.shape, b.shape);
        NDArray A = prepareBroadcastOperand(this, targetShape, targetType);
        NDArray B = prepareBroadcastOperand(b, targetShape, targetType);
        NDArray targetRes = validateResultArray(resArray, targetType, targetShape);
        return switch(targetType) {
            case FLOAT -> ArithmaticOps.mulFloat(A, B, targetRes);
            case DOUBLE -> ArithmaticOps.mulDouble(A, B, targetRes);
            case INTEGER -> ArithmaticOps.mulInt(A, B, targetRes);
        };
    }

    public NDArray mul(float b){
        DType targetType = promoteTypes(this.dtype, scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray resArray = NDArray.zeros(targetType, this.shape);
        return switch (targetType) {
            case FLOAT -> ArithmaticOps.mulFloat(A, b, resArray);
            case DOUBLE -> ArithmaticOps.mulDouble(A, b, resArray);
            case INTEGER -> ArithmaticOps.mulInt(A, (int) b, resArray);
        };
    }

    public NDArray mul(int b){
        DType targetType = promoteTypes(this.dtype, scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray resArray = NDArray.zeros(targetType, this.shape);
        return switch (targetType) {
            case FLOAT -> ArithmaticOps.mulFloat(A, b, resArray);
            case DOUBLE -> ArithmaticOps.mulDouble(A, b, resArray);
            case INTEGER -> ArithmaticOps.mulInt(A, b, resArray);
        };
    }

    public NDArray mul(double b){
        DType targetType = promoteTypes(this.dtype, scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray resArray = NDArray.zeros(targetType, this.shape);
        return switch (targetType) {
            case FLOAT -> ArithmaticOps.mulFloat(A, (float) b, resArray);
            case DOUBLE -> ArithmaticOps.mulDouble(A, b, resArray);
            case INTEGER -> ArithmaticOps.mulInt(A, (int) b, resArray);
        };
    }

    public NDArray mul(float b,NDArray resArray){
        DType targetType = promoteTypes(this.dtype, scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray targetRes = validateResultArray(resArray, targetType, this.shape);
        return switch (targetType) {
            case FLOAT -> ArithmaticOps.mulFloat(A, b, targetRes);
            case DOUBLE -> ArithmaticOps.mulDouble(A, b, targetRes);
            case INTEGER -> ArithmaticOps.mulInt(A, (int) b, targetRes);
        };
    }

    public NDArray mul(int b,NDArray resArray){
        DType targetType = promoteTypes(this.dtype, scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray targetRes = validateResultArray(resArray, targetType, this.shape);
        return switch (targetType) {
            case FLOAT -> ArithmaticOps.mulFloat(A, b, targetRes);
            case DOUBLE -> ArithmaticOps.mulDouble(A, b, targetRes);
            case INTEGER -> ArithmaticOps.mulInt(A, b, targetRes);
        };
    }

    public NDArray mul(double b,NDArray resArray){
        DType targetType = promoteTypes(this.dtype, scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray targetRes = validateResultArray(resArray, targetType, this.shape);
        return switch (targetType) {
            case FLOAT -> ArithmaticOps.mulFloat(A, (float) b, targetRes);
            case DOUBLE -> ArithmaticOps.mulDouble(A, b, targetRes);
            case INTEGER -> ArithmaticOps.mulInt(A, (int) b, targetRes);
        };
    }

    //division operations

    public NDArray div(NDArray b){
        DType targetType = promoteTypes(this.dtype, b.dtype);
        int[] targetShape = calculateBroadcastShape(this.shape, b.shape);
        NDArray A = prepareBroadcastOperand(this, targetShape, targetType);
        NDArray B = prepareBroadcastOperand(b, targetShape, targetType);
        NDArray resArray = NDArray.zeros(targetType, targetShape);
        return switch(targetType) {
            case FLOAT -> ArithmaticOps.divFloat(A, B, resArray);
            case DOUBLE -> ArithmaticOps.divDouble(A, B, resArray);
            case INTEGER -> ArithmaticOps.divInt(A, B, resArray);
        };
    }

    public NDArray div(NDArray b, NDArray resArray){
        DType targetType = promoteTypes(this.dtype, b.dtype);
        int[] targetShape = calculateBroadcastShape(this.shape, b.shape);
        NDArray A = prepareBroadcastOperand(this, targetShape, targetType);
        NDArray B = prepareBroadcastOperand(b, targetShape, targetType);
        NDArray targetRes = validateResultArray(resArray, targetType, targetShape);
        return switch(targetType) {
            case FLOAT -> ArithmaticOps.divFloat(A, B, targetRes);
            case DOUBLE -> ArithmaticOps.divDouble(A, B, targetRes);
            case INTEGER -> ArithmaticOps.divInt(A, B, targetRes);
        };
    }

    public NDArray div(float b){
        DType targetType = promoteTypes(this.dtype, scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray resArray = NDArray.zeros(targetType, this.shape);
        return switch (targetType) {
            case FLOAT -> ArithmaticOps.divFloat(A, b, resArray);
            case DOUBLE -> ArithmaticOps.divDouble(A, b, resArray);
            case INTEGER -> ArithmaticOps.divInt(A, (int) b, resArray);
        };
    }

    public NDArray div(int b){
        DType targetType = promoteTypes(this.dtype, scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray resArray = NDArray.zeros(targetType, this.shape);
        return switch (targetType) {
            case FLOAT -> ArithmaticOps.divFloat(A, b, resArray);
            case DOUBLE -> ArithmaticOps.divDouble(A, b, resArray);
            case INTEGER -> ArithmaticOps.divInt(A, b, resArray);
        };
    }

    public NDArray div(double b){
        DType targetType = promoteTypes(this.dtype, scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray resArray = NDArray.zeros(targetType, this.shape);
        return switch (targetType) {
            case FLOAT -> ArithmaticOps.divFloat(A, (float) b, resArray);
            case DOUBLE -> ArithmaticOps.divDouble(A, b, resArray);
            case INTEGER -> ArithmaticOps.divInt(A, (int) b, resArray);
        };
    }

    public NDArray div(float b,NDArray resArray){
        DType targetType = promoteTypes(this.dtype, scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray targetRes = validateResultArray(resArray, targetType, this.shape);
        return switch (targetType) {
            case FLOAT -> ArithmaticOps.divFloat(A, b, targetRes);
            case DOUBLE -> ArithmaticOps.divDouble(A, b, targetRes);
            case INTEGER -> ArithmaticOps.divInt(A, (int) b, targetRes);
        };
    }

    public NDArray div(int b,NDArray resArray){
        DType targetType = promoteTypes(this.dtype, scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray targetRes = validateResultArray(resArray, targetType, this.shape);
        return switch (targetType) {
            case FLOAT -> ArithmaticOps.divFloat(A, b, targetRes);
            case DOUBLE -> ArithmaticOps.divDouble(A, b, targetRes);
            case INTEGER -> ArithmaticOps.divInt(A, b, targetRes);
        };
    }

    public NDArray div(double b,NDArray resArray){
        DType targetType = promoteTypes(this.dtype, scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray targetRes = validateResultArray(resArray, targetType, this.shape);
        return switch (targetType) {
            case FLOAT -> ArithmaticOps.divFloat(A, (float) b, targetRes);
            case DOUBLE -> ArithmaticOps.divDouble(A, b, targetRes);
            case INTEGER -> ArithmaticOps.divInt(A, (int) b, targetRes);
        };
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

    //ExpOps.java methods

    public NDArray sqrt(){
        NDArray safeThis = this.isContiguous() ? this : this.contiguous();
        return switch(this.dtype){
            case FLOAT-> ExpOps.sqrtFloat(safeThis, NDArray.zeros(DType.FLOAT,this.shape));
            case DOUBLE-> ExpOps.sqrtDouble(safeThis, NDArray.zeros(DType.DOUBLE,this.shape));
            case INTEGER -> ExpOps.sqrtInt(safeThis, NDArray.zeros(DType.FLOAT, this.shape));
        };
    }

    public NDArray abs(){
        NDArray safeThis = this.isContiguous() ? this : this.contiguous();
        return switch(this.dtype){
            case FLOAT-> ExpOps.absFloat(safeThis, NDArray.zeros(DType.FLOAT,this.shape));
            case DOUBLE-> ExpOps.absDouble(safeThis, NDArray.zeros(DType.DOUBLE,this.shape));
            case INTEGER -> ExpOps.absInt(safeThis, NDArray.zeros(DType.INTEGER, this.shape));
        };
    }

    public NDArray exp(){
        NDArray safeThis = this.isContiguous() ? this : this.contiguous();
        return switch(this.dtype){
            case FLOAT-> ExpOps.expFloat(safeThis, NDArray.zeros(DType.FLOAT,this.shape));
            case DOUBLE-> ExpOps.expDouble(safeThis, NDArray.zeros(DType.DOUBLE,this.shape));
            case INTEGER -> ExpOps.expInt(safeThis, NDArray.zeros(DType.FLOAT, this.shape));
        };
    }

    public NDArray log(){
        NDArray safeThis = this.isContiguous() ? this : this.contiguous();
        return switch(this.dtype){
            case FLOAT-> ExpOps.logFloat(safeThis, NDArray.zeros(DType.FLOAT,this.shape));
            case DOUBLE-> ExpOps.logDouble(safeThis, NDArray.zeros(DType.DOUBLE,this.shape));
            case INTEGER -> ExpOps.logInt(safeThis, NDArray.zeros(DType.FLOAT, this.shape));
        };
    }

    public NDArray log10(){
        NDArray safeThis = this.isContiguous() ? this : this.contiguous();
        return switch(this.dtype){
            case FLOAT-> ExpOps.log10Float(safeThis, NDArray.zeros(DType.FLOAT,this.shape));
            case DOUBLE-> ExpOps.log10Double(safeThis, NDArray.zeros(DType.DOUBLE,this.shape));
            case INTEGER -> ExpOps.log10Int(safeThis, NDArray.zeros(DType.FLOAT, this.shape));
        };
    }

    //TrigOps.java methods

    public NDArray sin(){
        NDArray safeThis = this.isContiguous() ? this : this.contiguous();
        
        return switch(this.dtype){
            case FLOAT-> TrigOps.sinFloat(safeThis, NDArray.zeros(DType.FLOAT,this.shape));
            case DOUBLE-> TrigOps.sinDouble(safeThis, NDArray.zeros(DType.DOUBLE,this.shape));
            case INTEGER -> TrigOps.sinInt(safeThis, NDArray.zeros(DType.FLOAT, this.shape));
        };
    }

    public NDArray cos(){
        NDArray safeThis = this.isContiguous() ? this : this.contiguous();
        return switch(this.dtype){
            case FLOAT-> TrigOps.cosFloat(safeThis, NDArray.zeros(DType.FLOAT,this.shape));
            case DOUBLE-> TrigOps.cosDouble(safeThis, NDArray.zeros(DType.DOUBLE,this.shape));
            case INTEGER -> TrigOps.cosInt(safeThis, NDArray.zeros(DType.FLOAT, this.shape));
        };
    }

    public NDArray tan(){
        NDArray safeThis = this.isContiguous() ? this : this.contiguous();
        
        return switch(this.dtype){
            case FLOAT-> TrigOps.tanFloat(safeThis, NDArray.zeros(DType.FLOAT,this.shape));
            case DOUBLE-> TrigOps.tanDouble(safeThis, NDArray.zeros(DType.DOUBLE,this.shape));
            case INTEGER -> TrigOps.tanInt(safeThis, NDArray.zeros(DType.FLOAT, this.shape));
        };
    }

    public NDArray cot(){
        NDArray safeThis = this.isContiguous() ? this : this.contiguous();
        
        return switch(this.dtype){
            case FLOAT-> TrigOps.cotFloat(safeThis, NDArray.zeros(DType.FLOAT,this.shape));
            case DOUBLE-> TrigOps.cotDouble(safeThis, NDArray.zeros(DType.DOUBLE,this.shape));
            case INTEGER -> TrigOps.cotInt(safeThis, NDArray.zeros(DType.FLOAT, this.shape));
        };
    }

    public NDArray sinh(){
        NDArray safeThis = this.isContiguous() ? this : this.contiguous();
        return switch(this.dtype){
            case FLOAT-> TrigOps.sinhFloat(safeThis, NDArray.zeros(DType.FLOAT,this.shape));
            case DOUBLE-> TrigOps.sinhDouble(safeThis, NDArray.zeros(DType.DOUBLE,this.shape));
            case INTEGER -> TrigOps.sinhInt(safeThis, NDArray.zeros(DType.FLOAT, this.shape));
        };
    }

    public NDArray cosh(){
        NDArray safeThis = this.isContiguous() ? this : this.contiguous();
        return switch(this.dtype){
            case FLOAT-> TrigOps.coshFloat(safeThis, NDArray.zeros(DType.FLOAT,this.shape));
            case DOUBLE-> TrigOps.coshDouble(safeThis, NDArray.zeros(DType.DOUBLE,this.shape));
            case INTEGER -> TrigOps.coshInt(safeThis, NDArray.zeros(DType.FLOAT, this.shape));
        };
    }

    public NDArray tanh(){
        NDArray safeThis = this.isContiguous() ? this : this.contiguous();
        
        return switch(this.dtype){
            case FLOAT-> TrigOps.tanhFloat(safeThis, NDArray.zeros(DType.FLOAT,this.shape));
            case DOUBLE-> TrigOps.tanhDouble(safeThis, NDArray.zeros(DType.DOUBLE,this.shape));
            case INTEGER -> TrigOps.tanhInt(safeThis, NDArray.zeros(DType.FLOAT, this.shape));
        };
    }

    public NDArray coth(){
        NDArray safeThis = this.isContiguous() ? this : this.contiguous();
        
        return switch(this.dtype){
            case FLOAT-> TrigOps.cothFloat(safeThis, NDArray.zeros(DType.FLOAT,this.shape));
            case DOUBLE-> TrigOps.cothDouble(safeThis, NDArray.zeros(DType.DOUBLE,this.shape));
            case INTEGER -> TrigOps.cothInt(safeThis, NDArray.zeros(DType.FLOAT, this.shape));
        };
    }

    //MatMulOps.java methods

    public NDArray matmul(NDArray b){
        validateMatmulInputs(this, b);
        DType targetType = promoteTypes(this.dtype, b.dtype);
        NDArray A = this.cast(targetType);
        NDArray B = b.cast(targetType);
        int[] targetShape = new int[]{this.shape[0], b.shape[1]};
        NDArray resArray = NDArray.zeros(targetType, targetShape);
        return switch(targetType){
            case FLOAT -> MatMulOps.matmulFloat(A, B, resArray);
            case DOUBLE -> MatMulOps.matmulDouble(A, B, resArray);
            case INTEGER -> MatMulOps.matmulInt(A, B, resArray);
        };
    }

    public NDArray matmul(NDArray b, NDArray resArray){
        validateMatmulInputs(this, b);
        DType targetType = promoteTypes(this.dtype, b.dtype);
        NDArray A = this.cast(targetType);
        NDArray B = b.cast(targetType);
        int[] targetShape = new int[]{this.shape[0], b.shape[1]};
        NDArray targetRes = validateResultArray(resArray, targetType, targetShape);
        return switch(targetType){
            case FLOAT -> MatMulOps.matmulFloat(A, B, targetRes);
            case DOUBLE -> MatMulOps.matmulDouble(A, B, targetRes);
            case INTEGER -> MatMulOps.matmulInt(A, B, targetRes);
        };
    }
}
