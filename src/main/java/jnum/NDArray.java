package jnum;


import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Arrays;
import java.util.NoSuchElementException;
import java.util.concurrent.ThreadLocalRandom;

import jnum.jnumops.ArithmeticOps;
import jnum.jnumops.CompareOps;
import jnum.jnumops.ExpOps;
import jnum.jnumops.MatMulOps;
import jnum.jnumops.NDIter;
import jnum.jnumops.ReduceOps;
import jnum.jnumops.TrigOps;
import jnum.jnumutils.ShapeUtil;
import jnum.jnumutils.TypeUtil;
import jnum.jnumutils.ValidUtil;

public class NDArray{
    private final MemorySegment data;
    private final int[] shape;
    private final int[] strides;
    private final long size;
    private final DType dtype;

    private NDArray(MemorySegment data,int[] shape,int[] strides,DType dType){
        this.data=data;
        this.shape=shape;
        this.strides=strides;
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
        return new NDArray(segment,shape,ShapeUtil.calculateDefaultStrides(shape),dType);
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
        return new NDArray(segment, shape, ShapeUtil.calculateDefaultStrides(shape),dType);
    }

    public static NDArray rand(int... shape){
        NDArray resArray=NDArray.zeros(shape);
        for(long i = 0; i< resArray.getSize(); i++){
            resArray.getData().setAtIndex(ValueLayout.JAVA_FLOAT, i, ThreadLocalRandom.current().nextFloat());
        }
        return resArray;
    }

    public static NDArray rand(DType dType,int... shape){
        NDArray resArray=NDArray.zeros(dType, shape);
        switch(dType){
            case FLOAT->{for(long i = 0; i< resArray.getSize(); i++){
                resArray.getData().setAtIndex(ValueLayout.JAVA_FLOAT, i, ThreadLocalRandom.current().nextFloat());
            }}
            case INTEGER->{for(long i = 0; i< resArray.getSize(); i++){
                resArray.getData().setAtIndex(ValueLayout.JAVA_INT, i, ThreadLocalRandom.current().nextInt());
            }}
            case DOUBLE->{for(long i = 0; i< resArray.getSize(); i++){
                resArray.getData().setAtIndex(ValueLayout.JAVA_DOUBLE, i, ThreadLocalRandom.current().nextDouble());
            }}
        }
        return resArray;
    }

    public static NDArray rand(Arena arena,DType dType,int... shape){
        NDArray resArray=NDArray.zeros(arena,dType, shape);
        switch(dType){
            case FLOAT->{for(long i = 0; i< resArray.getSize(); i++){
                resArray.getData().setAtIndex(ValueLayout.JAVA_FLOAT, i, ThreadLocalRandom.current().nextFloat());
            }}
            case INTEGER->{for(long i = 0; i< resArray.getSize(); i++){
                resArray.getData().setAtIndex(ValueLayout.JAVA_INT, i, ThreadLocalRandom.current().nextInt());
            }}
            case DOUBLE->{for(long i = 0; i< resArray.getSize(); i++){
                resArray.getData().setAtIndex(ValueLayout.JAVA_DOUBLE, i, ThreadLocalRandom.current().nextDouble());
            }}
        }
        return resArray;
    }

    public static NDArray rand(int max,DType dType,int... shape){
        NDArray resArray=NDArray.zeros(dType, shape);
        switch(dType){
            case FLOAT->{for(long i = 0; i< resArray.getSize(); i++){
                resArray.getData().setAtIndex(ValueLayout.JAVA_FLOAT, i, ThreadLocalRandom.current().nextFloat(max));
            }}
            case INTEGER->{for(long i = 0; i< resArray.getSize(); i++){
                resArray.getData().setAtIndex(ValueLayout.JAVA_INT, i, ThreadLocalRandom.current().nextInt(max));
            }}
            case DOUBLE->{for(long i = 0; i< resArray.getSize(); i++){
                resArray.getData().setAtIndex(ValueLayout.JAVA_DOUBLE, i, ThreadLocalRandom.current().nextDouble(max));
            }}
        }
        return resArray;
    }

    public static NDArray rand(float max,DType dType,int... shape){
        NDArray resArray=NDArray.zeros(dType, shape);
        switch(dType){
            case FLOAT->{for(long i = 0; i< resArray.getSize(); i++){
                resArray.getData().setAtIndex(ValueLayout.JAVA_FLOAT, i, ThreadLocalRandom.current().nextFloat(max));
            }}
            case INTEGER->{for(long i = 0; i< resArray.getSize(); i++){
                throw new IllegalArgumentException(
                    "rand(float max, DType, shape) cannot generate FLOAT random values into dtype " +
                    dType + " for shape " + Arrays.toString(shape)
                );
            }}
            case DOUBLE->{for(long i = 0; i< resArray.getSize(); i++){
                resArray.getData().setAtIndex(ValueLayout.JAVA_DOUBLE, i, ThreadLocalRandom.current().nextDouble(max));
            }}
        }
        return resArray;
    }

    public static NDArray rand(double max,DType dType,int... shape){
        NDArray resArray=NDArray.zeros(dType, shape);
        switch(dType){
            case FLOAT->{for(long i = 0; i< resArray.getSize(); i++){
                throw new IllegalArgumentException(
                    "rand(double max, DType, shape) cannot generate DOUBLE random values into dtype " +
                    dType + " for shape " + Arrays.toString(shape)
                );
            }}
            case INTEGER->{for(long i = 0; i< resArray.getSize(); i++){
                throw new IllegalArgumentException(
                    "rand(double max, DType, shape) cannot generate DOUBLE random values into dtype " +
                    dType + " for shape " + Arrays.toString(shape)
                );
            }}
            case DOUBLE->{for(long i = 0; i< resArray.getSize(); i++){
                resArray.getData().setAtIndex(ValueLayout.JAVA_DOUBLE, i, ThreadLocalRandom.current().nextDouble(max));
            }}
        }
        return resArray;
    }

    public static NDArray rand(int min,int max,DType dType,int... shape){
        NDArray resArray=NDArray.zeros(dType, shape);
        switch(dType){
            case FLOAT->{for(long i = 0; i< resArray.getSize(); i++){
                resArray.getData().setAtIndex(ValueLayout.JAVA_FLOAT, i, ThreadLocalRandom.current().nextFloat(min,max));
            }}
            case INTEGER->{for(long i = 0; i< resArray.getSize(); i++){
                resArray.getData().setAtIndex(ValueLayout.JAVA_INT, i, ThreadLocalRandom.current().nextInt(min, max));
            }}
            case DOUBLE->{for(long i = 0; i< resArray.getSize(); i++){
                resArray.getData().setAtIndex(ValueLayout.JAVA_DOUBLE, i, ThreadLocalRandom.current().nextDouble(min,max));
            }}
        }
        return resArray;
    }

    public static NDArray rand(float min,float max,DType dType,int... shape){
        NDArray resArray=NDArray.zeros(dType, shape);
        switch(dType){
            case FLOAT->{for(long i = 0; i< resArray.getSize(); i++){
                resArray.getData().setAtIndex(ValueLayout.JAVA_FLOAT, i, ThreadLocalRandom.current().nextFloat(min,max));
            }}
            case INTEGER->throw new IllegalArgumentException(
                "rand(float min, float max, DType, shape) cannot generate FLOAT random values into dtype " +
                dType + " for shape " + Arrays.toString(shape)
            );
            case DOUBLE->{for(long i = 0; i< resArray.getSize(); i++){
                resArray.getData().setAtIndex(ValueLayout.JAVA_DOUBLE, i, ThreadLocalRandom.current().nextDouble(min,max));
            }}
        }
        return resArray;
    }

    public static NDArray rand(double min,double max,DType dType,int... shape){
        NDArray resArray=NDArray.zeros(dType, shape);
        switch(dType){
            case FLOAT->throw new IllegalArgumentException(
                "rand(double min, double max, DType, shape) cannot generate DOUBLE random values into dtype " +
                dType + " for shape " + Arrays.toString(shape)
            );
            case INTEGER->throw new IllegalArgumentException(
                "rand(double min, double max, DType, shape) cannot generate DOUBLE random values into dtype " +
                dType + " for shape " + Arrays.toString(shape)
            );
            case DOUBLE->{for(long i = 0; i< resArray.getSize(); i++){
                resArray.getData().setAtIndex(ValueLayout.JAVA_DOUBLE, i, ThreadLocalRandom.current().nextDouble(min,max));
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
            throw new IllegalArgumentException(
                "Requested shape " + Arrays.toString(shape) +
                " requires size " + CalcSize +
                ", but the provided array has length " + data.length + "."
            );
        }
        DType dType = DType.FLOAT;
        MemorySegment segment = arena.allocate(dType.layout, CalcSize);
        MemorySegment.copy(data, 0, segment, dType.layout, 0, data.length);
        return new NDArray(segment, shape, ShapeUtil.calculateDefaultStrides(shape), dType);
    }

    //int array from methods
    public static NDArray from(int[] data, int... shape) {
        return from(Arena.ofAuto(), data, shape);
    }

    public static NDArray from(Arena arena, int[] data, int... shape) {
        long CalcSize = 1;
        for (int dim : shape) CalcSize *= dim;
        if (CalcSize != data.length) {
            throw new IllegalArgumentException(
                "Requested shape " + Arrays.toString(shape) +
                " requires size " + CalcSize +
                ", but the provided array has length " + data.length + "."
            );
        }
        DType dType = DType.INTEGER;
        MemorySegment segment = arena.allocate(dType.layout, CalcSize);
        MemorySegment.copy(data, 0, segment, dType.layout, 0, data.length);
        return new NDArray(segment, shape, ShapeUtil.calculateDefaultStrides(shape), dType);
    }

    // DOUBLE array from method
    public static NDArray from(double[] data, int... shape) {
        return from(Arena.ofAuto(), data, shape);
    }

    public static NDArray from(Arena arena, double[] data, int... shape) {
        long CalcSize = 1;
        for (int dim : shape) CalcSize *= dim;
        if (CalcSize != data.length) {
            throw new IllegalArgumentException(
                "Requested shape " + Arrays.toString(shape) +
                " requires size " + CalcSize +
                ", but the provided array has length " + data.length + "."
            );
        }
        DType dType = DType.DOUBLE;
        MemorySegment segment = arena.allocate(dType.layout, CalcSize);
        MemorySegment.copy(data, 0, segment, dType.layout, 0, data.length);
        return new NDArray(segment, shape, ShapeUtil.calculateDefaultStrides(shape), dType);
    }

    public NDArray reshape(int... newShape) {
        long newCalcSize = 1;
        for (int dim : newShape) newCalcSize *= dim;
        if (newCalcSize != this.getSize()) {
            throw new IllegalArgumentException("Cannot reshape array of size " + this.getSize() + " into shape " + Arrays.toString(newShape));
        }
        NDArray safeThis = this.isContiguous() ? this : this.contiguous();
        return new NDArray(safeThis.getData(), newShape, ShapeUtil.calculateDefaultStrides(newShape), this.getDType());
    }

    public NDArray reshape(DType dType,int... newShape) {
        long newCalcSize = 1;
        for (int dim : newShape) newCalcSize *= dim;
        if (newCalcSize != this.getSize()) {
            throw new IllegalArgumentException("Cannot reshape array of size " + this.getSize() + " into shape " + Arrays.toString(newShape));
        }
        return this.reshape(newShape).cast(dType);
    }

    public NDArray transpose(){
        if(ndim()<2) return this;
        var newShape=new int[this.internalShapeUnsafe().length];
        var newStrides=new int[this.internalStridesUnsafe().length];
        for(int i = 0; i< this.internalShapeUnsafe().length; i++){
            newShape[i]= this.internalShapeUnsafe()[(this.internalShapeUnsafe().length-1-i)];
            newStrides[i]= this.internalStridesUnsafe()[(this.internalStridesUnsafe().length-1-i)];
        }
        return new NDArray(this.getData(),newShape,newStrides, this.getDType());
    }

    public boolean isContiguous(){
        var expStride=1;
        for(int i = this.internalShapeUnsafe().length-1; i>=0; i--){
            if(this.internalStridesUnsafe()[i]!=expStride) return false;
            expStride*= internalShapeUnsafe()[i];
        }
        return true;
    }

    public NDArray contiguous(){
        return contiguous(Arena.ofAuto());
    }

    public NDArray contiguous(Arena arena){
        if(this.isContiguous()) return this;
        var segment=arena.allocate(this.getDType().layout, this.getSize());
        var newStrides=ShapeUtil.calculateDefaultStrides(this.internalShapeUnsafe());
        for(int i = 0; i< this.getSize(); i++){
            var tempindex=i;
            var coord=new int[this.internalShapeUnsafe().length];
            for(int j = this.internalShapeUnsafe().length-1; j>=0; j--){
                coord[j]=tempindex% this.internalShapeUnsafe()[j];
                tempindex=tempindex/ this.internalShapeUnsafe()[j];
            }
            long oldFlatIndex = 0;
            for(int d = 0; d < this.internalShapeUnsafe().length; d++){
                oldFlatIndex += coord[d] * this.internalStridesUnsafe()[d];
            }
            switch(getDType()){
                case FLOAT->{
                    var val= this.getData().getAtIndex(ValueLayout.JAVA_FLOAT, oldFlatIndex);
                    segment.setAtIndex(ValueLayout.JAVA_FLOAT, i, val);
                }
                case INTEGER->{
                    var val= this.getData().getAtIndex(ValueLayout.JAVA_INT, oldFlatIndex);
                    segment.setAtIndex(ValueLayout.JAVA_INT, i, val);
                }
                case DOUBLE->{
                    var val= this.getData().getAtIndex(ValueLayout.JAVA_DOUBLE, oldFlatIndex);
                    segment.setAtIndex(ValueLayout.JAVA_DOUBLE, i, val);
                }
            }
        }
        return new NDArray(segment, this.internalShapeUnsafe(), newStrides, getDType());
    }

    public NDArray broadcastTo(int ... shape){
        if(Arrays.equals(this.internalShapeUnsafe(),shape)) return this;
        int ndim=shape.length;
        if(ndim<this.ndim()){
            throw new IllegalArgumentException(
                "Cannot broadcast array of shape " + Arrays.toString(this.internalShapeUnsafe()) +
                " to target shape " + Arrays.toString(shape) +
                " because the target has fewer dimensions."
            );
        }
        int[] newStrides=new int[ndim];
        int[] paddedShape=new int[ndim];
        int[] paddedStrides=new int[ndim];
        int offset=ndim-this.ndim();
        for(int i=0;i<ndim;i++){
            if(i<offset){
                paddedShape[i]=1;
                paddedStrides[i]=0;
            }
            else{
                paddedShape[i]= this.internalShapeUnsafe()[i-offset];
                paddedStrides[i]= this.internalStridesUnsafe()[i-offset];
            }
        }

        for(int i=0;i<ndim;i++){
            if(paddedShape[i]==shape[i]) newStrides[i]=paddedStrides[i];
            else if(paddedShape[i]==1) newStrides[i]=0;
            else {
                throw new IllegalArgumentException(
                    "Cannot broadcast array of shape " + Arrays.toString(this.internalShapeUnsafe()) +
                    " to target shape " + Arrays.toString(shape) +
                    " because dimension " + i + " is incompatible: source dimension " +
                    paddedShape[i] + " cannot expand to " + shape[i]
                );
            }
        }

        return new NDArray(this.getData(), shape, newStrides, this.getDType());
    }

    public NDArray cast(DType target){
        if (this.getDType() == target) {
            return this;
        }

        NDArray safeThis = this.isContiguous() ? this : this.contiguous();
        NDArray res = NDArray.zeros(target, safeThis.internalShapeUnsafe());
        for(long i = 0; i < safeThis.getSize(); i++){
            double val=switch (safeThis.getDType()){
                case FLOAT -> safeThis.getData().getAtIndex(ValueLayout.JAVA_FLOAT, i);
                case DOUBLE -> safeThis.getData().getAtIndex(ValueLayout.JAVA_DOUBLE, i);
                case INTEGER -> safeThis.getData().getAtIndex(ValueLayout.JAVA_INT, i);
            };
            switch (target) {
                case FLOAT -> res.getData().setAtIndex(ValueLayout.JAVA_FLOAT, i, (float) val);
                case DOUBLE -> res.getData().setAtIndex(ValueLayout.JAVA_DOUBLE, i, val);
                case INTEGER -> res.getData().setAtIndex(ValueLayout.JAVA_INT, i, (int) val);
            }
        }
        return res;
    }


    public String shapeString() {
        return Arrays.toString(internalShapeUnsafe()).replace("[", "(").replace("]", ")");
    }

    public NDArray copy(){
        NDArray dups=NDArray.zeros(this.getDType(), this.internalShapeUnsafe());
        if (this.isContiguous()) {
            long logicalBytes = this.getSize() * this.getDType().layout.byteSize();
            MemorySegment.copy(this.getData(), 0, dups.getData(), 0, logicalBytes);
            return dups;
        }
        NDIter srcIter = new NDIter(this.internalShapeUnsafe());
        long dstIndex = 0;
        while(srcIter.hasNext){
            long byteOffset = ShapeUtil.getByteOffset(srcIter.coords, this.internalStridesUnsafe(), this.getDType());
            switch(this.getDType()){
                case FLOAT -> dups.getData().setAtIndex(ValueLayout.JAVA_FLOAT, dstIndex, this.getData().get(ValueLayout.JAVA_FLOAT, byteOffset));
                case INTEGER -> dups.getData().setAtIndex(ValueLayout.JAVA_INT, dstIndex, this.getData().get(ValueLayout.JAVA_INT, byteOffset));
                case DOUBLE -> dups.getData().setAtIndex(ValueLayout.JAVA_DOUBLE, dstIndex, this.getData().get(ValueLayout.JAVA_DOUBLE, byteOffset));
            }
            dstIndex++;
            srcIter.next();
        }
        return dups;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof NDArray)) return false;
        NDArray other = (NDArray) o;

        if (this.getDType() != other.getDType()) return false;
        if (!Arrays.equals(this.internalShapeUnsafe(), other.internalShapeUnsafe())) return false;

        long logicalBytes = this.getSize() * this.getDType().layout.byteSize();

        if (this.isContiguous() && other.isContiguous()) {
            var sliceThis = this.getData().asSlice(0, logicalBytes);
            var sliceOther = other.getData().asSlice(0, logicalBytes);
            return sliceThis.mismatch(sliceOther) == -1;
        }

        for (long i = 0; i < this.getSize(); i++) {
            long offsetThis = getPhysicalOffset(i, this.internalShapeUnsafe(), this.internalStridesUnsafe());
            long offsetOther = getPhysicalOffset(i, other.internalShapeUnsafe(), other.internalStridesUnsafe());

            boolean match = switch(this.getDType()) {
                case FLOAT -> this.getData().getAtIndex(ValueLayout.JAVA_FLOAT, offsetThis) == other.getData().getAtIndex(ValueLayout.JAVA_FLOAT, offsetOther);
                case DOUBLE -> this.getData().getAtIndex(ValueLayout.JAVA_DOUBLE, offsetThis) == other.getData().getAtIndex(ValueLayout.JAVA_DOUBLE, offsetOther);
                case INTEGER -> this.getData().getAtIndex(ValueLayout.JAVA_INT, offsetThis) == other.getData().getAtIndex(ValueLayout.JAVA_INT, offsetOther);
            };
            if (!match) return false;
        }
        return true;
    }

    @Override
    public int hashCode() {
        int result = Arrays.hashCode(internalShapeUnsafe());
        result = 31 * result + getDType().hashCode();

        int elementsToHash = (int) Math.min(getSize(), 5);
        for (long i = 0; i < elementsToHash; i++) {
            long physicalOffset = getPhysicalOffset(i, this.internalShapeUnsafe(), this.internalStridesUnsafe());
            int valHash = switch (getDType()) {
                case FLOAT -> Float.hashCode(getData().getAtIndex(ValueLayout.JAVA_FLOAT, physicalOffset));
                case DOUBLE -> Double.hashCode(getData().getAtIndex(ValueLayout.JAVA_DOUBLE, physicalOffset));
                case INTEGER -> Integer.hashCode(getData().getAtIndex(ValueLayout.JAVA_INT, physicalOffset));
            };
            result = 31 * result + valHash;
        }
        return result;
    }

    private static long getPhysicalOffset(long logicalIndex, int[] shape, int[] strides) {
        long remaining = logicalIndex;
        long offset = 0;
        for (int i = shape.length - 1; i >= 0; i--) {
            int coord = (int) (remaining % shape[i]);
            remaining /= shape[i];
            offset += (long) coord * strides[i];
        }
        return offset;
    }

    private void validateFlatIndex(long index) {
        if (index < 0 || index >= this.getSize()) {
            throw new IndexOutOfBoundsException("Flat index " + index + " is out of bounds for size " + this.getSize());
        }
    }

    public double getFlat(long index){
        validateFlatIndex(index);
        long physicalOffset = this.isContiguous() ? index : getPhysicalOffset(index, this.internalShapeUnsafe(), this.internalStridesUnsafe());
        return switch(this.getDType()){
            case FLOAT -> getData().getAtIndex(ValueLayout.JAVA_FLOAT, physicalOffset);
            case INTEGER -> getData().getAtIndex(ValueLayout.JAVA_INT, physicalOffset);
            case DOUBLE -> getData().getAtIndex(ValueLayout.JAVA_DOUBLE, physicalOffset);
            default -> throw new AssertionError();
        };
    }

    public float getFlatFloat(long index) {
        validateFlatIndex(index);
        long physicalOffset = this.isContiguous() ? index : getPhysicalOffset(index, this.internalShapeUnsafe(), this.internalStridesUnsafe());
        return getData().getAtIndex(ValueLayout.JAVA_FLOAT, physicalOffset);
    }

    public int getFlatInt(long index){
        validateFlatIndex(index);
        long physicalOffset = this.isContiguous() ? index : getPhysicalOffset(index, this.internalShapeUnsafe(), this.internalStridesUnsafe());
        return getData().getAtIndex(ValueLayout.JAVA_INT, physicalOffset);
    }

    public double getFlatDouble(long index){
        validateFlatIndex(index);
        long physicalOffset = this.isContiguous() ? index : getPhysicalOffset(index, this.internalShapeUnsafe(), this.internalStridesUnsafe());
        return getData().getAtIndex(ValueLayout.JAVA_DOUBLE, physicalOffset);
    }

    public double get(int... indices){
        if(indices.length!= internalShapeUnsafe().length){
            throw new IllegalArgumentException("illegal indices :"+indices.length+" does not match with shape "+ internalShapeUnsafe().length);
        }
        long flatIndex=0;
        for(int i=0;i<indices.length;i++){
            if (indices[i] < 0 || indices[i] >= internalShapeUnsafe()[i]) {
                throw new IndexOutOfBoundsException("Index " + indices[i] + " is out of bounds for dimension " + i + " with size " + internalShapeUnsafe()[i]);
            }
            flatIndex+=(long)indices[i]* internalStridesUnsafe()[i];
        }
        return switch(this.getDType()){
            case FLOAT -> getData().getAtIndex(ValueLayout.JAVA_FLOAT, flatIndex);
            case INTEGER -> getData().getAtIndex(ValueLayout.JAVA_INT, flatIndex);
            case DOUBLE -> getData().getAtIndex(ValueLayout.JAVA_DOUBLE, flatIndex);
        };
    }

    public int getInt(int... indices){
        if(indices.length!= internalShapeUnsafe().length){
            throw new IllegalArgumentException("illegal indices :"+indices.length+" does not match with shape "+ internalShapeUnsafe().length);
        }
        long flatIndex=0;
        for(int i=0;i<indices.length;i++){
            if (indices[i] < 0 || indices[i] >= internalShapeUnsafe()[i]) {
                throw new IndexOutOfBoundsException("Index " + indices[i] + " is out of bounds for dimension " + i + " with size " + internalShapeUnsafe()[i]);
            }
            flatIndex+=(long)indices[i]* internalStridesUnsafe()[i];
        }
        return getData().getAtIndex(ValueLayout.JAVA_INT, flatIndex);
    }

    public float getFloat(int... indices){
        if(indices.length!= internalShapeUnsafe().length){
            throw new IllegalArgumentException("illegal indices :"+indices.length+" does not match with shape "+ internalShapeUnsafe().length);
        }
        long flatIndex=0;
        for(int i=0;i<indices.length;i++){
            if (indices[i] < 0 || indices[i] >= internalShapeUnsafe()[i]) {
                throw new IndexOutOfBoundsException("Index " + indices[i] + " is out of bounds for dimension " + i + " with size " + internalShapeUnsafe()[i]);
            }
            flatIndex+=(long)indices[i]* internalStridesUnsafe()[i];
        }
        return getData().getAtIndex(ValueLayout.JAVA_FLOAT, flatIndex);
    }

    public int[] indexOf(double b){
        NDIter iter = new NDIter(this.internalShapeUnsafe());
        switch(getDType()){
            case FLOAT->{
                var c= (float)b;
                var epsilon=1e-6f;
                while(iter.hasNext){
                    long byteOffset = ShapeUtil.getByteOffset(iter.coords, this.internalStridesUnsafe(), this.getDType());
                    float val = this.getData().get(ValueLayout.JAVA_FLOAT, byteOffset);
                    if(Math.abs(c - val) < epsilon){
                        return iter.coords.clone();
                    }
                    iter.next();
                }
            }
            case INTEGER->{
                var c= (int)b;
                while(iter.hasNext){
                    long byteOffset = ShapeUtil.getByteOffset(iter.coords, this.internalStridesUnsafe(), this.getDType());
                    if(c== this.getData().get(ValueLayout.JAVA_INT, byteOffset)){
                        return iter.coords.clone();
                    }
                    iter.next();
                }
            }
            case DOUBLE->{
                var c= b;
                var epsilon=1e-12;
                while(iter.hasNext){
                    long byteOffset = ShapeUtil.getByteOffset(iter.coords, this.internalStridesUnsafe(), this.getDType());
                    var val= this.getData().get(ValueLayout.JAVA_DOUBLE, byteOffset);
                    if(Math.abs(c - val) < epsilon){
                        return iter.coords.clone();
                    }
                    iter.next();
                }
            }
        }
        throw new NoSuchElementException();    
    }

    public int[] shape() {
        return internalShapeUnsafe().clone();
    }

    public int ndim() {
        return internalShapeUnsafe().length;
    }


    public DType getDType(){
        return dtype;
    }

    public int[] internalStridesUnsafe() {
        return strides;
    }

    public int[] internalShapeUnsafe() {
        return shape;
    }

    public MemorySegment getData() {
        return data;
    }

    public MemorySegment getDataReadOnly(){
        return data.asReadOnly();
    }

    public long getSize() {
        return size;
    }

    @Override
    public String toString(){
        StringBuilder sb = new StringBuilder("NDArray" + shapeString() + " [");
        int maxPrint = 6;
        NDIter iter = new NDIter(this.internalShapeUnsafe());
        for (long i = 0; i < getSize(); i++) {
            if (i == maxPrint / 2 && getSize() > maxPrint) {
                sb.append("..., ");
            } else if (getSize() <= maxPrint || i < maxPrint / 2 || i >= getSize() - (maxPrint / 2)) {
                long byteOffset = ShapeUtil.getByteOffset(iter.coords, this.internalStridesUnsafe(), this.getDType());
                switch(this.getDType()){
                    case INTEGER->sb.append(getData().get(ValueLayout.JAVA_INT, byteOffset));
                    case FLOAT->sb.append(getData().get(ValueLayout.JAVA_FLOAT, byteOffset));
                    case DOUBLE->sb.append(getData().get(ValueLayout.JAVA_DOUBLE, byteOffset));
                }
                if (i < getSize() - 1) sb.append(", ");
            }
            iter.next();
        }
        return sb.append("]").toString();
    }

    public double max() {
        return switch(this.getDType()) {
            case FLOAT -> ReduceOps.maxFloat(this);
            case DOUBLE -> ReduceOps.maxDouble(this);
            case INTEGER -> ReduceOps.maxInt(this);
        };
    }

    public double min() {
        return switch(this.getDType()) {
            case FLOAT -> ReduceOps.minFloat(this);
            case DOUBLE -> ReduceOps.minDouble(this);
            case INTEGER -> ReduceOps.minInt(this);
        };
    }

    public double sum() {
        return switch(this.getDType()) {
            case FLOAT -> ReduceOps.sumFloat(this);
            case DOUBLE -> ReduceOps.sumDouble(this);
            case INTEGER -> ReduceOps.sumInt(this);
        };
    }

    public NDArray sum(int axis){
        int[] reducedShape = ShapeUtil.calculateReductionShape(this.internalShapeUnsafe(), axis);
        NDArray resArray = NDArray.zeros(this.getDType(), reducedShape);
        return switch(this.getDType()) {
            case FLOAT -> ReduceOps.sumFloatAxis(this,axis,resArray);
            case DOUBLE -> ReduceOps.sumDoubleAxis(this,axis,resArray);
            case INTEGER -> ReduceOps.sumIntAxis(this,axis,resArray);
        };
    }

    public NDArray max(int axis) {
        int[] reducedShape = ShapeUtil.calculateReductionShape(this.internalShapeUnsafe(), axis);
        NDArray resArray = NDArray.zeros(this.getDType(), reducedShape);
        return switch(this.getDType()) {
            case FLOAT -> ReduceOps.maxFloatAxis(this, axis, resArray);
            case DOUBLE -> ReduceOps.maxDoubleAxis(this, axis, resArray);
            case INTEGER -> ReduceOps.maxIntAxis(this, axis, resArray);
        };
    }

    public NDArray min(int axis) {
        int[] reducedShape = ShapeUtil.calculateReductionShape(this.internalShapeUnsafe(), axis);
        NDArray resArray = NDArray.zeros(this.getDType(), reducedShape);
        return switch(this.getDType()) {
            case FLOAT -> ReduceOps.minFloatAxis(this, axis, resArray);
            case DOUBLE -> ReduceOps.minDoubleAxis(this, axis, resArray);
            case INTEGER -> ReduceOps.minIntAxis(this, axis, resArray);
        };
    }

    public double dot(NDArray b){
        if (this.ndim() != 1 || b.ndim() != 1) {
            throw new IllegalArgumentException("Dot product requires 1D vectors. Shapes: " + this.shapeString() + ", " + b.shapeString());
        }
        if (this.getSize() != b.getSize()) {
            throw new IllegalArgumentException("Vector sizes must match for dot product.");
        }
        DType targetType = TypeUtil.promoteTypes(this.getDType(), b.getDType());
        NDArray A = this.cast(targetType);
        NDArray B = b.cast(targetType);
        if (!A.isContiguous()) {
            A = A.contiguous();
        }
        if (!B.isContiguous()) {
            B = B.contiguous();
        }
        return switch(targetType){
            case FLOAT->ReduceOps.dotFloat(A, B);
            case INTEGER->ReduceOps.dotInt(A, B);
            case DOUBLE->ReduceOps.dotDouble(A, B);
        };
    }

    public double avg() {
        return this.sum() / (double) this.getSize();
    }

    public NDArray maximum(NDArray b){
        DType targetType = TypeUtil.promoteTypes(this.getDType(), b.getDType());
        int[] targetShape = ShapeUtil.calculateBroadcastShape(this.internalShapeUnsafe(), b.internalShapeUnsafe());
        NDArray A = ValidUtil.prepareBroadcastOperand(this, targetShape, targetType);
        NDArray B = ValidUtil.prepareBroadcastOperand(b, targetShape, targetType);
        NDArray resArray = NDArray.zeros(targetType, targetShape);

        return switch(targetType){
            case FLOAT->CompareOps.maximumFloat(A, B, resArray);
            case INTEGER->CompareOps.maximumInt(A, B, resArray);
            case DOUBLE->CompareOps.maximumDouble(A, B, resArray);
        };
    }

    public NDArray maximum(float b) {
        DType targetType = TypeUtil.promoteTypes(this.getDType(), TypeUtil.scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray resArray = NDArray.zeros(targetType, this.internalShapeUnsafe());
        
        return switch (targetType) {
            case FLOAT -> CompareOps.maximumFloat(A, b, resArray);
            case INTEGER->throw new UnsupportedOperationException();
            case DOUBLE->CompareOps.maximumDouble(A, b, resArray);
        };
    }

    public NDArray maximum(int b) {
        DType targetType = TypeUtil.promoteTypes(this.getDType(), TypeUtil.scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray resArray = NDArray.zeros(targetType, this.internalShapeUnsafe());
        
        return switch (targetType) {
            case FLOAT -> CompareOps.maximumFloat(A, b, resArray);
            case INTEGER->CompareOps.maximumInt(A, b, resArray);
            case DOUBLE->CompareOps.maximumDouble(A, b, resArray);
        };
    }

    public NDArray maximum(double b) {
        DType targetType = TypeUtil.promoteTypes(this.getDType(), TypeUtil.scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray resArray = NDArray.zeros(targetType, this.internalShapeUnsafe());
        
        return switch (targetType) {
            case FLOAT -> throw new UnsupportedOperationException();
            case INTEGER-> throw new UnsupportedOperationException();
            case DOUBLE-> CompareOps.maximumDouble(A, b, resArray);
        };
    }

    public NDArray minimum(NDArray b){
        DType targetType = TypeUtil.promoteTypes(this.getDType(), b.getDType());
        int[] targetShape = ShapeUtil.calculateBroadcastShape(this.internalShapeUnsafe(), b.internalShapeUnsafe());
        NDArray A = ValidUtil.prepareBroadcastOperand(this, targetShape, targetType);
        NDArray B = ValidUtil.prepareBroadcastOperand(b, targetShape, targetType);
        NDArray resArray = NDArray.zeros(targetType, targetShape);

        return switch(targetType){
            case FLOAT->CompareOps.minimumFloat(A, B, resArray);
            case INTEGER->CompareOps.minimumInt(A, B, resArray);
            case DOUBLE->CompareOps.minimumDouble(A, B, resArray);
        };
    }

    public NDArray minimum(float b) {
        DType targetType = TypeUtil.promoteTypes(this.getDType(), TypeUtil.scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray resArray = NDArray.zeros(targetType, this.internalShapeUnsafe());
        
        return switch (targetType) {
            case FLOAT -> CompareOps.minimumFloat(A, b, resArray);
            case INTEGER->throw new UnsupportedOperationException();
            case DOUBLE->CompareOps.minimumDouble(A, b, resArray);
        };
    }

    public NDArray minimum(int b) {
        DType targetType = TypeUtil.promoteTypes(this.getDType(), TypeUtil.scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray resArray = NDArray.zeros(targetType, this.internalShapeUnsafe());
        
        return switch (targetType) {
            case FLOAT -> CompareOps.minimumFloat(A, b, resArray);
            case INTEGER->CompareOps.minimumInt(A, b, resArray);
            case DOUBLE->CompareOps.minimumDouble(A, b, resArray);
        };
    }

    public NDArray minimum(double b) {
        DType targetType = TypeUtil.promoteTypes(this.getDType(), TypeUtil.scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray resArray = NDArray.zeros(targetType, this.internalShapeUnsafe());
        
        return switch (targetType) {
            case FLOAT -> throw new UnsupportedOperationException();
            case INTEGER-> throw new UnsupportedOperationException();
            case DOUBLE-> CompareOps.minimumDouble(A, b, resArray);
        };
    }

    //ArithmaticOps.java 

    //addition operation

    public NDArray add(NDArray b){
        DType targetType = TypeUtil.promoteTypes(this.getDType(), b.getDType());
        int[] targetShape = ShapeUtil.calculateBroadcastShape(this.internalShapeUnsafe(), b.internalShapeUnsafe());
        NDArray A = ValidUtil.prepareBroadcastOperand(this, targetShape, targetType);
        NDArray B = ValidUtil.prepareBroadcastOperand(b, targetShape, targetType);
        NDArray resArray = NDArray.zeros(targetType, targetShape);
        return switch(targetType) {
            case FLOAT -> ArithmeticOps.addFloat(A, B, resArray);
            case DOUBLE -> ArithmeticOps.addDouble(A, B, resArray);
            case INTEGER -> ArithmeticOps.addInt(A, B, resArray);
        };
    }
    
    public NDArray add(NDArray b,NDArray resArray){
        DType targetType = TypeUtil.promoteTypes(this.getDType(), b.getDType());
        int[] targetShape = ShapeUtil.calculateBroadcastShape(this.internalShapeUnsafe(), b.internalShapeUnsafe());
        NDArray A = ValidUtil.prepareBroadcastOperand(this, targetShape, targetType);
        NDArray B = ValidUtil.prepareBroadcastOperand(b, targetShape, targetType);
        NDArray targetRes = ValidUtil.validateResultArray(resArray, targetType, targetShape);
        return switch(targetType) {
            case FLOAT -> ArithmeticOps.addFloat(A, B, targetRes);
            case DOUBLE -> ArithmeticOps.addDouble(A, B, targetRes);
            case INTEGER -> ArithmeticOps.addInt(A, B, targetRes);
        };
    }

    public NDArray add(float b){
        DType targetType = TypeUtil.promoteTypes(this.getDType(), TypeUtil.scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray resArray = NDArray.zeros(targetType, this.internalShapeUnsafe());
        return switch (targetType) {
            case FLOAT -> ArithmeticOps.addFloat(A, b, resArray);
            case DOUBLE -> ArithmeticOps.addDouble(A, b, resArray);
            case INTEGER -> ArithmeticOps.addInt(A, (int) b, resArray);
        };
    }

    public NDArray add(int b){
        DType targetType = TypeUtil.promoteTypes(this.getDType(), TypeUtil.scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray resArray = NDArray.zeros(targetType, this.internalShapeUnsafe());
        return switch (targetType) {
            case FLOAT -> ArithmeticOps.addFloat(A, b, resArray);
            case DOUBLE -> ArithmeticOps.addDouble(A, b, resArray);
            case INTEGER -> ArithmeticOps.addInt(A, b, resArray);
        };
    }

    public NDArray add(double b){
        DType targetType = TypeUtil.promoteTypes(this.getDType(), TypeUtil.scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray resArray = NDArray.zeros(targetType, this.internalShapeUnsafe());
        return switch (targetType) {
            case FLOAT -> ArithmeticOps.addFloat(A, (float) b, resArray);
            case DOUBLE -> ArithmeticOps.addDouble(A, b, resArray);
            case INTEGER -> ArithmeticOps.addInt(A, (int) b, resArray);
        };
    }

    public NDArray add(float b,NDArray resArray){
        DType targetType = TypeUtil.promoteTypes(this.getDType(), TypeUtil.scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray targetRes = ValidUtil.validateResultArray(resArray, targetType, this.internalShapeUnsafe());
        return switch (targetType) {
            case FLOAT -> ArithmeticOps.addFloat(A, b, targetRes);
            case DOUBLE -> ArithmeticOps.addDouble(A, b, targetRes);
            case INTEGER -> ArithmeticOps.addInt(A, (int) b, targetRes);
        };
    }

    public NDArray add(int b,NDArray resArray){
        DType targetType = TypeUtil.promoteTypes(this.getDType(), TypeUtil.scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray targetRes = ValidUtil.validateResultArray(resArray, targetType, this.internalShapeUnsafe());
        return switch (targetType) {
            case FLOAT -> ArithmeticOps.addFloat(A, b, targetRes);
            case DOUBLE -> ArithmeticOps.addDouble(A, b, targetRes);
            case INTEGER -> ArithmeticOps.addInt(A, b, targetRes);
        };
    }

    public NDArray add(double b,NDArray resArray){
        DType targetType = TypeUtil.promoteTypes(this.getDType(), TypeUtil.scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray targetRes = ValidUtil.validateResultArray(resArray, targetType, this.internalShapeUnsafe());
        return switch (targetType) {
            case FLOAT -> ArithmeticOps.addFloat(A, (float) b, targetRes);
            case DOUBLE -> ArithmeticOps.addDouble(A, b, targetRes);
            case INTEGER -> ArithmeticOps.addInt(A, (int) b, targetRes);
        };
    }

    //subtract operations

    public NDArray sub(NDArray b){
        DType targetType = TypeUtil.promoteTypes(this.getDType(), b.getDType());
        int[] targetShape = ShapeUtil.calculateBroadcastShape(this.internalShapeUnsafe(), b.internalShapeUnsafe());
        NDArray A = ValidUtil.prepareBroadcastOperand(this, targetShape, targetType);
        NDArray B = ValidUtil.prepareBroadcastOperand(b, targetShape, targetType);
        NDArray resArray = NDArray.zeros(targetType, targetShape);
        return switch(targetType) {
            case FLOAT -> ArithmeticOps.subFloat(A, B, resArray);
            case DOUBLE -> ArithmeticOps.subDouble(A, B, resArray);
            case INTEGER -> ArithmeticOps.subInt(A, B, resArray);
        };
    }

    public NDArray sub(NDArray b,NDArray resArray){
        DType targetType = TypeUtil.promoteTypes(this.getDType(), b.getDType());
        int[] targetShape = ShapeUtil.calculateBroadcastShape(this.internalShapeUnsafe(), b.internalShapeUnsafe());
        NDArray A = ValidUtil.prepareBroadcastOperand(this, targetShape, targetType);
        NDArray B = ValidUtil.prepareBroadcastOperand(b, targetShape, targetType);
        NDArray targetRes = ValidUtil.validateResultArray(resArray, targetType, targetShape);
        return switch(targetType) {
            case FLOAT -> ArithmeticOps.subFloat(A, B, targetRes);
            case DOUBLE -> ArithmeticOps.subDouble(A, B, targetRes);
            case INTEGER -> ArithmeticOps.subInt(A, B, targetRes);
        };
    }

    public NDArray sub(float b){
        DType targetType = TypeUtil.promoteTypes(this.getDType(), TypeUtil.scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray resArray = NDArray.zeros(targetType, this.internalShapeUnsafe());
        return switch (targetType) {
            case FLOAT -> ArithmeticOps.subFloat(A, b, resArray);
            case DOUBLE -> ArithmeticOps.subDouble(A, b, resArray);
            case INTEGER -> ArithmeticOps.subInt(A, (int) b, resArray);
        };
    }

    public NDArray sub(int b){
        DType targetType = TypeUtil.promoteTypes(this.getDType(), TypeUtil.scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray resArray = NDArray.zeros(targetType, this.internalShapeUnsafe());
        return switch (targetType) {
            case FLOAT -> ArithmeticOps.subFloat(A, b, resArray);
            case DOUBLE -> ArithmeticOps.subDouble(A, b, resArray);
            case INTEGER -> ArithmeticOps.subInt(A, b, resArray);
        };
    }

    public NDArray sub(double b){
        DType targetType = TypeUtil.promoteTypes(this.getDType(), TypeUtil.scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray resArray = NDArray.zeros(targetType, this.internalShapeUnsafe());
        return switch (targetType) {
            case FLOAT -> ArithmeticOps.subFloat(A, (float) b, resArray);
            case DOUBLE -> ArithmeticOps.subDouble(A, b, resArray);
            case INTEGER -> ArithmeticOps.subInt(A, (int) b, resArray);
        };
    }

    public NDArray sub(float b,NDArray resArray){
        DType targetType = TypeUtil.promoteTypes(this.getDType(), TypeUtil.scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray targetRes = ValidUtil.validateResultArray(resArray, targetType, this.internalShapeUnsafe());
        return switch (targetType) {
            case FLOAT -> ArithmeticOps.subFloat(A, b, targetRes);
            case DOUBLE -> ArithmeticOps.subDouble(A, b, targetRes);
            case INTEGER -> ArithmeticOps.subInt(A, (int) b, targetRes);
        };
    }

    public NDArray sub(int b,NDArray resArray){
        DType targetType = TypeUtil.promoteTypes(this.getDType(), TypeUtil.scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray targetRes = ValidUtil.validateResultArray(resArray, targetType, this.internalShapeUnsafe());
        return switch (targetType) {
            case FLOAT -> ArithmeticOps.subFloat(A, b, targetRes);
            case DOUBLE -> ArithmeticOps.subDouble(A, b, targetRes);
            case INTEGER -> ArithmeticOps.subInt(A, b, targetRes);
        };
    }

    public NDArray sub(double b,NDArray resArray){
        DType targetType = TypeUtil.promoteTypes(this.getDType(), TypeUtil.scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray targetRes = ValidUtil.validateResultArray(resArray, targetType, this.internalShapeUnsafe());
        return switch (targetType) {
            case FLOAT -> ArithmeticOps.subFloat(A, (float) b, targetRes);
            case DOUBLE -> ArithmeticOps.subDouble(A, b, targetRes);
            case INTEGER -> ArithmeticOps.subInt(A, (int) b, targetRes);
        };
    }

    //multiplication operations 

    public NDArray mul(NDArray b){
        DType targetType = TypeUtil.promoteTypes(this.getDType(), b.getDType());
        int[] targetShape = ShapeUtil.calculateBroadcastShape(this.internalShapeUnsafe(), b.internalShapeUnsafe());
        NDArray A = ValidUtil.prepareBroadcastOperand(this, targetShape, targetType);
        NDArray B = ValidUtil.prepareBroadcastOperand(b, targetShape, targetType);
        NDArray resArray = NDArray.zeros(targetType, targetShape);
        return switch(targetType) {
            case FLOAT -> ArithmeticOps.mulFloat(A, B, resArray);
            case DOUBLE -> ArithmeticOps.mulDouble(A, B, resArray);
            case INTEGER -> ArithmeticOps.mulInt(A, B, resArray);
        };
    }

    public NDArray mul(NDArray b, NDArray resArray){
        DType targetType = TypeUtil.promoteTypes(this.getDType(), b.getDType());
        int[] targetShape = ShapeUtil.calculateBroadcastShape(this.internalShapeUnsafe(), b.internalShapeUnsafe());
        NDArray A = ValidUtil.prepareBroadcastOperand(this, targetShape, targetType);
        NDArray B = ValidUtil.prepareBroadcastOperand(b, targetShape, targetType);
        NDArray targetRes = ValidUtil.validateResultArray(resArray, targetType, targetShape);
        return switch(targetType) {
            case FLOAT -> ArithmeticOps.mulFloat(A, B, targetRes);
            case DOUBLE -> ArithmeticOps.mulDouble(A, B, targetRes);
            case INTEGER -> ArithmeticOps.mulInt(A, B, targetRes);
        };
    }

    public NDArray mul(float b){
        DType targetType = TypeUtil.promoteTypes(this.getDType(), TypeUtil.scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray resArray = NDArray.zeros(targetType, this.internalShapeUnsafe());
        return switch (targetType) {
            case FLOAT -> ArithmeticOps.mulFloat(A, b, resArray);
            case DOUBLE -> ArithmeticOps.mulDouble(A, b, resArray);
            case INTEGER -> ArithmeticOps.mulInt(A, (int) b, resArray);
        };
    }

    public NDArray mul(int b){
        DType targetType = TypeUtil.promoteTypes(this.getDType(), TypeUtil.scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray resArray = NDArray.zeros(targetType, this.internalShapeUnsafe());
        return switch (targetType) {
            case FLOAT -> ArithmeticOps.mulFloat(A, b, resArray);
            case DOUBLE -> ArithmeticOps.mulDouble(A, b, resArray);
            case INTEGER -> ArithmeticOps.mulInt(A, b, resArray);
        };
    }

    public NDArray mul(double b){
        DType targetType = TypeUtil.promoteTypes(this.getDType(), TypeUtil.scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray resArray = NDArray.zeros(targetType, this.internalShapeUnsafe());
        return switch (targetType) {
            case FLOAT -> ArithmeticOps.mulFloat(A, (float) b, resArray);
            case DOUBLE -> ArithmeticOps.mulDouble(A, b, resArray);
            case INTEGER -> ArithmeticOps.mulInt(A, (int) b, resArray);
        };
    }

    public NDArray mul(float b,NDArray resArray){
        DType targetType = TypeUtil.promoteTypes(this.getDType(), TypeUtil.scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray targetRes = ValidUtil.validateResultArray(resArray, targetType, this.internalShapeUnsafe());
        return switch (targetType) {
            case FLOAT -> ArithmeticOps.mulFloat(A, b, targetRes);
            case DOUBLE -> ArithmeticOps.mulDouble(A, b, targetRes);
            case INTEGER -> ArithmeticOps.mulInt(A, (int) b, targetRes);
        };
    }

    public NDArray mul(int b,NDArray resArray){
        DType targetType = TypeUtil.promoteTypes(this.getDType(), TypeUtil.scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray targetRes = ValidUtil.validateResultArray(resArray, targetType, this.internalShapeUnsafe());
        return switch (targetType) {
            case FLOAT -> ArithmeticOps.mulFloat(A, b, targetRes);
            case DOUBLE -> ArithmeticOps.mulDouble(A, b, targetRes);
            case INTEGER -> ArithmeticOps.mulInt(A, b, targetRes);
        };
    }

    public NDArray mul(double b,NDArray resArray){
        DType targetType = TypeUtil.promoteTypes(this.getDType(), TypeUtil.scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray targetRes = ValidUtil.validateResultArray(resArray, targetType, this.internalShapeUnsafe());
        return switch (targetType) {
            case FLOAT -> ArithmeticOps.mulFloat(A, (float) b, targetRes);
            case DOUBLE -> ArithmeticOps.mulDouble(A, b, targetRes);
            case INTEGER -> ArithmeticOps.mulInt(A, (int) b, targetRes);
        };
    }

    //division operations

    public NDArray div(NDArray b){
        DType targetType = TypeUtil.promoteTypes(this.getDType(), b.getDType());
        int[] targetShape = ShapeUtil.calculateBroadcastShape(this.internalShapeUnsafe(), b.internalShapeUnsafe());
        NDArray A = ValidUtil.prepareBroadcastOperand(this, targetShape, targetType);
        NDArray B = ValidUtil.prepareBroadcastOperand(b, targetShape, targetType);
        NDArray resArray = NDArray.zeros(targetType, targetShape);
        return switch(targetType) {
            case FLOAT -> ArithmeticOps.divFloat(A, B, resArray);
            case DOUBLE -> ArithmeticOps.divDouble(A, B, resArray);
            case INTEGER -> ArithmeticOps.divInt(A, B, resArray);
        };
    }

    public NDArray div(NDArray b, NDArray resArray){
        DType targetType = TypeUtil.promoteTypes(this.getDType(), b.getDType());
        int[] targetShape = ShapeUtil.calculateBroadcastShape(this.internalShapeUnsafe(), b.internalShapeUnsafe());
        NDArray A = ValidUtil.prepareBroadcastOperand(this, targetShape, targetType);
        NDArray B = ValidUtil.prepareBroadcastOperand(b, targetShape, targetType);
        NDArray targetRes = ValidUtil.validateResultArray(resArray, targetType, targetShape);
        return switch(targetType) {
            case FLOAT -> ArithmeticOps.divFloat(A, B, targetRes);
            case DOUBLE -> ArithmeticOps.divDouble(A, B, targetRes);
            case INTEGER -> ArithmeticOps.divInt(A, B, targetRes);
        };
    }

    public NDArray div(float b){
        DType targetType = TypeUtil.promoteTypes(this.getDType(), TypeUtil.scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray resArray = NDArray.zeros(targetType, this.internalShapeUnsafe());
        return switch (targetType) {
            case FLOAT -> ArithmeticOps.divFloat(A, b, resArray);
            case DOUBLE -> ArithmeticOps.divDouble(A, b, resArray);
            case INTEGER -> ArithmeticOps.divInt(A, (int) b, resArray);
        };
    }

    public NDArray div(int b){
        DType targetType = TypeUtil.promoteTypes(this.getDType(), TypeUtil.scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray resArray = NDArray.zeros(targetType, this.internalShapeUnsafe());
        return switch (targetType) {
            case FLOAT -> ArithmeticOps.divFloat(A, b, resArray);
            case DOUBLE -> ArithmeticOps.divDouble(A, b, resArray);
            case INTEGER -> ArithmeticOps.divInt(A, b, resArray);
        };
    }

    public NDArray div(double b){
        DType targetType = TypeUtil.promoteTypes(this.getDType(), TypeUtil.scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray resArray = NDArray.zeros(targetType, this.internalShapeUnsafe());
        return switch (targetType) {
            case FLOAT -> ArithmeticOps.divFloat(A, (float) b, resArray);
            case DOUBLE -> ArithmeticOps.divDouble(A, b, resArray);
            case INTEGER -> ArithmeticOps.divInt(A, (int) b, resArray);
        };
    }

    public NDArray div(float b,NDArray resArray){
        DType targetType = TypeUtil.promoteTypes(this.getDType(), TypeUtil.scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray targetRes = ValidUtil.validateResultArray(resArray, targetType, this.internalShapeUnsafe());
        return switch (targetType) {
            case FLOAT -> ArithmeticOps.divFloat(A, b, targetRes);
            case DOUBLE -> ArithmeticOps.divDouble(A, b, targetRes);
            case INTEGER -> ArithmeticOps.divInt(A, (int) b, targetRes);
        };
    }

    public NDArray div(int b,NDArray resArray){
        DType targetType = TypeUtil.promoteTypes(this.getDType(), TypeUtil.scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray targetRes = ValidUtil.validateResultArray(resArray, targetType, this.internalShapeUnsafe());
        return switch (targetType) {
            case FLOAT -> ArithmeticOps.divFloat(A, b, targetRes);
            case DOUBLE -> ArithmeticOps.divDouble(A, b, targetRes);
            case INTEGER -> ArithmeticOps.divInt(A, b, targetRes);
        };
    }

    public NDArray div(double b,NDArray resArray){
        DType targetType = TypeUtil.promoteTypes(this.getDType(), TypeUtil.scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray targetRes = ValidUtil.validateResultArray(resArray, targetType, this.internalShapeUnsafe());
        return switch (targetType) {
            case FLOAT -> ArithmeticOps.divFloat(A, (float) b, targetRes);
            case DOUBLE -> ArithmeticOps.divDouble(A, b, targetRes);
            case INTEGER -> ArithmeticOps.divInt(A, (int) b, targetRes);
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
        return switch(this.getDType()){
            case FLOAT-> ExpOps.sqrtFloat(safeThis, NDArray.zeros(DType.FLOAT, this.internalShapeUnsafe()));
            case DOUBLE-> ExpOps.sqrtDouble(safeThis, NDArray.zeros(DType.DOUBLE, this.internalShapeUnsafe()));
            case INTEGER -> ExpOps.sqrtInt(safeThis, NDArray.zeros(DType.FLOAT, this.internalShapeUnsafe()));
        };
    }

    public NDArray abs(){
        NDArray safeThis = this.isContiguous() ? this : this.contiguous();
        return switch(this.getDType()){
            case FLOAT-> ExpOps.absFloat(safeThis, NDArray.zeros(DType.FLOAT, this.internalShapeUnsafe()));
            case DOUBLE-> ExpOps.absDouble(safeThis, NDArray.zeros(DType.DOUBLE, this.internalShapeUnsafe()));
            case INTEGER -> ExpOps.absInt(safeThis, NDArray.zeros(DType.INTEGER, this.internalShapeUnsafe()));
        };
    }

    public NDArray exp(){
        NDArray safeThis = this.isContiguous() ? this : this.contiguous();
        return switch(this.getDType()){
            case FLOAT-> ExpOps.expFloat(safeThis, NDArray.zeros(DType.FLOAT, this.internalShapeUnsafe()));
            case DOUBLE-> ExpOps.expDouble(safeThis, NDArray.zeros(DType.DOUBLE, this.internalShapeUnsafe()));
            case INTEGER -> ExpOps.expInt(safeThis, NDArray.zeros(DType.FLOAT, this.internalShapeUnsafe()));
        };
    }

    public NDArray log(){
        NDArray safeThis = this.isContiguous() ? this : this.contiguous();
        return switch(this.getDType()){
            case FLOAT-> ExpOps.logFloat(safeThis, NDArray.zeros(DType.FLOAT, this.internalShapeUnsafe()));
            case DOUBLE-> ExpOps.logDouble(safeThis, NDArray.zeros(DType.DOUBLE, this.internalShapeUnsafe()));
            case INTEGER -> ExpOps.logInt(safeThis, NDArray.zeros(DType.FLOAT, this.internalShapeUnsafe()));
        };
    }

    public NDArray log10(){
        NDArray safeThis = this.isContiguous() ? this : this.contiguous();
        return switch(this.getDType()){
            case FLOAT-> ExpOps.log10Float(safeThis, NDArray.zeros(DType.FLOAT, this.internalShapeUnsafe()));
            case DOUBLE-> ExpOps.log10Double(safeThis, NDArray.zeros(DType.DOUBLE, this.internalShapeUnsafe()));
            case INTEGER -> ExpOps.log10Int(safeThis, NDArray.zeros(DType.FLOAT, this.internalShapeUnsafe()));
        };
    }

    //TrigOps.java methods

    public NDArray sin(){
        NDArray safeThis = this.isContiguous() ? this : this.contiguous();
        
        return switch(this.getDType()){
            case FLOAT-> TrigOps.sinFloat(safeThis, NDArray.zeros(DType.FLOAT, this.internalShapeUnsafe()));
            case DOUBLE-> TrigOps.sinDouble(safeThis, NDArray.zeros(DType.DOUBLE, this.internalShapeUnsafe()));
            case INTEGER -> TrigOps.sinInt(safeThis, NDArray.zeros(DType.FLOAT, this.internalShapeUnsafe()));
        };
    }

    public NDArray cos(){
        NDArray safeThis = this.isContiguous() ? this : this.contiguous();
        return switch(this.getDType()){
            case FLOAT-> TrigOps.cosFloat(safeThis, NDArray.zeros(DType.FLOAT, this.internalShapeUnsafe()));
            case DOUBLE-> TrigOps.cosDouble(safeThis, NDArray.zeros(DType.DOUBLE, this.internalShapeUnsafe()));
            case INTEGER -> TrigOps.cosInt(safeThis, NDArray.zeros(DType.FLOAT, this.internalShapeUnsafe()));
        };
    }

    public NDArray tan(){
        NDArray safeThis = this.isContiguous() ? this : this.contiguous();
        
        return switch(this.getDType()){
            case FLOAT-> TrigOps.tanFloat(safeThis, NDArray.zeros(DType.FLOAT, this.internalShapeUnsafe()));
            case DOUBLE-> TrigOps.tanDouble(safeThis, NDArray.zeros(DType.DOUBLE, this.internalShapeUnsafe()));
            case INTEGER -> TrigOps.tanInt(safeThis, NDArray.zeros(DType.FLOAT, this.internalShapeUnsafe()));
        };
    }

    public NDArray cot(){
        NDArray safeThis = this.isContiguous() ? this : this.contiguous();
        
        return switch(this.getDType()){
            case FLOAT-> TrigOps.cotFloat(safeThis, NDArray.zeros(DType.FLOAT, this.internalShapeUnsafe()));
            case DOUBLE-> TrigOps.cotDouble(safeThis, NDArray.zeros(DType.DOUBLE, this.internalShapeUnsafe()));
            case INTEGER -> TrigOps.cotInt(safeThis, NDArray.zeros(DType.FLOAT, this.internalShapeUnsafe()));
        };
    }

    public NDArray sinh(){
        NDArray safeThis = this.isContiguous() ? this : this.contiguous();
        return switch(this.getDType()){
            case FLOAT-> TrigOps.sinhFloat(safeThis, NDArray.zeros(DType.FLOAT, this.internalShapeUnsafe()));
            case DOUBLE-> TrigOps.sinhDouble(safeThis, NDArray.zeros(DType.DOUBLE, this.internalShapeUnsafe()));
            case INTEGER -> TrigOps.sinhInt(safeThis, NDArray.zeros(DType.FLOAT, this.internalShapeUnsafe()));
        };
    }

    public NDArray cosh(){
        NDArray safeThis = this.isContiguous() ? this : this.contiguous();
        return switch(this.getDType()){
            case FLOAT-> TrigOps.coshFloat(safeThis, NDArray.zeros(DType.FLOAT, this.internalShapeUnsafe()));
            case DOUBLE-> TrigOps.coshDouble(safeThis, NDArray.zeros(DType.DOUBLE, this.internalShapeUnsafe()));
            case INTEGER -> TrigOps.coshInt(safeThis, NDArray.zeros(DType.FLOAT, this.internalShapeUnsafe()));
        };
    }

    public NDArray tanh(){
        NDArray safeThis = this.isContiguous() ? this : this.contiguous();
        
        return switch(this.getDType()){
            case FLOAT-> TrigOps.tanhFloat(safeThis, NDArray.zeros(DType.FLOAT, this.internalShapeUnsafe()));
            case DOUBLE-> TrigOps.tanhDouble(safeThis, NDArray.zeros(DType.DOUBLE, this.internalShapeUnsafe()));
            case INTEGER -> TrigOps.tanhInt(safeThis, NDArray.zeros(DType.FLOAT, this.internalShapeUnsafe()));
        };
    }

    public NDArray coth(){
        NDArray safeThis = this.isContiguous() ? this : this.contiguous();
        
        return switch(this.getDType()){
            case FLOAT-> TrigOps.cothFloat(safeThis, NDArray.zeros(DType.FLOAT, this.internalShapeUnsafe()));
            case DOUBLE-> TrigOps.cothDouble(safeThis, NDArray.zeros(DType.DOUBLE, this.internalShapeUnsafe()));
            case INTEGER -> TrigOps.cothInt(safeThis, NDArray.zeros(DType.FLOAT, this.internalShapeUnsafe()));
        };
    }

    //MatMulOps.java methods

    public NDArray matmul(NDArray b){
        ValidUtil.validateMatmulInputs(this, b);
        DType targetType = TypeUtil.promoteTypes(this.getDType(), b.getDType());
        NDArray A = this.cast(targetType);
        NDArray B = b.cast(targetType);
        int[] targetShape = new int[]{this.internalShapeUnsafe()[0], b.internalShapeUnsafe()[1]};
        NDArray resArray = NDArray.zeros(targetType, targetShape);
        return switch(targetType){
            case FLOAT -> MatMulOps.matmulFloat(A, B, resArray);
            case DOUBLE -> MatMulOps.matmulDouble(A, B, resArray);
            case INTEGER -> MatMulOps.matmulInt(A, B, resArray);
        };
    }

    public NDArray matmul(NDArray b, NDArray resArray){
        ValidUtil.validateMatmulInputs(this, b);
        DType targetType = TypeUtil.promoteTypes(this.getDType(), b.getDType());
        NDArray A = this.cast(targetType);
        NDArray B = b.cast(targetType);
        int[] targetShape = new int[]{this.internalShapeUnsafe()[0], b.internalShapeUnsafe()[1]};
        NDArray targetRes = ValidUtil.validateResultArray(resArray, targetType, targetShape);
        ValidUtil.validateOutputBuffer(targetRes);
        return switch(targetType){
            case FLOAT -> MatMulOps.matmulFloat(A, B, targetRes);
            case DOUBLE -> MatMulOps.matmulDouble(A, B, targetRes);
            case INTEGER -> MatMulOps.matmulInt(A, B, targetRes);
        };
    }


}
