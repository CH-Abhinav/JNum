package jnum;


import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Arrays;
import java.util.NoSuchElementException;
import java.util.concurrent.ThreadLocalRandom;

import jnum.jnumops.ArithmeticOps;
import jnum.jnumops.BooleanOps;
import jnum.jnumops.CompareOps;
import jnum.jnumops.ExpOps;
import jnum.jnumops.MatMulOps;
import jnum.jnumops.NDIter;
import jnum.jnumops.ReduceOps;
import jnum.jnumops.TrigOps;
import jnum.jnumutils.ShapeUtil;
import jnum.jnumutils.TypeUtil;
import jnum.jnumutils.ValidUtil;
import static jnum.DType.*;

public class NDArray{
    private final MemorySegment data;
    private final long[] shape;
    private final long[] strides;
    private final long size;
    private final DType dtype;

    private NDArray(){
        throw new AssertionError("NDArray can't be instantiated via constructor");
    }

    private NDArray(MemorySegment data,long[] shape,long[] strides,DType dType){
        this.data=data;
        this.shape=shape;
        this.strides=strides;
        long CalcSize=1;
        for (long dim : shape) CalcSize *= dim;
        this.size = CalcSize;
        this.dtype=dType;
    }

    public static NDArray ofRaw(MemorySegment data, long[] shape, long[] strides, DType dtype){
        return new NDArray(data, shape, strides, dtype);
    }

    public NDArray reshape(long... newShape) {
        long newCalcSize = 1;
        for (long dim : newShape) newCalcSize *= dim;
        if (newCalcSize != this.getSize()) {
            throw new IllegalArgumentException("Cannot reshape array of size " + this.getSize() + " into shape " + Arrays.toString(newShape));
        }
        NDArray safeThis = this.isContiguous() ? this : this.contiguous();
        return new NDArray(safeThis.getData(), newShape, ShapeUtil.calculateDefaultStrides(newShape), this.getDType());
    }

    public NDArray reshape(DType dType,long... newShape) {
        long newCalcSize = 1;
        for (long dim : newShape) newCalcSize *= dim;
        if (newCalcSize != this.getSize()) {
            throw new IllegalArgumentException("Cannot reshape array of size " + this.getSize() + " into shape " + Arrays.toString(newShape));
        }
        return this.reshape(newShape).cast(dType);
    }

    public NDArray transpose(){
        if(dim()<2) return this;
        var newShape=new long[this.internalShapeUnsafe().length];
        var newStrides=new long[this.internalStridesUnsafe().length];
        for(int i = 0; i< this.internalShapeUnsafe().length; i++){
            newShape[i]= this.internalShapeUnsafe()[(this.internalShapeUnsafe().length-1-i)];
            newStrides[i]= this.internalStridesUnsafe()[(this.internalStridesUnsafe().length-1-i)];
        }
        return new NDArray(this.getData(),newShape,newStrides, this.getDType());
    }

    public boolean isContiguous(){
        var expStride=1L;
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
        long byteSize=this.getSize()*this.getDType().layout.byteSize();
        var segment=arena.allocate(byteSize, 64);
        var newStrides=ShapeUtil.calculateDefaultStrides(this.internalShapeUnsafe());
        for(long i = 0; i< this.getSize(); i++){
            long tempindex=i;
            var coord=new long[this.internalShapeUnsafe().length];
            for(int j = this.internalShapeUnsafe().length-1; j>=0; j--){
                coord[j]=tempindex% this.internalShapeUnsafe()[j];
                tempindex=tempindex/ this.internalShapeUnsafe()[j];
            }
            long oldFlatIndex = 0;
            for(int d = 0; d < this.internalShapeUnsafe().length; d++){
                oldFlatIndex += coord[d] * this.internalStridesUnsafe()[d];
            }
            switch(getDType()){
                case f32 ->{
                    var val= this.getData().getAtIndex(ValueLayout.JAVA_FLOAT, oldFlatIndex);
                    segment.setAtIndex(ValueLayout.JAVA_FLOAT, i, val);
                }
                case i32 ->{
                    var val= this.getData().getAtIndex(ValueLayout.JAVA_INT, oldFlatIndex);
                    segment.setAtIndex(ValueLayout.JAVA_INT, i, val);
                }
                case f64 ->{
                    var val= this.getData().getAtIndex(ValueLayout.JAVA_DOUBLE, oldFlatIndex);
                    segment.setAtIndex(ValueLayout.JAVA_DOUBLE, i, val);
                }
                case bool ->{
                    var val= this.getData().getAtIndex(ValueLayout.JAVA_BYTE, oldFlatIndex);
                    segment.setAtIndex(ValueLayout.JAVA_BYTE, i, val);
                }
                default -> throw new UnsupportedOperationException("This dtype "+getDType()+" doesn't support this method");
            }
        }
        return new NDArray(segment, this.internalShapeUnsafe(), newStrides, getDType());
    }

    public NDArray broadcastTo(long ... shape){
        if(Arrays.equals(this.internalShapeUnsafe(),shape)) return this;
        int ndim=shape.length;
        if(ndim<this.dim()){
            throw new IllegalArgumentException(
                "Cannot broadcast array of shape " + Arrays.toString(this.internalShapeUnsafe()) +
                " to target shape " + Arrays.toString(shape) +
                " because the target has fewer dimensions."
            );
        }
        var newStrides=new long[ndim];
        long[] paddedShape=new long[ndim];
        var paddedStrides=new long[ndim];
        int offset=ndim-this.dim();
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
        NDArray res = JNum.zeros(target, safeThis.internalShapeUnsafe());
        for(long i = 0; i < safeThis.getSize(); i++){
            double val=switch (safeThis.getDType()){
                case f32 -> safeThis.getData().getAtIndex(ValueLayout.JAVA_FLOAT, i);
                case f64 -> safeThis.getData().getAtIndex(ValueLayout.JAVA_DOUBLE, i);
                case i32 -> safeThis.getData().getAtIndex(ValueLayout.JAVA_INT, i);
                case bool -> safeThis.getData().getAtIndex(ValueLayout.JAVA_BYTE, i) != 0 ? 1.0 : 0.0;
                default -> throw new UnsupportedOperationException("This dtype "+safeThis.getDType()+" doesn't support this method");
            };
            switch (target) {
                case f32 -> res.getData().setAtIndex(ValueLayout.JAVA_FLOAT, i, (float) val);
                case f64 -> res.getData().setAtIndex(ValueLayout.JAVA_DOUBLE, i, val);
                case i32 -> res.getData().setAtIndex(ValueLayout.JAVA_INT, i, (int) val);
                case bool -> res.getData().setAtIndex(ValueLayout.JAVA_BYTE, i, (byte) (val != 0.0 ? 1 : 0));
                default -> throw new UnsupportedOperationException("This dtype "+target+" doesn't support this method");
            }
        }
        return res;
    }


    public String shapeString() {
        return Arrays.toString(internalShapeUnsafe()).replace("[", "(").replace("]", ")");
    }

    public NDArray copy(){
        NDArray dups=JNum.zeros(this.getDType(), this.internalShapeUnsafe());
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
                case f32 -> dups.getData().setAtIndex(ValueLayout.JAVA_FLOAT, dstIndex, this.getData().get(ValueLayout.JAVA_FLOAT, byteOffset));
                case i32 -> dups.getData().setAtIndex(ValueLayout.JAVA_INT, dstIndex, this.getData().get(ValueLayout.JAVA_INT, byteOffset));
                case f64 -> dups.getData().setAtIndex(ValueLayout.JAVA_DOUBLE, dstIndex, this.getData().get(ValueLayout.JAVA_DOUBLE, byteOffset));
                case bool -> dups.getData().setAtIndex(ValueLayout.JAVA_BYTE, dstIndex, this.getData().get(ValueLayout.JAVA_BYTE, byteOffset));
                default -> throw new UnsupportedOperationException("This dtype "+this.getDType()+" doesn't support this method");
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
                case f32 -> this.getData().getAtIndex(ValueLayout.JAVA_FLOAT, offsetThis) == other.getData().getAtIndex(ValueLayout.JAVA_FLOAT, offsetOther);
                case f64 -> this.getData().getAtIndex(ValueLayout.JAVA_DOUBLE, offsetThis) == other.getData().getAtIndex(ValueLayout.JAVA_DOUBLE, offsetOther);
                case i32 -> this.getData().getAtIndex(ValueLayout.JAVA_INT, offsetThis) == other.getData().getAtIndex(ValueLayout.JAVA_INT, offsetOther);
                case bool -> this.getData().getAtIndex(ValueLayout.JAVA_BYTE, offsetThis) == other.getData().getAtIndex(ValueLayout.JAVA_BYTE, offsetOther);
                default -> throw new UnsupportedOperationException("This dtype "+this.getDType()+" doesn't support this method");
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
                case f32 -> Float.hashCode(getData().getAtIndex(ValueLayout.JAVA_FLOAT, physicalOffset));
                case f64 -> Double.hashCode(getData().getAtIndex(ValueLayout.JAVA_DOUBLE, physicalOffset));
                case i32 -> Integer.hashCode(getData().getAtIndex(ValueLayout.JAVA_INT, physicalOffset));
                case bool -> Byte.hashCode(getData().getAtIndex(ValueLayout.JAVA_BYTE, physicalOffset));
                default -> throw new UnsupportedOperationException("This dtype "+getDType()+" doesn't support this method");
            };
            result = 31 * result + valHash;
        }
        return result;
    }

    private static long getPhysicalOffset(long logicalIndex, long[] shape, long[] strides) {
        long remaining = logicalIndex;
        long offset = 0;
        for (int i = shape.length - 1; i >= 0; i--) {
            long coord = (remaining % shape[i]);
            remaining /= shape[i];
            offset += coord * strides[i];
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
            case f32 -> getData().getAtIndex(ValueLayout.JAVA_FLOAT, physicalOffset);
            case i32 -> getData().getAtIndex(ValueLayout.JAVA_INT, physicalOffset);
            case f64 -> getData().getAtIndex(ValueLayout.JAVA_DOUBLE, physicalOffset);
            case bool -> getData().getAtIndex(ValueLayout.JAVA_BYTE, physicalOffset) != 0 ? 1.0 : 0.0;
            default -> throw new UnsupportedOperationException("This dtype "+this.getDType()+" doesn't support this method");
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

    public boolean getFlatBoolean(long index){
        validateFlatIndex(index);
        long physicalOffset = this.isContiguous() ? index : getPhysicalOffset(index, this.internalShapeUnsafe(), this.internalStridesUnsafe());
        return getData().getAtIndex(ValueLayout.JAVA_BYTE, physicalOffset) != 0;
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
            case f32 -> getData().getAtIndex(ValueLayout.JAVA_FLOAT, flatIndex);
            case i32 -> getData().getAtIndex(ValueLayout.JAVA_INT, flatIndex);
            case f64 -> getData().getAtIndex(ValueLayout.JAVA_DOUBLE, flatIndex);
            case bool -> getData().getAtIndex(ValueLayout.JAVA_BYTE, flatIndex) != 0 ? 1.0 : 0.0;
            default -> throw new UnsupportedOperationException("This dtype "+this.getDType()+" doesn't support this method");
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

    public boolean getBoolean(int... indices){
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
        return getData().getAtIndex(ValueLayout.JAVA_BYTE, flatIndex) != 0;
    }

    public long[] indexOf(double b){
        NDIter iter = new NDIter(this.internalShapeUnsafe());
        switch(getDType()){
            case f32 ->{
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
            case i32 ->{
                var c= (int)b;
                while(iter.hasNext){
                    long byteOffset = ShapeUtil.getByteOffset(iter.coords, this.internalStridesUnsafe(), this.getDType());
                    if(c== this.getData().get(ValueLayout.JAVA_INT, byteOffset)){
                        return iter.coords.clone();
                    }
                    iter.next();
                }
            }
            case f64 ->{
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
            case bool ->{
                byte c = (byte) (b != 0 ? 1 : 0);
                while(iter.hasNext){
                    long byteOffset = ShapeUtil.getByteOffset(iter.coords, this.internalStridesUnsafe(), this.getDType());
                    if(c == this.getData().get(ValueLayout.JAVA_BYTE, byteOffset)){
                        return iter.coords.clone();
                    }
                    iter.next();
                }
            }
            default -> throw new UnsupportedOperationException("This dtype "+getDType()+" doesn't support this method");
        }
        throw new NoSuchElementException();    
    }

    public long[] getShape() {
        return internalShapeUnsafe().clone();
    }

    public int dim() {
        return internalShapeUnsafe().length;
    }

    public DType getDType(){
        return dtype;
    }

    public long[] internalStridesUnsafe() {
        return strides;
    }

    public long[] internalShapeUnsafe() {
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
                    case i32 ->sb.append(getData().get(ValueLayout.JAVA_INT, byteOffset));
                    case f32 ->sb.append(getData().get(ValueLayout.JAVA_FLOAT, byteOffset));
                    case f64 ->sb.append(getData().get(ValueLayout.JAVA_DOUBLE, byteOffset));
                    case bool ->sb.append(getData().get(ValueLayout.JAVA_BYTE, byteOffset) != 0 ? "true" : "false");
                    default -> throw new UnsupportedOperationException("This dtype "+this.getDType()+" doesn't support this method");
                }
                if (i < getSize() - 1) sb.append(", ");
            }
            iter.next();
        }
        return sb.append("]").toString();
    }

    public double max() {
        return switch(this.getDType()) {
            case f32 -> ReduceOps.maxFloat(this);
            case f64 -> ReduceOps.maxDouble(this);
            case i32 -> ReduceOps.maxInt(this);
            default -> throw new UnsupportedOperationException("This dtype "+this.getDType()+" doesn't support this method");
        };
    }

    public double min() {
        return switch(this.getDType()) {
            case f32 -> ReduceOps.minFloat(this);
            case f64 -> ReduceOps.minDouble(this);
            case i32 -> ReduceOps.minInt(this);
            default -> throw new UnsupportedOperationException("This dtype "+this.getDType()+" doesn't support this method");
        };
    }

    public double sum() {
        return switch(this.getDType()) {
            case f32 -> ReduceOps.sumFloat(this);
            case f64 -> ReduceOps.sumDouble(this);
            case i32 -> ReduceOps.sumInt(this);
            default -> throw new UnsupportedOperationException("This dtype "+this.getDType()+" doesn't support this method");
        };
    }

    public NDArray sum(int axis){
        long[] reducedShape = ShapeUtil.calculateReductionShape(this.internalShapeUnsafe(), axis);
        NDArray resArray = JNum.zeros(this.getDType(), reducedShape);
        return switch(this.getDType()) {
            case f32 -> ReduceOps.sumFloatAxis(this,axis,resArray);
            case f64 -> ReduceOps.sumDoubleAxis(this,axis,resArray);
            case i32 -> ReduceOps.sumIntAxis(this,axis,resArray);
            default -> throw new UnsupportedOperationException("This dtype "+this.getDType()+" doesn't support this method");
        };
    }

    public NDArray max(int axis) {
        long[] reducedShape = ShapeUtil.calculateReductionShape(this.internalShapeUnsafe(), axis);
        NDArray resArray = JNum.zeros(this.getDType(), reducedShape);
        return switch(this.getDType()) {
            case f32 -> ReduceOps.maxFloatAxis(this, axis, resArray);
            case f64 -> ReduceOps.maxDoubleAxis(this, axis, resArray);
            case i32 -> ReduceOps.maxIntAxis(this, axis, resArray);
            default -> throw new UnsupportedOperationException("This dtype "+this.getDType()+" doesn't support this method");
        };
    }

    public NDArray min(int axis) {
        long[] reducedShape = ShapeUtil.calculateReductionShape(this.internalShapeUnsafe(), axis);
        NDArray resArray = JNum.zeros(this.getDType(), reducedShape);
        return switch(this.getDType()) {
            case f32 -> ReduceOps.minFloatAxis(this, axis, resArray);
            case f64 -> ReduceOps.minDoubleAxis(this, axis, resArray);
            case i32 -> ReduceOps.minIntAxis(this, axis, resArray);
            default -> throw new UnsupportedOperationException("This dtype "+this.getDType()+" doesn't support this method");
        };
    }

    public double dot(NDArray b){
        if (this.dim() != 1 || b.dim() != 1) {
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
            case f32 ->ReduceOps.dotFloat(A, B);
            case i32 ->ReduceOps.dotInt(A, B);
            case f64 ->ReduceOps.dotDouble(A, B);
            default -> throw new UnsupportedOperationException("This dtype "+targetType+" doesn't support this method");
        };
    }

    public double avg() {
        return this.sum() / (double) this.getSize();
    }

    public NDArray maximum(NDArray b){
        DType targetType = TypeUtil.promoteTypes(this.getDType(), b.getDType());
        long[] targetShape = ShapeUtil.calculateBroadcastShape(this.internalShapeUnsafe(), b.internalShapeUnsafe());
        NDArray A = ValidUtil.prepareBroadcastOperand(this, targetShape, targetType);
        NDArray B = ValidUtil.prepareBroadcastOperand(b, targetShape, targetType);
        NDArray resArray = JNum.zeros(targetType, targetShape);

        return switch(targetType){
            case f32 ->CompareOps.maximumFloat(A, B, resArray);
            case i32 ->CompareOps.maximumInt(A, B, resArray);
            case f64 ->CompareOps.maximumDouble(A, B, resArray);
            default -> throw new UnsupportedOperationException("This dtype "+targetType+" doesn't support this method");
        };
    }

    public NDArray maximum(float b) {
        DType targetType = TypeUtil.promoteTypes(this.getDType(), TypeUtil.scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray resArray = JNum.zeros(targetType, this.internalShapeUnsafe());
        
        return switch (targetType) {
            case f32 -> CompareOps.maximumFloat(A, b, resArray);
            case i32 ->throw new UnsupportedOperationException();
            case f64 ->CompareOps.maximumDouble(A, b, resArray);
            default -> throw new UnsupportedOperationException("This dtype "+targetType+" doesn't support this method");
        };
    }

    public NDArray maximum(int b) {
        DType targetType = TypeUtil.promoteTypes(this.getDType(), TypeUtil.scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray resArray = JNum.zeros(targetType, this.internalShapeUnsafe());
        
        return switch (targetType) {
            case f32 -> CompareOps.maximumFloat(A, b, resArray);
            case i32 ->CompareOps.maximumInt(A, b, resArray);
            case f64 ->CompareOps.maximumDouble(A, b, resArray);
            default -> throw new UnsupportedOperationException("This dtype "+targetType+" doesn't support this method");
        };
    }

    public NDArray maximum(double b) {
        DType targetType = TypeUtil.promoteTypes(this.getDType(), TypeUtil.scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray resArray = JNum.zeros(targetType, this.internalShapeUnsafe());
        
        return switch (targetType) {
            case f32 -> throw new UnsupportedOperationException();
            case i32 -> throw new UnsupportedOperationException();
            case f64 -> CompareOps.maximumDouble(A, b, resArray);
            default -> throw new UnsupportedOperationException("This dtype "+targetType+" doesn't support this method");
        };
    }

    public NDArray minimum(NDArray b){
        DType targetType = TypeUtil.promoteTypes(this.getDType(), b.getDType());
        long[] targetShape = ShapeUtil.calculateBroadcastShape(this.internalShapeUnsafe(), b.internalShapeUnsafe());
        NDArray A = ValidUtil.prepareBroadcastOperand(this, targetShape, targetType);
        NDArray B = ValidUtil.prepareBroadcastOperand(b, targetShape, targetType);
        NDArray resArray = JNum.zeros(targetType, targetShape);

        return switch(targetType){
            case f32 ->CompareOps.minimumFloat(A, B, resArray);
            case i32 ->CompareOps.minimumInt(A, B, resArray);
            case f64 ->CompareOps.minimumDouble(A, B, resArray);
            default -> throw new UnsupportedOperationException("This dtype "+targetType+" doesn't support this method");
        };
    }

    public NDArray minimum(float b) {
        DType targetType = TypeUtil.promoteTypes(this.getDType(), TypeUtil.scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray resArray = JNum.zeros(targetType, this.internalShapeUnsafe());
        
        return switch (targetType) {
            case f32 -> CompareOps.minimumFloat(A, b, resArray);
            case i32 ->throw new UnsupportedOperationException();
            case f64 ->CompareOps.minimumDouble(A, b, resArray);
            default -> throw new UnsupportedOperationException("This dtype "+targetType+" doesn't support this method");
        };
    }

    public NDArray minimum(int b) {
        DType targetType = TypeUtil.promoteTypes(this.getDType(), TypeUtil.scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray resArray = JNum.zeros(targetType, this.internalShapeUnsafe());
        
        return switch (targetType) {
            case f32 -> CompareOps.minimumFloat(A, b, resArray);
            case i32 ->CompareOps.minimumInt(A, b, resArray);
            case f64 ->CompareOps.minimumDouble(A, b, resArray);
            default -> throw new UnsupportedOperationException("This dtype "+targetType+" doesn't support this method");
        };
    }

    public NDArray minimum(double b) {
        DType targetType = TypeUtil.promoteTypes(this.getDType(), TypeUtil.scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray resArray = JNum.zeros(targetType, this.internalShapeUnsafe());
        
        return switch (targetType) {
            case f32 -> throw new UnsupportedOperationException();
            case i32 -> throw new UnsupportedOperationException();
            case f64 -> CompareOps.minimumDouble(A, b, resArray);
            default -> throw new UnsupportedOperationException("This dtype "+targetType+" doesn't support this method");
        };
    }

    //ArithmaticOps.java 

    //addition operation

    public NDArray add(NDArray b){
        DType targetType = TypeUtil.promoteTypes(this.getDType(), b.getDType());
        long[] targetShape = ShapeUtil.calculateBroadcastShape(this.internalShapeUnsafe(), b.internalShapeUnsafe());
        NDArray A = ValidUtil.prepareBroadcastOperand(this, targetShape, targetType);
        NDArray B = ValidUtil.prepareBroadcastOperand(b, targetShape, targetType);
        NDArray resArray = JNum.zeros(targetType, targetShape);
        return switch(targetType) {
            case f32 -> ArithmeticOps.addFloat(A, B, resArray);
            case f64 -> ArithmeticOps.addDouble(A, B, resArray);
            case i32 -> ArithmeticOps.addInt(A, B, resArray);
            default -> throw new UnsupportedOperationException("This dtype "+targetType+" doesn't support this method");
        };
    }
    
    public NDArray add(NDArray b,NDArray resArray){
        DType targetType = TypeUtil.promoteTypes(this.getDType(), b.getDType());
        long[] targetShape = ShapeUtil.calculateBroadcastShape(this.internalShapeUnsafe(), b.internalShapeUnsafe());
        NDArray A = ValidUtil.prepareBroadcastOperand(this, targetShape, targetType);
        NDArray B = ValidUtil.prepareBroadcastOperand(b, targetShape, targetType);
        NDArray targetRes = ValidUtil.validateResultArray(resArray, targetType, targetShape);
        return switch(targetType) {
            case f32 -> ArithmeticOps.addFloat(A, B, targetRes);
            case f64 -> ArithmeticOps.addDouble(A, B, targetRes);
            case i32 -> ArithmeticOps.addInt(A, B, targetRes);
            default -> throw new UnsupportedOperationException("This dtype "+targetType+" doesn't support this method");
        };
    }

    public NDArray add(float b){
        DType targetType = TypeUtil.promoteTypes(this.getDType(), TypeUtil.scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray resArray = JNum.zeros(targetType, this.internalShapeUnsafe());
        return switch (targetType) {
            case f32 -> ArithmeticOps.addFloat(A, b, resArray);
            case f64 -> ArithmeticOps.addDouble(A, b, resArray);
            case i32 -> ArithmeticOps.addInt(A, (int) b, resArray);
            default -> throw new UnsupportedOperationException("This dtype "+targetType+" doesn't support this method");
        };
    }

    public NDArray add(int b){
        DType targetType = TypeUtil.promoteTypes(this.getDType(), TypeUtil.scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray resArray = JNum.zeros(targetType, this.internalShapeUnsafe());
        return switch (targetType) {
            case f32 -> ArithmeticOps.addFloat(A, b, resArray);
            case f64 -> ArithmeticOps.addDouble(A, b, resArray);
            case i32 -> ArithmeticOps.addInt(A, b, resArray);
            default -> throw new UnsupportedOperationException("This dtype "+targetType+" doesn't support this method");
        };
    }

    public NDArray add(double b){
        DType targetType = TypeUtil.promoteTypes(this.getDType(), TypeUtil.scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray resArray = JNum.zeros(targetType, this.internalShapeUnsafe());
        return switch (targetType) {
            case f32 -> ArithmeticOps.addFloat(A, (float) b, resArray);
            case f64 -> ArithmeticOps.addDouble(A, b, resArray);
            case i32 -> ArithmeticOps.addInt(A, (int) b, resArray);
            default -> throw new UnsupportedOperationException("This dtype "+targetType+" doesn't support this method");
        };
    }

    public NDArray add(float b,NDArray resArray){
        DType targetType = TypeUtil.promoteTypes(this.getDType(), TypeUtil.scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray targetRes = ValidUtil.validateResultArray(resArray, targetType, this.internalShapeUnsafe());
        return switch (targetType) {
            case f32 -> ArithmeticOps.addFloat(A, b, targetRes);
            case f64 -> ArithmeticOps.addDouble(A, b, targetRes);
            case i32 -> ArithmeticOps.addInt(A, (int) b, targetRes);
            default -> throw new UnsupportedOperationException("This dtype "+targetType+" doesn't support this method");
        };
    }

    public NDArray add(int b,NDArray resArray){
        DType targetType = TypeUtil.promoteTypes(this.getDType(), TypeUtil.scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray targetRes = ValidUtil.validateResultArray(resArray, targetType, this.internalShapeUnsafe());
        return switch (targetType) {
            case f32 -> ArithmeticOps.addFloat(A, b, targetRes);
            case f64 -> ArithmeticOps.addDouble(A, b, targetRes);
            case i32 -> ArithmeticOps.addInt(A, b, targetRes);
            default -> throw new UnsupportedOperationException("This dtype "+targetType+" doesn't support this method");
        };
    }

    public NDArray add(double b,NDArray resArray){
        DType targetType = TypeUtil.promoteTypes(this.getDType(), TypeUtil.scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray targetRes = ValidUtil.validateResultArray(resArray, targetType, this.internalShapeUnsafe());
        return switch (targetType) {
            case f32 -> ArithmeticOps.addFloat(A, (float) b, targetRes);
            case f64 -> ArithmeticOps.addDouble(A, b, targetRes);
            case i32 -> ArithmeticOps.addInt(A, (int) b, targetRes);
            default -> throw new UnsupportedOperationException("This dtype "+targetType+" doesn't support this method");
        };
    }

    //subtract operations

    public NDArray sub(NDArray b){
        DType targetType = TypeUtil.promoteTypes(this.getDType(), b.getDType());
        long[] targetShape = ShapeUtil.calculateBroadcastShape(this.internalShapeUnsafe(), b.internalShapeUnsafe());
        NDArray A = ValidUtil.prepareBroadcastOperand(this, targetShape, targetType);
        NDArray B = ValidUtil.prepareBroadcastOperand(b, targetShape, targetType);
        NDArray resArray = JNum.zeros(targetType, targetShape);
        return switch(targetType) {
            case f32 -> ArithmeticOps.subFloat(A, B, resArray);
            case f64 -> ArithmeticOps.subDouble(A, B, resArray);
            case i32 -> ArithmeticOps.subInt(A, B, resArray);
            default -> throw new UnsupportedOperationException("This dtype "+targetType+" doesn't support this method");
        };
    }

    public NDArray sub(NDArray b,NDArray resArray){
        DType targetType = TypeUtil.promoteTypes(this.getDType(), b.getDType());
        long[] targetShape = ShapeUtil.calculateBroadcastShape(this.internalShapeUnsafe(), b.internalShapeUnsafe());
        NDArray A = ValidUtil.prepareBroadcastOperand(this, targetShape, targetType);
        NDArray B = ValidUtil.prepareBroadcastOperand(b, targetShape, targetType);
        NDArray targetRes = ValidUtil.validateResultArray(resArray, targetType, targetShape);
        return switch(targetType) {
            case f32 -> ArithmeticOps.subFloat(A, B, targetRes);
            case f64 -> ArithmeticOps.subDouble(A, B, targetRes);
            case i32 -> ArithmeticOps.subInt(A, B, targetRes);
            default -> throw new UnsupportedOperationException("This dtype "+targetType+" doesn't support this method");
        };
    }

    public NDArray sub(float b){
        DType targetType = TypeUtil.promoteTypes(this.getDType(), TypeUtil.scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray resArray = JNum.zeros(targetType, this.internalShapeUnsafe());
        return switch (targetType) {
            case f32 -> ArithmeticOps.subFloat(A, b, resArray);
            case f64 -> ArithmeticOps.subDouble(A, b, resArray);
            case i32 -> ArithmeticOps.subInt(A, (int) b, resArray);
            default -> throw new UnsupportedOperationException("This dtype "+targetType+" doesn't support this method");
        };
    }

    public NDArray sub(int b){
        DType targetType = TypeUtil.promoteTypes(this.getDType(), TypeUtil.scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray resArray = JNum.zeros(targetType, this.internalShapeUnsafe());
        return switch (targetType) {
            case f32 -> ArithmeticOps.subFloat(A, b, resArray);
            case f64 -> ArithmeticOps.subDouble(A, b, resArray);
            case i32 -> ArithmeticOps.subInt(A, b, resArray);
            default -> throw new UnsupportedOperationException("This dtype "+targetType+" doesn't support this method");
        };
    }

    public NDArray sub(double b){
        DType targetType = TypeUtil.promoteTypes(this.getDType(), TypeUtil.scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray resArray = JNum.zeros(targetType, this.internalShapeUnsafe());
        return switch (targetType) {
            case f32 -> ArithmeticOps.subFloat(A, (float) b, resArray);
            case f64 -> ArithmeticOps.subDouble(A, b, resArray);
            case i32 -> ArithmeticOps.subInt(A, (int) b, resArray);
            default -> throw new UnsupportedOperationException("This dtype "+targetType+" doesn't support this method");
        };
    }

    public NDArray sub(float b,NDArray resArray){
        DType targetType = TypeUtil.promoteTypes(this.getDType(), TypeUtil.scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray targetRes = ValidUtil.validateResultArray(resArray, targetType, this.internalShapeUnsafe());
        return switch (targetType) {
            case f32 -> ArithmeticOps.subFloat(A, b, targetRes);
            case f64 -> ArithmeticOps.subDouble(A, b, targetRes);
            case i32 -> ArithmeticOps.subInt(A, (int) b, targetRes);
            default -> throw new UnsupportedOperationException("This dtype "+targetType+" doesn't support this method");
        };
    }

    public NDArray sub(int b,NDArray resArray){
        DType targetType = TypeUtil.promoteTypes(this.getDType(), TypeUtil.scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray targetRes = ValidUtil.validateResultArray(resArray, targetType, this.internalShapeUnsafe());
        return switch (targetType) {
            case f32 -> ArithmeticOps.subFloat(A, b, targetRes);
            case f64 -> ArithmeticOps.subDouble(A, b, targetRes);
            case i32 -> ArithmeticOps.subInt(A, b, targetRes);
            default -> throw new UnsupportedOperationException("This dtype "+targetType+" doesn't support this method");
        };
    }

    public NDArray sub(double b,NDArray resArray){
        DType targetType = TypeUtil.promoteTypes(this.getDType(), TypeUtil.scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray targetRes = ValidUtil.validateResultArray(resArray, targetType, this.internalShapeUnsafe());
        return switch (targetType) {
            case f32 -> ArithmeticOps.subFloat(A, (float) b, targetRes);
            case f64 -> ArithmeticOps.subDouble(A, b, targetRes);
            case i32 -> ArithmeticOps.subInt(A, (int) b, targetRes);
            default -> throw new UnsupportedOperationException("This dtype "+targetType+" doesn't support this method");
        };
    }

    //multiplication operations 

    public NDArray mul(NDArray b){
        DType targetType = TypeUtil.promoteTypes(this.getDType(), b.getDType());
        long[] targetShape = ShapeUtil.calculateBroadcastShape(this.internalShapeUnsafe(), b.internalShapeUnsafe());
        NDArray A = ValidUtil.prepareBroadcastOperand(this, targetShape, targetType);
        NDArray B = ValidUtil.prepareBroadcastOperand(b, targetShape, targetType);
        NDArray resArray = JNum.zeros(targetType, targetShape);
        return switch(targetType) {
            case f32 -> ArithmeticOps.mulFloat(A, B, resArray);
            case f64 -> ArithmeticOps.mulDouble(A, B, resArray);
            case i32 -> ArithmeticOps.mulInt(A, B, resArray);
            default -> throw new UnsupportedOperationException("This dtype "+targetType+" doesn't support this method");
        };
    }

    public NDArray mul(NDArray b, NDArray resArray){
        DType targetType = TypeUtil.promoteTypes(this.getDType(), b.getDType());
        long[] targetShape = ShapeUtil.calculateBroadcastShape(this.internalShapeUnsafe(), b.internalShapeUnsafe());
        NDArray A = ValidUtil.prepareBroadcastOperand(this, targetShape, targetType);
        NDArray B = ValidUtil.prepareBroadcastOperand(b, targetShape, targetType);
        NDArray targetRes = ValidUtil.validateResultArray(resArray, targetType, targetShape);
        return switch(targetType) {
            case f32 -> ArithmeticOps.mulFloat(A, B, targetRes);
            case f64 -> ArithmeticOps.mulDouble(A, B, targetRes);
            case i32 -> ArithmeticOps.mulInt(A, B, targetRes);
            default -> throw new UnsupportedOperationException("This dtype "+targetType+" doesn't support this method");
        };
    }

    public NDArray mul(float b){
        DType targetType = TypeUtil.promoteTypes(this.getDType(), TypeUtil.scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray resArray = JNum.zeros(targetType, this.internalShapeUnsafe());
        return switch (targetType) {
            case f32 -> ArithmeticOps.mulFloat(A, b, resArray);
            case f64 -> ArithmeticOps.mulDouble(A, b, resArray);
            case i32 -> ArithmeticOps.mulInt(A, (int) b, resArray);
            default -> throw new UnsupportedOperationException("This dtype "+targetType+" doesn't support this method");
        };
    }

    public NDArray mul(int b){
        DType targetType = TypeUtil.promoteTypes(this.getDType(), TypeUtil.scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray resArray = JNum.zeros(targetType, this.internalShapeUnsafe());
        return switch (targetType) {
            case f32 -> ArithmeticOps.mulFloat(A, b, resArray);
            case f64 -> ArithmeticOps.mulDouble(A, b, resArray);
            case i32 -> ArithmeticOps.mulInt(A, b, resArray);
            default -> throw new UnsupportedOperationException("This dtype "+targetType+" doesn't support this method");
        };
    }

    public NDArray mul(double b){
        DType targetType = TypeUtil.promoteTypes(this.getDType(), TypeUtil.scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray resArray = JNum.zeros(targetType, this.internalShapeUnsafe());
        return switch (targetType) {
            case f32 -> ArithmeticOps.mulFloat(A, (float) b, resArray);
            case f64 -> ArithmeticOps.mulDouble(A, b, resArray);
            case i32 -> ArithmeticOps.mulInt(A, (int) b, resArray);
            default -> throw new UnsupportedOperationException("This dtype "+targetType+" doesn't support this method");
        };
    }

    public NDArray mul(float b,NDArray resArray){
        DType targetType = TypeUtil.promoteTypes(this.getDType(), TypeUtil.scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray targetRes = ValidUtil.validateResultArray(resArray, targetType, this.internalShapeUnsafe());
        return switch (targetType) {
            case f32 -> ArithmeticOps.mulFloat(A, b, targetRes);
            case f64 -> ArithmeticOps.mulDouble(A, b, targetRes);
            case i32 -> ArithmeticOps.mulInt(A, (int) b, targetRes);
            default -> throw new UnsupportedOperationException("This dtype "+targetType+" doesn't support this method");
        };
    }

    public NDArray mul(int b,NDArray resArray){
        DType targetType = TypeUtil.promoteTypes(this.getDType(), TypeUtil.scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray targetRes = ValidUtil.validateResultArray(resArray, targetType, this.internalShapeUnsafe());
        return switch (targetType) {
            case f32 -> ArithmeticOps.mulFloat(A, b, targetRes);
            case f64 -> ArithmeticOps.mulDouble(A, b, targetRes);
            case i32 -> ArithmeticOps.mulInt(A, b, targetRes);
            default -> throw new UnsupportedOperationException("This dtype "+targetType+" doesn't support this method");
        };
    }

    public NDArray mul(double b,NDArray resArray){
        DType targetType = TypeUtil.promoteTypes(this.getDType(), TypeUtil.scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray targetRes = ValidUtil.validateResultArray(resArray, targetType, this.internalShapeUnsafe());
        return switch (targetType) {
            case f32 -> ArithmeticOps.mulFloat(A, (float) b, targetRes);
            case f64 -> ArithmeticOps.mulDouble(A, b, targetRes);
            case i32 -> ArithmeticOps.mulInt(A, (int) b, targetRes);
            default -> throw new UnsupportedOperationException("This dtype "+targetType+" doesn't support this method");
        };
    }

    //division operations

    public NDArray div(NDArray b){
        DType targetType = TypeUtil.promoteTypes(this.getDType(), b.getDType());
        long[] targetShape = ShapeUtil.calculateBroadcastShape(this.internalShapeUnsafe(), b.internalShapeUnsafe());
        NDArray A = ValidUtil.prepareBroadcastOperand(this, targetShape, targetType);
        NDArray B = ValidUtil.prepareBroadcastOperand(b, targetShape, targetType);
        NDArray resArray = JNum.zeros(targetType, targetShape);
        return switch(targetType) {
            case f32 -> ArithmeticOps.divFloat(A, B, resArray);
            case f64 -> ArithmeticOps.divDouble(A, B, resArray);
            case i32 -> ArithmeticOps.divInt(A, B, resArray);
            default -> throw new UnsupportedOperationException("This dtype "+targetType+" doesn't support this method");
        };
    }

    public NDArray div(NDArray b, NDArray resArray){
        DType targetType = TypeUtil.promoteTypes(this.getDType(), b.getDType());
        long[] targetShape = ShapeUtil.calculateBroadcastShape(this.internalShapeUnsafe(), b.internalShapeUnsafe());
        NDArray A = ValidUtil.prepareBroadcastOperand(this, targetShape, targetType);
        NDArray B = ValidUtil.prepareBroadcastOperand(b, targetShape, targetType);
        NDArray targetRes = ValidUtil.validateResultArray(resArray, targetType, targetShape);
        return switch(targetType) {
            case f32 -> ArithmeticOps.divFloat(A, B, targetRes);
            case f64 -> ArithmeticOps.divDouble(A, B, targetRes);
            case i32 -> ArithmeticOps.divInt(A, B, targetRes);
            default -> throw new UnsupportedOperationException("This dtype "+targetType+" doesn't support this method");
        };
    }

    public NDArray div(float b){
        DType targetType = TypeUtil.promoteTypes(this.getDType(), TypeUtil.scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray resArray = JNum.zeros(targetType, this.internalShapeUnsafe());
        return switch (targetType) {
            case f32 -> ArithmeticOps.divFloat(A, b, resArray);
            case f64 -> ArithmeticOps.divDouble(A, b, resArray);
            case i32 -> ArithmeticOps.divInt(A, (int) b, resArray);
            default -> throw new UnsupportedOperationException("This dtype "+targetType+" doesn't support this method");
        };
    }

    public NDArray div(int b){
        DType targetType = TypeUtil.promoteTypes(this.getDType(), TypeUtil.scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray resArray = JNum.zeros(targetType, this.internalShapeUnsafe());
        return switch (targetType) {
            case f32 -> ArithmeticOps.divFloat(A, b, resArray);
            case f64 -> ArithmeticOps.divDouble(A, b, resArray);
            case i32 -> ArithmeticOps.divInt(A, b, resArray);
            default -> throw new UnsupportedOperationException("This dtype "+targetType+" doesn't support this method");
        };
    }

    public NDArray div(double b){
        DType targetType = TypeUtil.promoteTypes(this.getDType(), TypeUtil.scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray resArray = JNum.zeros(targetType, this.internalShapeUnsafe());
        return switch (targetType) {
            case f32 -> ArithmeticOps.divFloat(A, (float) b, resArray);
            case f64 -> ArithmeticOps.divDouble(A, b, resArray);
            case i32 -> ArithmeticOps.divInt(A, (int) b, resArray);
            default -> throw new UnsupportedOperationException("This dtype "+targetType+" doesn't support this method");
        };
    }

    public NDArray div(float b,NDArray resArray){
        DType targetType = TypeUtil.promoteTypes(this.getDType(), TypeUtil.scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray targetRes = ValidUtil.validateResultArray(resArray, targetType, this.internalShapeUnsafe());
        return switch (targetType) {
            case f32 -> ArithmeticOps.divFloat(A, b, targetRes);
            case f64 -> ArithmeticOps.divDouble(A, b, targetRes);
            case i32 -> ArithmeticOps.divInt(A, (int) b, targetRes);
            default -> throw new UnsupportedOperationException("This dtype "+targetType+" doesn't support this method");
        };
    }

    public NDArray div(int b,NDArray resArray){
        DType targetType = TypeUtil.promoteTypes(this.getDType(), TypeUtil.scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray targetRes = ValidUtil.validateResultArray(resArray, targetType, this.internalShapeUnsafe());
        return switch (targetType) {
            case f32 -> ArithmeticOps.divFloat(A, b, targetRes);
            case f64 -> ArithmeticOps.divDouble(A, b, targetRes);
            case i32 -> ArithmeticOps.divInt(A, b, targetRes);
            default -> throw new UnsupportedOperationException("This dtype "+targetType+" doesn't support this method");
        };
    }

    public NDArray div(double b,NDArray resArray){
        DType targetType = TypeUtil.promoteTypes(this.getDType(), TypeUtil.scalarType(b));
        NDArray A = this.cast(targetType);
        NDArray targetRes = ValidUtil.validateResultArray(resArray, targetType, this.internalShapeUnsafe());
        return switch (targetType) {
            case f32 -> ArithmeticOps.divFloat(A, (float) b, targetRes);
            case f64 -> ArithmeticOps.divDouble(A, b, targetRes);
            case i32 -> ArithmeticOps.divInt(A, (int) b, targetRes);
            default -> throw new UnsupportedOperationException("This dtype "+targetType+" doesn't support this method");
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
            case f32 -> ExpOps.sqrtFloat(safeThis, JNum.zeros(DType.f32, this.internalShapeUnsafe()));
            case f64 -> ExpOps.sqrtDouble(safeThis, JNum.zeros(DType.f64, this.internalShapeUnsafe()));
            case i32 -> ExpOps.sqrtInt(safeThis, JNum.zeros(DType.f32, this.internalShapeUnsafe()));
            default -> throw new UnsupportedOperationException("This dtype "+safeThis.getDType()+" doesn't support this method");
        };
    }

    public NDArray abs(){
        NDArray safeThis = this.isContiguous() ? this : this.contiguous();
        return switch(this.getDType()){
            case f32 -> ExpOps.absFloat(safeThis, JNum.zeros(DType.f32, this.internalShapeUnsafe()));
            case f64 -> ExpOps.absDouble(safeThis, JNum.zeros(DType.f64, this.internalShapeUnsafe()));
            case i32 -> ExpOps.absInt(safeThis, JNum.zeros(DType.i32, this.internalShapeUnsafe()));
            default -> throw new UnsupportedOperationException("This dtype "+safeThis.getDType()+" doesn't support this method");
        };
    }

    public NDArray exp(){
        NDArray safeThis = this.isContiguous() ? this : this.contiguous();
        return switch(this.getDType()){
            case f32 -> ExpOps.expFloat(safeThis, JNum.zeros(DType.f32, this.internalShapeUnsafe()));
            case f64 -> ExpOps.expDouble(safeThis, JNum.zeros(DType.f64, this.internalShapeUnsafe()));
            case i32 -> ExpOps.expInt(safeThis, JNum.zeros(DType.f32, this.internalShapeUnsafe()));
            default -> throw new UnsupportedOperationException("This dtype "+safeThis.getDType()+" doesn't support this method");
        };
    }

    public NDArray log(){
        NDArray safeThis = this.isContiguous() ? this : this.contiguous();
        return switch(this.getDType()){
            case f32 -> ExpOps.logFloat(safeThis, JNum.zeros(DType.f32, this.internalShapeUnsafe()));
            case f64 -> ExpOps.logDouble(safeThis, JNum.zeros(DType.f64, this.internalShapeUnsafe()));
            case i32 -> ExpOps.logInt(safeThis, JNum.zeros(DType.f32, this.internalShapeUnsafe()));
            default -> throw new UnsupportedOperationException("This dtype "+safeThis.getDType()+" doesn't support this method");
        };
    }

    public NDArray log10(){
        NDArray safeThis = this.isContiguous() ? this : this.contiguous();
        return switch(this.getDType()){
            case f32 -> ExpOps.log10Float(safeThis, JNum.zeros(DType.f32, this.internalShapeUnsafe()));
            case f64 -> ExpOps.log10Double(safeThis, JNum.zeros(DType.f64, this.internalShapeUnsafe()));
            case i32 -> ExpOps.log10Int(safeThis, JNum.zeros(DType.f32, this.internalShapeUnsafe()));
            default -> throw new UnsupportedOperationException("This dtype "+safeThis.getDType()+" doesn't support this method");
        };
    }

    public NDArray sigmoid() {
        NDArray safeThis = this.isContiguous() ? this : this.contiguous();
        return switch(this.getDType()) {
            case f32 -> ExpOps.sigmoidFloat(safeThis, JNum.zeros(DType.f32, this.internalShapeUnsafe()));
            case f64 -> ExpOps.sigmoidDouble(safeThis, JNum.zeros(DType.f64, this.internalShapeUnsafe()));
            case i32 -> ExpOps.sigmoidInt(safeThis, JNum.zeros(DType.f32, this.internalShapeUnsafe()));
            case bool -> ExpOps.sigmoidFloat(safeThis.cast(DType.f32), JNum.zeros(DType.f32, this.internalShapeUnsafe()));
            default -> throw new UnsupportedOperationException("This dtype "+safeThis.getDType()+" doesn't support this method");
        };
    }

    //TrigOps.java methods

    public NDArray sin(){
        NDArray safeThis = this.isContiguous() ? this : this.contiguous();
        
        return switch(this.getDType()){
            case f32 -> TrigOps.sinFloat(safeThis, JNum.zeros(DType.f32, this.internalShapeUnsafe()));
            case f64 -> TrigOps.sinDouble(safeThis, JNum.zeros(DType.f64, this.internalShapeUnsafe()));
            case i32 -> TrigOps.sinInt(safeThis, JNum.zeros(DType.f32, this.internalShapeUnsafe()));
            default -> throw new UnsupportedOperationException("This dtype "+safeThis.getDType()+" doesn't support this method");
        };
    }

    public NDArray cos(){
        NDArray safeThis = this.isContiguous() ? this : this.contiguous();
        return switch(this.getDType()){
            case f32 -> TrigOps.cosFloat(safeThis, JNum.zeros(DType.f32, this.internalShapeUnsafe()));
            case f64 -> TrigOps.cosDouble(safeThis, JNum.zeros(DType.f64, this.internalShapeUnsafe()));
            case i32 -> TrigOps.cosInt(safeThis, JNum.zeros(DType.f32, this.internalShapeUnsafe()));
            default -> throw new UnsupportedOperationException("This dtype "+safeThis.getDType()+" doesn't support this method");
        };
    }

    public NDArray tan(){
        NDArray safeThis = this.isContiguous() ? this : this.contiguous();
        
        return switch(this.getDType()){
            case f32 -> TrigOps.tanFloat(safeThis, JNum.zeros(DType.f32, this.internalShapeUnsafe()));
            case f64 -> TrigOps.tanDouble(safeThis, JNum.zeros(DType.f64, this.internalShapeUnsafe()));
            case i32 -> TrigOps.tanInt(safeThis, JNum.zeros(DType.f32, this.internalShapeUnsafe()));
            default -> throw new UnsupportedOperationException("This dtype "+safeThis.getDType()+" doesn't support this method");
        };
    }

    public NDArray cot(){
        NDArray safeThis = this.isContiguous() ? this : this.contiguous();
        
        return switch(this.getDType()){
            case f32 -> TrigOps.cotFloat(safeThis, JNum.zeros(DType.f32, this.internalShapeUnsafe()));
            case f64 -> TrigOps.cotDouble(safeThis, JNum.zeros(DType.f64, this.internalShapeUnsafe()));
            case i32 -> TrigOps.cotInt(safeThis, JNum.zeros(DType.f32, this.internalShapeUnsafe()));
            default -> throw new UnsupportedOperationException("This dtype "+safeThis.getDType()+" doesn't support this method");
        };
    }

    public NDArray sinh(){
        NDArray safeThis = this.isContiguous() ? this : this.contiguous();
        return switch(this.getDType()){
            case f32 -> TrigOps.sinhFloat(safeThis, JNum.zeros(DType.f32, this.internalShapeUnsafe()));
            case f64 -> TrigOps.sinhDouble(safeThis, JNum.zeros(DType.f64, this.internalShapeUnsafe()));
            case i32 -> TrigOps.sinhInt(safeThis, JNum.zeros(DType.f32, this.internalShapeUnsafe()));
            default -> throw new UnsupportedOperationException("This dtype "+safeThis.getDType()+" doesn't support this method");
        };
    }

    public NDArray cosh(){
        NDArray safeThis = this.isContiguous() ? this : this.contiguous();
        return switch(this.getDType()){
            case f32 -> TrigOps.coshFloat(safeThis, JNum.zeros(DType.f32, this.internalShapeUnsafe()));
            case f64 -> TrigOps.coshDouble(safeThis, JNum.zeros(DType.f64, this.internalShapeUnsafe()));
            case i32 -> TrigOps.coshInt(safeThis, JNum.zeros(DType.f32, this.internalShapeUnsafe()));
            default -> throw new UnsupportedOperationException("This dtype "+safeThis.getDType()+" doesn't support this method");
        };
    }

    public NDArray tanh(){
        NDArray safeThis = this.isContiguous() ? this : this.contiguous();
        
        return switch(this.getDType()){
            case f32 -> TrigOps.tanhFloat(safeThis, JNum.zeros(DType.f32, this.internalShapeUnsafe()));
            case f64 -> TrigOps.tanhDouble(safeThis, JNum.zeros(DType.f64, this.internalShapeUnsafe()));
            case i32 -> TrigOps.tanhInt(safeThis, JNum.zeros(DType.f32, this.internalShapeUnsafe()));
            default -> throw new UnsupportedOperationException("This dtype "+safeThis.getDType()+" doesn't support this method");
        };
    }

    public NDArray coth(){
        NDArray safeThis = this.isContiguous() ? this : this.contiguous();
        
        return switch(this.getDType()){
            case f32 -> TrigOps.cothFloat(safeThis, JNum.zeros(DType.f32, this.internalShapeUnsafe()));
            case f64 -> TrigOps.cothDouble(safeThis, JNum.zeros(DType.f64, this.internalShapeUnsafe()));
            case i32 -> TrigOps.cothInt(safeThis, JNum.zeros(DType.f32, this.internalShapeUnsafe()));
            default -> throw new UnsupportedOperationException("This dtype "+safeThis.getDType()+" doesn't support this method");
        };
    }

    //MatMulOps.java methods

    public NDArray matmul(NDArray b){
        ValidUtil.validateMatmulInputs(this, b);
        DType targetType = TypeUtil.promoteTypes(this.getDType(), b.getDType());
        NDArray A = this.cast(targetType);
        NDArray B = b.cast(targetType);
        long[] targetShape = new long[]{this.internalShapeUnsafe()[0], b.internalShapeUnsafe()[1]};
        NDArray resArray = JNum.zeros(targetType, targetShape);
        return switch(targetType){
            case f32 -> MatMulOps.matmulFloat(A, B, resArray);
            case f64 -> MatMulOps.matmulDouble(A, B, resArray);
            case i32 -> MatMulOps.matmulInt(A, B, resArray);
            default -> throw new UnsupportedOperationException("This dtype "+targetType+" doesn't support this method");
        };
    }

    public NDArray matmul(NDArray b, NDArray resArray){
        ValidUtil.validateMatmulInputs(this, b);
        DType targetType = TypeUtil.promoteTypes(this.getDType(), b.getDType());
        NDArray A = this.cast(targetType);
        NDArray B = b.cast(targetType);
        long[] targetShape = new long[]{this.internalShapeUnsafe()[0], b.internalShapeUnsafe()[1]};
        NDArray targetRes = ValidUtil.validateResultArray(resArray, targetType, targetShape);
        ValidUtil.validateOutputBuffer(targetRes);
        return switch(targetType){
            case f32 -> MatMulOps.matmulFloat(A, B, targetRes);
            case f64 -> MatMulOps.matmulDouble(A, B, targetRes);
            case i32 -> MatMulOps.matmulInt(A, B, targetRes);
            default -> throw new UnsupportedOperationException("This dtype "+targetType+" doesn't support this method");
        };
    }

    // BooleanOps.java methods

    public NDArray and(NDArray b) {
        long[] targetShape = ShapeUtil.calculateBroadcastShape(this.internalShapeUnsafe(), b.internalShapeUnsafe());
        NDArray A = ValidUtil.prepareBroadcastOperand(this, targetShape, DType.bool);
        NDArray B = ValidUtil.prepareBroadcastOperand(b, targetShape, DType.bool);
        NDArray resArray = JNum.zeros(DType.bool, targetShape);
        return BooleanOps.and(A, B, resArray);
    }

    public NDArray and(NDArray b, NDArray resArray) {
        long[] targetShape = ShapeUtil.calculateBroadcastShape(this.internalShapeUnsafe(), b.internalShapeUnsafe());
        NDArray A = ValidUtil.prepareBroadcastOperand(this, targetShape, DType.bool);
        NDArray B = ValidUtil.prepareBroadcastOperand(b, targetShape, DType.bool);
        NDArray targetRes = ValidUtil.validateResultArray(resArray, DType.bool, targetShape);
        return BooleanOps.and(A, B, targetRes);
    }

    public NDArray or(NDArray b) {
        long[] targetShape = ShapeUtil.calculateBroadcastShape(this.internalShapeUnsafe(), b.internalShapeUnsafe());
        NDArray A = ValidUtil.prepareBroadcastOperand(this, targetShape, DType.bool);
        NDArray B = ValidUtil.prepareBroadcastOperand(b, targetShape, DType.bool);
        NDArray resArray = JNum.zeros(DType.bool, targetShape);
        return BooleanOps.or(A, B, resArray);
    }

    public NDArray or(NDArray b, NDArray resArray) {
        long[] targetShape = ShapeUtil.calculateBroadcastShape(this.internalShapeUnsafe(), b.internalShapeUnsafe());
        NDArray A = ValidUtil.prepareBroadcastOperand(this, targetShape, DType.bool);
        NDArray B = ValidUtil.prepareBroadcastOperand(b, targetShape, DType.bool);
        NDArray targetRes = ValidUtil.validateResultArray(resArray, DType.bool, targetShape);
        return BooleanOps.or(A, B, targetRes);
    }

    public NDArray xor(NDArray b) {
        long[] targetShape = ShapeUtil.calculateBroadcastShape(this.internalShapeUnsafe(), b.internalShapeUnsafe());
        NDArray A = ValidUtil.prepareBroadcastOperand(this, targetShape, DType.bool);
        NDArray B = ValidUtil.prepareBroadcastOperand(b, targetShape, DType.bool);
        NDArray resArray = JNum.zeros(DType.bool, targetShape);
        return BooleanOps.xor(A, B, resArray);
    }

    public NDArray xor(NDArray b, NDArray resArray) {
        long[] targetShape = ShapeUtil.calculateBroadcastShape(this.internalShapeUnsafe(), b.internalShapeUnsafe());
        NDArray A = ValidUtil.prepareBroadcastOperand(this, targetShape, DType.bool);
        NDArray B = ValidUtil.prepareBroadcastOperand(b, targetShape, DType.bool);
        NDArray targetRes = ValidUtil.validateResultArray(resArray, DType.bool, targetShape);
        return BooleanOps.xor(A, B, targetRes);
    }

    public NDArray not() {
        NDArray A = this.getDType() == DType.bool ? this : this.cast(DType.bool);
        NDArray safeThis = A.isContiguous() ? A : A.contiguous();
        NDArray resArray = JNum.zeros(DType.bool, safeThis.internalShapeUnsafe());
        return BooleanOps.not(safeThis, resArray);
    }

    public NDArray not(NDArray resArray) {
        NDArray A = this.getDType() == DType.bool ? this : this.cast(DType.bool);
        NDArray safeThis = A.isContiguous() ? A : A.contiguous();
        NDArray targetRes = ValidUtil.validateResultArray(resArray, DType.bool, safeThis.internalShapeUnsafe());
        return BooleanOps.not(safeThis, targetRes);
    }

    public boolean any() {
        NDArray A = this.getDType() == DType.bool ? this : this.cast(DType.bool);
        return BooleanOps.any(A);
    }

    public boolean all() {
        NDArray A = this.getDType() == DType.bool ? this : this.cast(DType.bool);
        return BooleanOps.all(A);
    }

}
