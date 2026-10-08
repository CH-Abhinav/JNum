package jnum;


import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Arrays;
import java.util.NoSuchElementException;

import jnum.internal.ops.ArithmeticOps;
import jnum.internal.ops.BooleanOps;
import jnum.internal.ops.CompareOps;
import jnum.internal.ops.UnaryOps;
import jnum.internal.ops.LinalgOps;
import jnum.internal.ops.GetOps;
import jnum.internal.ops.SetOps;
import jnum.internal.layout.NDIter;
import jnum.internal.ops.ReduceOps;
import jnum.internal.ops.TrigOps;
import jnum.internal.layout.ShapeUtil;
import jnum.internal.kernel.linalg.Det.SlogdetResult;
import jnum.internal.kernel.linalg.QR.QRResult;
import jnum.internal.kernel.linalg.Eigh.EighResult;
import jnum.internal.kernel.linalg.SVD.SVDResult;
import jnum.internal.kernel.linalg.Eig.EigResult;


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
        return GetOps.getPhysicalOffset(logicalIndex, shape, strides);
    }

    // --- Flat Getters ---

    public double getFlat(long index){ return GetOps.getFlat(this, index); }
    public float getFlatFloat(long index) { return GetOps.getFlatFloat(this, index); }
    public int getFlatInt(long index){ return GetOps.getFlatInt(this, index); }
    public double getFlatDouble(long index){ return GetOps.getFlatDouble(this, index); }
    public boolean getFlatBoolean(long index) { return GetOps.getFlatBoolean(this, index); }

    /*
        getters and setters
     */

    //TODO : getters/setters are get/setType() which is verbose. need to make it less verbose if possible.

    public double get(long... indices){ return GetOps.get(this, indices); }

    public void set(double val,long... indices){ SetOps.set(this, val, indices); }

    // --- FLOAT ---
    public float getFloat(long x) { return GetOps.getFloat(this, x); }
    public float getFloat(long x, long y) { return GetOps.getFloat(this, x, y); }
    public float getFloat(long x, long y, long z) { return GetOps.getFloat(this, x, y, z); }
    public float getFloat(long... indices){ return GetOps.getFloat(this, indices); }

    public void setFloat(float val, long x) { SetOps.setFloat(this, val, x); }
    public void setFloat(float val, long x, long y) { SetOps.setFloat(this, val, x, y); }
    public void setFloat(float val, long x, long y, long z) { SetOps.setFloat(this, val, x, y, z); }
    public void setFloat(float val,long... indices){ SetOps.setFloat(this, val, indices); }

    // --- DOUBLE ---
    public double getDouble(long x) { return GetOps.getDouble(this, x); }
    public double getDouble(long x, long y) { return GetOps.getDouble(this, x, y); }
    public double getDouble(long x, long y, long z) { return GetOps.getDouble(this, x, y, z); }
    public double getDouble(long... indices){ return GetOps.getDouble(this, indices); }

    public void setDouble(double val, long x) { SetOps.setDouble(this, val, x); }
    public void setDouble(double val, long x, long y) { SetOps.setDouble(this, val, x, y); }
    public void setDouble(double val, long x, long y, long z) { SetOps.setDouble(this, val, x, y, z); }
    public void setFloat(double val,long... indices){ SetOps.setDouble(this, val, indices); }

    // --- INT ---
    public int getInt(long x) { return GetOps.getInt(this, x); }
    public int getInt(long x, long y) { return GetOps.getInt(this, x, y); }
    public int getInt(long x, long y, long z) { return GetOps.getInt(this, x, y, z); }

    public void setInt(int val, long x) { SetOps.setInt(this, val, x); }
    public void setInt(int val, long x, long y) { SetOps.setInt(this, val, x, y); }
    public void setInt(int val, long x, long y, long z) { SetOps.setInt(this, val, x, y, z); }
    public void setInt(int val,long... indices){ SetOps.setInt(this, val, indices); }

    // --- BOOLEAN ---
    public boolean getBoolean(long x) { return GetOps.getBoolean(this, x); }
    public boolean getBoolean(long x, long y) { return GetOps.getBoolean(this, x, y); }
    public boolean getBoolean(long x, long y, long z) { return GetOps.getBoolean(this, x, y, z); }

    public void setBoolean(boolean val, long x) { SetOps.setBoolean(this, val, x); }
    public void setBoolean(boolean val, long x, long y) { SetOps.setBoolean(this, val, x, y); }
    public void setBoolean(boolean val, long x, long y, long z) { SetOps.setBoolean(this, val, x, y, z); }
    public void setBoolean(boolean val,long... indices){ SetOps.setBoolean(this, val, indices); }

    /*
        view and slice
     */

    public NDArray subview(long index) {
        if (this.dim() == 0) {
            throw new IllegalStateException("Cannot take a subview of a 0-dimensional scalar array.");
        }

        long dim0 = this.shape[0];
        if (index < 0) index += dim0;
        if (index < 0 || index >= dim0) {
            throw new IndexOutOfBoundsException("Index " + index + " out of bounds for axis 0 with size " + dim0);
        }

        long byteOffset = index * this.strides[0] * this.dtype.layout.byteSize();
        MemorySegment subSegment = this.data.asSlice(byteOffset);

        long[] subShape = Arrays.copyOfRange(this.shape, 1, this.shape.length);
        long[] subStrides = Arrays.copyOfRange(this.strides, 1, this.strides.length);

        return NDArray.ofRaw(subSegment, subShape, subStrides, this.dtype);
    }

    public NDArray subview(long... indices) {
        if (indices.length >= this.dim()) {
            throw new IllegalArgumentException("Cannot strip " + indices.length + " dimensions from an array of rank " + this.dim() + ". Use scalar getters instead.");
        }

        long byteOffset = 0;
        for (int d = 0; d < indices.length; d++) {
            long idx = indices[d];
            long dimSize = this.shape[d];
            if (idx < 0) idx += dimSize;
            if (idx < 0 || idx >= dimSize) {
                throw new IndexOutOfBoundsException("Index " + idx + " out of bounds for axis " + d + " with size " + dimSize);
            }
            byteOffset += idx * this.strides[d];
        }

        byteOffset *= this.dtype.layout.byteSize();
        MemorySegment subSegment = this.data.asSlice(byteOffset);

        long[] subShape = Arrays.copyOfRange(this.shape, indices.length, this.shape.length);
        long[] subStrides = Arrays.copyOfRange(this.strides, indices.length, this.strides.length);

        return NDArray.ofRaw(subSegment, subShape, subStrides, this.dtype);
    }

    public NDArray slice(Slice... slices) {
        int rank = this.dim();
        if (slices.length > rank) {
            throw new IllegalArgumentException("Too many slices: " + slices.length + " provided for array of rank " + rank);
        }

        long[] newShape = new long[rank];
        long[] newStrides = new long[rank];
        long baseElementOffset = 0;

        for (int d = 0; d < rank; d++) {
            Slice s = d < slices.length ? slices[d] : Slice.all();
            Slice.ResolvedSlice resolved = s.resolve(this.shape[d]);
            newShape[d] = resolved.length();
            newStrides[d] = this.strides[d] * resolved.step();
            baseElementOffset += resolved.start() * this.strides[d];
        }

        long byteOffset = baseElementOffset * this.dtype.layout.byteSize();
        MemorySegment slicedSegment = this.data.asSlice(byteOffset);

        return NDArray.ofRaw(slicedSegment, newShape, newStrides, this.dtype);
    }

    public NDArray slice(String sliceExpr) {
        String[] parts = sliceExpr.split(",");
        Slice[] sliceObjs = new Slice[parts.length];

        for (int i = 0; i < parts.length; i++) {
            String part = parts[i].trim();
            if (part.equals(":")) {
                sliceObjs[i] = Slice.all();
                continue;
            }
            String[] tokens = part.split(":", -1);
            long step = tokens.length == 3 && !tokens[2].isEmpty() ? Long.parseLong(tokens[2]) : 1;
            long start = tokens.length > 0 && !tokens[0].isEmpty() ? Long.parseLong(tokens[0]) : (step > 0 ? Slice.UNBOUNDED_START : Slice.UNBOUNDED_STOP);
            long stop = tokens.length > 1 && !tokens[1].isEmpty() ? Long.parseLong(tokens[1]) : (step > 0 ? Slice.UNBOUNDED_STOP : Slice.UNBOUNDED_START);

            sliceObjs[i] = new Slice(start, stop, step);
        }
        return this.slice(sliceObjs);
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

    //squeeze and unsqueeze

    public NDArray unsqueeze(int axis) {
        int currentDim = this.dim();
        if (axis < 0) axis += (currentDim + 1);
        if (axis < 0 || axis > currentDim) {
            throw new IllegalArgumentException("Axis " + axis + " is out of bounds for array of dimension " + currentDim);
        }

        long[] newShape = new long[currentDim + 1];
        for (int i = 0, j = 0; i < newShape.length; i++) {
            if (i == axis) {
                newShape[i] = 1;
            } else {
                newShape[i] = this.internalShapeUnsafe()[j++];
            }
        }
        return this.reshape(newShape);
    }

    public NDArray squeeze(int axis) {
        int currentDim = this.dim();
        if (axis < 0) axis += currentDim;
        if (axis < 0 || axis >= currentDim) {
            throw new IllegalArgumentException("Axis " + axis + " is out of bounds for array of dimension " + currentDim);
        }
        if (this.internalShapeUnsafe()[axis] != 1) {
            throw new IllegalArgumentException("Cannot squeeze axis " + axis + " because its size is not 1. Shape is: " + this.shapeString());
        }

        long[] newShape = new long[currentDim - 1];
        for (int i = 0, j = 0; i < currentDim; i++) {
            if (i != axis) {
                newShape[j++] = this.internalShapeUnsafe()[i];
            }
        }
        return this.reshape(newShape);
    }

    public double max() {
        return ReduceOps.max(this);
    }

    public double min() {
        return ReduceOps.min(this);
    }

    public double sum() {
        return ReduceOps.sum(this);
    }

    public NDArray sum(int axis) {
        return ReduceOps.sum(this, axis);
    }

    public NDArray max(int axis) {
        return ReduceOps.max(this, axis);
    }

    public NDArray min(int axis) {
        return ReduceOps.min(this, axis);
    }

    public NDArray sum(int axis, boolean keepDims) {
        NDArray res = this.sum(axis);
        if (keepDims) {
            int targetAxis = axis < 0 ? axis + this.dim() : axis;
            return res.unsqueeze(targetAxis);
        }
        return res;
    }

    public NDArray max(int axis, boolean keepDims) {
        NDArray res = this.max(axis);
        if (keepDims) {
            int targetAxis = axis < 0 ? axis + this.dim() : axis;
            return res.unsqueeze(targetAxis);
        }
        return res;
    }

    public NDArray min(int axis, boolean keepDims) {
        NDArray res = this.min(axis);
        if (keepDims) {
            int targetAxis = axis < 0 ? axis + this.dim() : axis;
            return res.unsqueeze(targetAxis);
        }
        return res;
    }

    public double dot(NDArray b) {
        return ReduceOps.dot(this, b);
    }

    public double avg() {
        return this.sum() / (double) this.getSize();
    }

    public NDArray maximum(NDArray b) {
        return CompareOps.maximum(this, b);
    }

    public NDArray maximum(float b) {
        return CompareOps.maximum(this, b);
    }

    public NDArray maximum(int b) {
        return CompareOps.maximum(this, b);
    }

    public NDArray maximum(double b) {
        return CompareOps.maximum(this, b);
    }

    public NDArray minimum(NDArray b) {
        return CompareOps.minimum(this, b);
    }

    public NDArray minimum(float b) {
        return CompareOps.minimum(this, b);
    }

    public NDArray minimum(int b) {
        return CompareOps.minimum(this, b);
    }

    public NDArray minimum(double b) {
        return CompareOps.minimum(this, b);
    }

    //ArithmaticOps.java 

    //addition operation

    public NDArray add(NDArray b) {
        return ArithmeticOps.add(this, b);
    }
    
    public NDArray add(NDArray b, NDArray resArray) {
        return ArithmeticOps.add(this, b, resArray);
    }

    public NDArray add(float b) {
        return ArithmeticOps.add(this, b);
    }

    public NDArray add(int b) {
        return ArithmeticOps.add(this, b);
    }

    public NDArray add(double b) {
        return ArithmeticOps.add(this, b);
    }

    public NDArray add(float b, NDArray resArray) {
        return ArithmeticOps.add(this, b, resArray);
    }

    public NDArray add(int b, NDArray resArray) {
        return ArithmeticOps.add(this, b, resArray);
    }

    public NDArray add(double b, NDArray resArray) {
        return ArithmeticOps.add(this, b, resArray);
    }

    //subtract operations

    public NDArray sub(NDArray b) {
        return ArithmeticOps.sub(this, b);
    }

    public NDArray sub(NDArray b, NDArray resArray) {
        return ArithmeticOps.sub(this, b, resArray);
    }

    public NDArray sub(float b) {
        return ArithmeticOps.sub(this, b);
    }

    public NDArray sub(int b) {
        return ArithmeticOps.sub(this, b);
    }

    public NDArray sub(double b) {
        return ArithmeticOps.sub(this, b);
    }

    public NDArray sub(float b, NDArray resArray) {
        return ArithmeticOps.sub(this, b, resArray);
    }

    public NDArray sub(int b, NDArray resArray) {
        return ArithmeticOps.sub(this, b, resArray);
    }

    public NDArray sub(double b, NDArray resArray) {
        return ArithmeticOps.sub(this, b, resArray);
    }

    //multiplication operations 

    public NDArray mul(NDArray b) {
        return ArithmeticOps.mul(this, b);
    }

    public NDArray mul(NDArray b, NDArray resArray) {
        return ArithmeticOps.mul(this, b, resArray);
    }

    public NDArray mul(float b) {
        return ArithmeticOps.mul(this, b);
    }

    public NDArray mul(int b) {
        return ArithmeticOps.mul(this, b);
    }

    public NDArray mul(double b) {
        return ArithmeticOps.mul(this, b);
    }

    public NDArray mul(float b, NDArray resArray) {
        return ArithmeticOps.mul(this, b, resArray);
    }

    public NDArray mul(int b, NDArray resArray) {
        return ArithmeticOps.mul(this, b, resArray);
    }

    public NDArray mul(double b, NDArray resArray) {
        return ArithmeticOps.mul(this, b, resArray);
    }

    //division operations

    public NDArray div(NDArray b) {
        return ArithmeticOps.div(this, b);
    }

    public NDArray div(NDArray b, NDArray resArray) {
        return ArithmeticOps.div(this, b, resArray);
    }

    public NDArray div(float b) {
        return ArithmeticOps.div(this, b);
    }

    public NDArray div(int b) {
        return ArithmeticOps.div(this, b);
    }

    public NDArray div(double b) {
        return ArithmeticOps.div(this, b);
    }

    public NDArray div(float b, NDArray resArray) {
        return ArithmeticOps.div(this, b, resArray);
    }

    public NDArray div(int b, NDArray resArray) {
        return ArithmeticOps.div(this, b, resArray);
    }

    public NDArray div(double b, NDArray resArray) {
        return ArithmeticOps.div(this, b, resArray);
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

    //UnaryOps.java methods

    public NDArray sqrt() {
        return UnaryOps.sqrt(this);
    }

    public NDArray abs() {
        return UnaryOps.abs(this);
    }

    public NDArray exp() {
        return UnaryOps.exp(this);
    }

    public NDArray log() {
        return UnaryOps.log(this);
    }

    public NDArray log10() {
        return UnaryOps.log10(this);
    }

    public NDArray sigmoid() {
        return UnaryOps.sigmoid(this);
    }

    //TrigOps.java methods

    public NDArray sin() {
        return TrigOps.sin(this);
    }

    public NDArray cos() {
        return TrigOps.cos(this);
    }

    public NDArray tan() {
        return TrigOps.tan(this);
    }

    public NDArray sinh() {
        return TrigOps.sinh(this);
    }

    public NDArray cosh() {
        return TrigOps.cosh(this);
    }

    public NDArray tanh() {
        return TrigOps.tanh(this);
    }

    //LinalgOps.java methods

    public NDArray matmul(NDArray b) {
        return LinalgOps.matmul(this, b);
    }

    public NDArray matmul(NDArray b, NDArray resArray) {
        return LinalgOps.matmul(this, b, resArray);
    }

    public double trace() {
        return LinalgOps.trace(this);
    }

    public double trace(int offset) {
        return LinalgOps.trace(this, offset);
    }

    public double norm() {
        return LinalgOps.norm(this);
    }

    public double norm(int ord) {
        return LinalgOps.norm(this, ord);
    }

    public double det() {
        return LinalgOps.det(this);
    }

    public SlogdetResult slogdet() {
        return LinalgOps.slogdet(this);
    }

    public NDArray inv() {
        return LinalgOps.inv(this);
    }

    public NDArray inv(Arena arena) {
        return LinalgOps.inv(this, arena);
    }

    public NDArray solve(NDArray b) {
        return LinalgOps.solve(this, b);
    }

    public NDArray solve(NDArray b, Arena arena) {
        return LinalgOps.solve(this, b, arena);
    }

    public NDArray cholesky() {
        return LinalgOps.cholesky(this);
    }

    public NDArray cholesky(Arena arena) {
        return LinalgOps.cholesky(this, arena);
    }

    public QRResult qr() {
        return LinalgOps.qr(this);
    }

    public QRResult qr(Arena arena) {
        return LinalgOps.qr(this, arena);
    }

    public EighResult eigh() {
        return LinalgOps.eigh(this);
    }

    public EighResult eigh(Arena arena) {
        return LinalgOps.eigh(this, arena);
    }

    public SVDResult svd() {
        return LinalgOps.svd(this);
    }

    public SVDResult svd(Arena arena) {
        return LinalgOps.svd(this, arena);
    }

    public int matrixRank() {
        return LinalgOps.matrixRank(this);
    }

    public int matrixRank(double tol) {
        return LinalgOps.matrixRank(this, tol);
    }

    public int matrixRank(double tol, Arena arena) {
        return LinalgOps.matrixRank(this, tol, arena);
    }

    public double cond() {
        return LinalgOps.cond(this);
    }

    public double cond(Arena arena) {
        return LinalgOps.cond(this, arena);
    }

    public NDArray pinv() {
        return LinalgOps.pinv(this);
    }

    public NDArray pinv(double rcond) {
        return LinalgOps.pinv(this, rcond);
    }

    public NDArray pinv(double rcond, Arena arena) {
        return LinalgOps.pinv(this, rcond, arena);
    }

    public EigResult eig() {
        return LinalgOps.eig(this);
    }

    public EigResult eig(Arena arena) {
        return LinalgOps.eig(this, arena);
    }

    public double cosineSimilarity(NDArray b) {
        return LinalgOps.cosineSimilarity(this, b);
    }

    // BooleanOps.java methods

    public NDArray and(NDArray b) {
        return BooleanOps.and(this, b);
    }

    public NDArray and(NDArray b, NDArray resArray) {
        return BooleanOps.and(this, b, resArray);
    }

    public NDArray or(NDArray b) {
        return BooleanOps.or(this, b);
    }

    public NDArray or(NDArray b, NDArray resArray) {
        return BooleanOps.or(this, b, resArray);
    }

    public NDArray xor(NDArray b) {
        return BooleanOps.xor(this, b);
    }

    public NDArray xor(NDArray b, NDArray resArray) {
        return BooleanOps.xor(this, b, resArray);
    }

    public NDArray not() {
        return BooleanOps.not(this);
    }

    public NDArray not(NDArray resArray) {
        return BooleanOps.not(this, resArray);
    }

    public boolean any() {
        return BooleanOps.any(this);
    }

    public boolean all() {
        return BooleanOps.all(this);
    }

}
