/*
 * Copyright © 2026 Abhinav Chinnala (chinnala.abhinav5@gmail.com)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package jnum;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Arrays;

import jnum.internal.kernel.linalg.Det.SlogdetResult;
import jnum.internal.kernel.linalg.QR.QRResult;
import jnum.internal.kernel.linalg.Eigh.EighResult;
import jnum.internal.kernel.linalg.SVD.SVDResult;
import jnum.internal.kernel.linalg.Eig.EigResult;
import jnum.internal.layout.LayoutOps;
import jnum.internal.layout.NDIter;
import jnum.internal.layout.ShapeUtil;
import jnum.internal.memory.MemoryOps;
import jnum.internal.ops.ArithmeticOps;
import jnum.internal.ops.BooleanOps;
import jnum.internal.ops.CompareOps;
import jnum.internal.ops.GetOps;
import jnum.internal.ops.LinalgOps;
import jnum.internal.ops.ReduceOps;
import jnum.internal.ops.SetOps;
import jnum.internal.ops.TrigOps;
import jnum.internal.ops.UnaryOps;

/**
 * A multidimensional array (tensor) supporting hardware-accelerated SIMD numerical computations.
 *
 * <p>An {@code NDArray} encapsulates a contiguous or strided segment of native memory
 * managed via Java's Foreign Function &amp; Memory (FFM) {@link MemorySegment}, coupled with
 * dimensional {@code shape} and physical {@code strides} information.</p>
 *
 * <p>Instances are immutable in shape and layout, but their underlying memory segment
 * may be mutated via in-place operations (e.g. {@code addi}) or setter methods.</p>
 */
public class NDArray {

    // =========================================================================
    // Parameters & Construction
    // =========================================================================

    /** The underlying FFM memory segment storing the tensor data. */
    private final MemorySegment data;

    /** Dimensions of the tensor. */
    private final long[] shape;

    /** Number of elements to step in physical memory to advance by 1 along each dimension. */
    private final long[] strides;

    /** Total number of logical elements contained in the tensor. */
    private final long size;

    /** Data type descriptor of the elements. */
    private final DType dtype;

    /**
     * Private constructor to prevent direct instantiation without factory methods.
     */
    private NDArray() {
        throw new AssertionError("NDArray cannot be instantiated via default constructor.");
    }

    /**
     * Internal canonical constructor.
     */
    private NDArray(MemorySegment data, long[] shape, long[] strides, DType dType) {
        this.data = data;
        this.shape = shape;
        this.strides = strides;
        long calcSize = 1;
        for (long dim : shape) calcSize *= dim;
        this.size = calcSize;
        this.dtype = dType;
    }

    /**
     * Creates an {@code NDArray} wrapping a raw memory segment with explicit shape and strides.
     *
     * <p><b>Warning:</b> This method does not validate bounds or memory alignment.
     * It is intended for internal operations and factory methods.</p>
     *
     * @param data backing native memory segment
     * @param shape dimensions of the tensor
     * @param strides element-wise strides along each dimension
     * @param dtype data type of the elements
     * @return a new {@link NDArray} view over the provided memory segment
     */
    public static NDArray ofRaw(MemorySegment data, long[] shape, long[] strides, DType dtype) {
        return new NDArray(data, shape, strides, dtype);
    }

    // =========================================================================
    // Metadata & Dimensions (Getters & Setters)
    // =========================================================================

    /**
     * Returns a copy of the tensor dimensions.
     *
     * @return a defensive copy of the shape array
     */
    public long[] getShape() {
        return internalShapeUnsafe().clone();
    }

    /**
     * Returns the rank (number of dimensions) of this tensor.
     *
     * @return number of dimensions
     */
    public int dim() {
        return internalShapeUnsafe().length;
    }

    /**
     * Returns the total number of logical elements in the tensor.
     *
     * @return total element count
     */
    public long getSize() {
        return size;
    }

    /**
     * Returns the {@link DType} of the elements stored in this tensor.
     *
     * @return the data type
     */
    public DType getDType() {
        return dtype;
    }

    /**
     * Returns a formatted string representing the shape tuple (e.g., {@code "(3, 224, 224)"}).
     *
     * @return shape tuple string
     */
    public String shapeString() {
        return Arrays.toString(internalShapeUnsafe()).replace("[", "(").replace("]", ")");
    }

    /**
     * Internal unsafe accessor returning the backing shape array without defensive copying.
     *
     * @return internal shape array
     */
    public long[] internalShapeUnsafe() {
        return shape;
    }

    /**
     * Internal unsafe accessor returning the backing strides array without defensive copying.
     *
     * @return internal strides array
     */
    public long[] internalStridesUnsafe() {
        return strides;
    }

    /**
     * Returns the underlying FFM {@link MemorySegment}.
     *
     * @return mutable memory segment
     */
    public MemorySegment getData() {
        return data;
    }

    /**
     * Returns a read-only view of the underlying FFM {@link MemorySegment}.
     *
     * @return read-only memory segment
     */
    public MemorySegment getDataReadOnly() {
        return data.asReadOnly();
    }

    // =========================================================================
    // Element Access (Getters & Setters)
    // =========================================================================

    /**
     * Retrieves the element at the specified multidimensional coordinates as a {@code double}.
     *
     * @param indices multidimensional coordinates
     * @return element value cast to double
     * @throws IndexOutOfBoundsException if coordinates are out of bounds
     */
    public double get(long... indices) { return GetOps.get(this, indices); }

    /**
     * Sets the element at the specified multidimensional coordinates with a {@code double} value.
     *
     * @param val value to set
     * @param indices multidimensional coordinates
     * @throws IndexOutOfBoundsException if coordinates are out of bounds
     */
    public void set(double val, long... indices) { SetOps.set(this, val, indices); }

    // --- FLOAT ACCESSORS (Zero-allocation overloads for 1D, 2D, 3D, and N-D) ---

    /**
     * Retrieves the float element at coordinate {@code [x]} in a 1D tensor without heap allocation.
     *
     * @param x coordinate along axis 0
     * @return the float value at the coordinate
     * @throws IndexOutOfBoundsException if {@code x} is out of bounds
     */
    public float getFloat(long x) { return GetOps.getFloat(this, x); }

    /**
     * Retrieves the float element at coordinates {@code [x, y]} in a 2D tensor without heap allocation.
     *
     * @param x coordinate along axis 0 (row)
     * @param y coordinate along axis 1 (column)
     * @return the float value at the coordinates
     * @throws IndexOutOfBoundsException if coordinates are out of bounds
     */
    public float getFloat(long x, long y) { return GetOps.getFloat(this, x, y); }

    /**
     * Retrieves the float element at coordinates {@code [x, y, z]} in a 3D tensor without heap allocation.
     *
     * @param x coordinate along axis 0
     * @param y coordinate along axis 1
     * @param z coordinate along axis 2
     * @return the float value at the coordinates
     * @throws IndexOutOfBoundsException if coordinates are out of bounds
     */
    public float getFloat(long x, long y, long z) { return GetOps.getFloat(this, x, y, z); }

    /**
     * Retrieves the float element at the given N-dimensional coordinates.
     *
     * @param indices multidimensional coordinate array
     * @return the float value at the coordinates
     * @throws IndexOutOfBoundsException if coordinates are out of bounds
     */
    public float getFloat(long... indices) { return GetOps.getFloat(this, indices); }

    /**
     * Sets the float element at coordinate {@code [x]} in a 1D tensor without heap allocation.
     *
     * @param val float value to write
     * @param x coordinate along axis 0
     * @throws IndexOutOfBoundsException if {@code x} is out of bounds
     */
    public void setFloat(float val, long x) { SetOps.setFloat(this, val, x); }

    /**
     * Sets the float element at coordinates {@code [x, y]} in a 2D tensor without heap allocation.
     *
     * @param val float value to write
     * @param x coordinate along axis 0 (row)
     * @param y coordinate along axis 1 (column)
     * @throws IndexOutOfBoundsException if coordinates are out of bounds
     */
    public void setFloat(float val, long x, long y) { SetOps.setFloat(this, val, x, y); }

    /**
     * Sets the float element at coordinates {@code [x, y, z]} in a 3D tensor without heap allocation.
     *
     * @param val float value to write
     * @param x coordinate along axis 0
     * @param y coordinate along axis 1
     * @param z coordinate along axis 2
     * @throws IndexOutOfBoundsException if coordinates are out of bounds
     */
    public void setFloat(float val, long x, long y, long z) { SetOps.setFloat(this, val, x, y, z); }

    /**
     * Sets the float element at the given N-dimensional coordinates.
     *
     * @param val float value to write
     * @param indices multidimensional coordinate array
     * @throws IndexOutOfBoundsException if coordinates are out of bounds
     */
    public void setFloat(float val, long... indices) { SetOps.setFloat(this, val, indices); }

    // --- DOUBLE ACCESSORS (Zero-allocation overloads for 1D, 2D, 3D, and N-D) ---

    /**
     * Retrieves the double element at coordinate {@code [x]} in a 1D tensor without heap allocation.
     *
     * @param x coordinate along axis 0
     * @return the double value at the coordinate
     * @throws IndexOutOfBoundsException if {@code x} is out of bounds
     */
    public double getDouble(long x) { return GetOps.getDouble(this, x); }

    /**
     * Retrieves the double element at coordinates {@code [x, y]} in a 2D tensor without heap allocation.
     *
     * @param x coordinate along axis 0 (row)
     * @param y coordinate along axis 1 (column)
     * @return the double value at the coordinates
     * @throws IndexOutOfBoundsException if coordinates are out of bounds
     */
    public double getDouble(long x, long y) { return GetOps.getDouble(this, x, y); }

    /**
     * Retrieves the double element at coordinates {@code [x, y, z]} in a 3D tensor without heap allocation.
     *
     * @param x coordinate along axis 0
     * @param y coordinate along axis 1
     * @param z coordinate along axis 2
     * @return the double value at the coordinates
     * @throws IndexOutOfBoundsException if coordinates are out of bounds
     */
    public double getDouble(long x, long y, long z) { return GetOps.getDouble(this, x, y, z); }

    /**
     * Retrieves the double element at the given N-dimensional coordinates.
     *
     * @param indices multidimensional coordinate array
     * @return the double value at the coordinates
     * @throws IndexOutOfBoundsException if coordinates are out of bounds
     */
    public double getDouble(long... indices) { return GetOps.getDouble(this, indices); }

    /**
     * Sets the double element at coordinate {@code [x]} in a 1D tensor without heap allocation.
     *
     * @param val double value to write
     * @param x coordinate along axis 0
     * @throws IndexOutOfBoundsException if {@code x} is out of bounds
     */
    public void setDouble(double val, long x) { SetOps.setDouble(this, val, x); }

    /**
     * Sets the double element at coordinates {@code [x, y]} in a 2D tensor without heap allocation.
     *
     * @param val double value to write
     * @param x coordinate along axis 0 (row)
     * @param y coordinate along axis 1 (column)
     * @throws IndexOutOfBoundsException if coordinates are out of bounds
     */
    public void setDouble(double val, long x, long y) { SetOps.setDouble(this, val, x, y); }

    /**
     * Sets the double element at coordinates {@code [x, y, z]} in a 3D tensor without heap allocation.
     *
     * @param val double value to write
     * @param x coordinate along axis 0
     * @param y coordinate along axis 1
     * @param z coordinate along axis 2
     * @throws IndexOutOfBoundsException if coordinates are out of bounds
     */
    public void setDouble(double val, long x, long y, long z) { SetOps.setDouble(this, val, x, y, z); }

    /**
     * Sets the double element at the given N-dimensional coordinates.
     *
     * @param val double value to write
     * @param indices multidimensional coordinate array
     * @throws IndexOutOfBoundsException if coordinates are out of bounds
     */
    public void setDouble(double val, long... indices) { SetOps.setDouble(this, val, indices); }

    // --- INT ACCESSORS (Zero-allocation overloads for 1D, 2D, 3D, and N-D) ---

    /**
     * Retrieves the int element at coordinate {@code [x]} in a 1D tensor without heap allocation.
     *
     * @param x coordinate along axis 0
     * @return the int value at the coordinate
     * @throws IndexOutOfBoundsException if {@code x} is out of bounds
     */
    public int getInt(long x) { return GetOps.getInt(this, x); }

    /**
     * Retrieves the int element at coordinates {@code [x, y]} in a 2D tensor without heap allocation.
     *
     * @param x coordinate along axis 0 (row)
     * @param y coordinate along axis 1 (column)
     * @return the int value at the coordinates
     * @throws IndexOutOfBoundsException if coordinates are out of bounds
     */
    public int getInt(long x, long y) { return GetOps.getInt(this, x, y); }

    /**
     * Retrieves the int element at coordinates {@code [x, y, z]} in a 3D tensor without heap allocation.
     *
     * @param x coordinate along axis 0
     * @param y coordinate along axis 1
     * @param z coordinate along axis 2
     * @return the int value at the coordinates
     * @throws IndexOutOfBoundsException if coordinates are out of bounds
     */
    public int getInt(long x, long y, long z) { return GetOps.getInt(this, x, y, z); }

    /**
     * Sets the int element at coordinate {@code [x]} in a 1D tensor without heap allocation.
     *
     * @param val int value to write
     * @param x coordinate along axis 0
     * @throws IndexOutOfBoundsException if {@code x} is out of bounds
     */
    public void setInt(int val, long x) { SetOps.setInt(this, val, x); }

    /**
     * Sets the int element at coordinates {@code [x, y]} in a 2D tensor without heap allocation.
     *
     * @param val int value to write
     * @param x coordinate along axis 0 (row)
     * @param y coordinate along axis 1 (column)
     * @throws IndexOutOfBoundsException if coordinates are out of bounds
     */
    public void setInt(int val, long x, long y) { SetOps.setInt(this, val, x, y); }

    /**
     * Sets the int element at coordinates {@code [x, y, z]} in a 3D tensor without heap allocation.
     *
     * @param val int value to write
     * @param x coordinate along axis 0
     * @param y coordinate along axis 1
     * @param z coordinate along axis 2
     * @throws IndexOutOfBoundsException if coordinates are out of bounds
     */
    public void setInt(int val, long x, long y, long z) { SetOps.setInt(this, val, x, y, z); }

    /**
     * Sets the int element at the given N-dimensional coordinates.
     *
     * @param val int value to write
     * @param indices multidimensional coordinate array
     * @throws IndexOutOfBoundsException if coordinates are out of bounds
     */
    public void setInt(int val, long... indices) { SetOps.setInt(this, val, indices); }

    // --- BOOLEAN ACCESSORS (Zero-allocation overloads for 1D, 2D, 3D, and N-D) ---

    /**
     * Retrieves the boolean flag at coordinate {@code [x]} in a 1D tensor without heap allocation.
     *
     * @param x coordinate along axis 0
     * @return true if non-zero, false otherwise
     * @throws IndexOutOfBoundsException if {@code x} is out of bounds
     */
    public boolean getBoolean(long x) { return GetOps.getBoolean(this, x); }

    /**
     * Retrieves the boolean flag at coordinates {@code [x, y]} in a 2D tensor without heap allocation.
     *
     * @param x coordinate along axis 0 (row)
     * @param y coordinate along axis 1 (column)
     * @return true if non-zero, false otherwise
     * @throws IndexOutOfBoundsException if coordinates are out of bounds
     */
    public boolean getBoolean(long x, long y) { return GetOps.getBoolean(this, x, y); }

    /**
     * Retrieves the boolean flag at coordinates {@code [x, y, z]} in a 3D tensor without heap allocation.
     *
     * @param x coordinate along axis 0
     * @param y coordinate along axis 1
     * @param z coordinate along axis 2
     * @return true if non-zero, false otherwise
     * @throws IndexOutOfBoundsException if coordinates are out of bounds
     */
    public boolean getBoolean(long x, long y, long z) { return GetOps.getBoolean(this, x, y, z); }

    /**
     * Sets the boolean flag at coordinate {@code [x]} in a 1D tensor without heap allocation.
     *
     * @param val boolean flag to write
     * @param x coordinate along axis 0
     * @throws IndexOutOfBoundsException if {@code x} is out of bounds
     */
    public void setBoolean(boolean val, long x) { SetOps.setBoolean(this, val, x); }

    /**
     * Sets the boolean flag at coordinates {@code [x, y]} in a 2D tensor without heap allocation.
     *
     * @param val boolean flag to write
     * @param x coordinate along axis 0 (row)
     * @param y coordinate along axis 1 (column)
     * @throws IndexOutOfBoundsException if coordinates are out of bounds
     */
    public void setBoolean(boolean val, long x, long y) { SetOps.setBoolean(this, val, x, y); }

    /**
     * Sets the boolean flag at coordinates {@code [x, y, z]} in a 3D tensor without heap allocation.
     *
     * @param val boolean flag to write
     * @param x coordinate along axis 0
     * @param y coordinate along axis 1
     * @param z coordinate along axis 2
     * @throws IndexOutOfBoundsException if coordinates are out of bounds
     */
    public void setBoolean(boolean val, long x, long y, long z) { SetOps.setBoolean(this, val, x, y, z); }

    /**
     * Sets the boolean flag at the given N-dimensional coordinates.
     *
     * @param val boolean flag to write
     * @param indices multidimensional coordinate array
     * @throws IndexOutOfBoundsException if coordinates are out of bounds
     */
    public void setBoolean(boolean val, long... indices) { SetOps.setBoolean(this, val, indices); }

    // --- FLAT (1D LINEAR) GETTERS ---

    /**
     * Retrieves the element at the flat row-major index, promoted to {@code double}.
     *
     * @param index linear flat index in range {@code [0, size - 1]}
     * @return element value cast to double
     * @throws IndexOutOfBoundsException if {@code index} is out of bounds
     */
    public double getFlat(long index) { return GetOps.getFlat(this, index); }

    /**
     * Retrieves the float element at the flat row-major index.
     *
     * @param index linear flat index in range {@code [0, size - 1]}
     * @return float value at the flat index
     * @throws IndexOutOfBoundsException if {@code index} is out of bounds
     */
    public float getFlatFloat(long index) { return GetOps.getFlatFloat(this, index); }

    /**
     * Retrieves the int element at the flat row-major index.
     *
     * @param index linear flat index in range {@code [0, size - 1]}
     * @return int value at the flat index
     * @throws IndexOutOfBoundsException if {@code index} is out of bounds
     */
    public int getFlatInt(long index) { return GetOps.getFlatInt(this, index); }

    /**
     * Retrieves the double element at the flat row-major index.
     *
     * @param index linear flat index in range {@code [0, size - 1]}
     * @return double value at the flat index
     * @throws IndexOutOfBoundsException if {@code index} is out of bounds
     */
    public double getFlatDouble(long index) { return GetOps.getFlatDouble(this, index); }

    /**
     * Retrieves the boolean flag at the flat row-major index.
     *
     * @param index linear flat index in range {@code [0, size - 1]}
     * @return true if non-zero, false otherwise
     * @throws IndexOutOfBoundsException if {@code index} is out of bounds
     */
    public boolean getFlatBoolean(long index) { return GetOps.getFlatBoolean(this, index); }

    /**
     * Finds the coordinates of the first element matching {@code b}.
     *
     * @param b scalar value to search for
     * @return multidimensional coordinate array
     * @throws java.util.NoSuchElementException if no matching element is found
     */
    public long[] indexOf(double b) {
        return GetOps.indexOf(this, b);
    }

    // =========================================================================
    // Structural & Zero-Copy Views
    // =========================================================================

    /**
     * Reverses the order of axes of this tensor, returning a zero-copy transposed view.
     *
     * <p>Transposition is accomplished purely by reversing the {@code shape} and {@code strides}
     * arrays, referencing the exact same native {@link MemorySegment} without copying or reallocating
     * any underlying data bytes.</p>
     *
     * @return a zero-copy transposed {@link NDArray} view sharing the underlying buffer
     */
    public NDArray transpose() {
        return LayoutOps.transpose(this);
    }

    /**
     * Broadcasts this tensor to a larger compatible shape by creating a zero-copy view with stride 0.
     *
     * <p>Broadcasting adheres to standard NumPy broadcasting rules. Dimensions of size 1 are expanded
     * by setting their physical stride to {@code 0}, allowing a single slice of physical memory to be
     * repeated virtually across higher ranks without duplicating memory.</p>
     *
     * @param shape target broadcast dimensions compatible with the current shape
     * @return a zero-copy broadcast view of this tensor sharing the underlying buffer
     * @throws IllegalArgumentException if the shapes cannot be aligned according to broadcasting rules
     */
    public NDArray broadcastTo(long... shape) {
        return LayoutOps.broadcastTo(this, shape);
    }

    /**
     * Extracts a sub-tensor by fixing the first dimension (axis 0) to {@code index}.
     *
     * <p>The returned tensor is a zero-copy view into the same native memory segment, offset by
     * {@code index * strides[0]}, with the first dimension dropped (rank reduced by 1).</p>
     *
     * @param index coordinate along axis 0
     * @return a zero-copy sub-tensor view with rank reduced by 1
     * @throws IndexOutOfBoundsException if {@code index} is outside {@code [0, shape[0] - 1]}
     */
    public NDArray subview(long index) {
        return LayoutOps.subview(this, index);
    }

    /**
     * Extracts a sub-tensor by fixing the leading dimensions to the specified {@code indices}.
     *
     * <p>The returned tensor is a zero-copy view with rank reduced by the number of indices provided.</p>
     *
     * @param indices coordinates for leading axes
     * @return a zero-copy sub-tensor view
     * @throws IndexOutOfBoundsException if any coordinate is out of bounds or more indices than dimensions are provided
     */
    public NDArray subview(long... indices) {
        return LayoutOps.subview(this, indices);
    }

    /**
     * Slices this tensor across multiple dimensions using {@link Slice} definitions.
     *
     * <p>Produces a zero-copy view with adjusted memory offset, shape, and strides
     * (supporting start, stop, and positive/negative steps).</p>
     *
     * @param slices slice specifications per axis
     * @return a sliced zero-copy {@link NDArray} view sharing the same backing buffer
     * @throws IllegalArgumentException if slice parameters are invalid or rank mismatch occurs
     */
    public NDArray slice(Slice... slices) {
        return LayoutOps.slice(this, slices);
    }

    /**
     * Slices this tensor using a NumPy-style string expression (e.g. {@code "0:5, :"}).
     *
     * @param sliceExpr comma-separated slice string expression
     * @return a sliced zero-copy {@link NDArray} view sharing the same backing buffer
     * @throws IllegalArgumentException if the slice expression syntax is invalid
     */
    public NDArray slice(String sliceExpr) {
        return LayoutOps.slice(this, sliceExpr);
    }

    /**
     * Removes a singleton dimension (size 1) at the specified axis.
     *
     * <p>Returns a zero-copy view with identical physical data layout but rank reduced by 1.</p>
     *
     * @param axis dimension index to squeeze
     * @return reshaped zero-copy {@link NDArray} view without the singleton axis
     * @throws IllegalArgumentException if the dimension at {@code axis} is not of size 1, or axis is invalid
     */
    public NDArray squeeze(int axis) {
        return LayoutOps.squeeze(this, axis);
    }

    /**
     * Inserts a singleton dimension (size 1) at the specified axis position.
     *
     * <p>Returns a zero-copy view with rank increased by 1.</p>
     *
     * @param axis position to insert the new dimension
     * @return reshaped zero-copy {@link NDArray} view with rank increased by 1
     * @throws IllegalArgumentException if {@code axis} is outside valid rank range {@code [0, dim()]}
     */
    public NDArray unsqueeze(int axis) {
        return LayoutOps.unsqueeze(this, axis);
    }

    // =========================================================================
    // Memory & Materialization
    // =========================================================================

    /**
     * Returns a tensor with the same logical elements reshaped into new dimensions.
     *
     * <p>If this tensor is already C-contiguous, a zero-copy view is returned. If the tensor is strided,
     * sliced, or transposed, data is automatically materialized into a newly allocated contiguous buffer.</p>
     *
     * @param newShape new dimensions; exactly one dimension may be {@code -1} for automatic inference
     * @return a reshaped {@link NDArray}
     * @throws IllegalArgumentException if the total element count does not match the product of {@code newShape}
     */
    public NDArray reshape(long... newShape) {
        return MemoryOps.reshape(this, newShape);
    }

    /**
     * Reshapes and casts this tensor to the specified {@link DType}.
     *
     * @param dType target data type descriptor
     * @param newShape new dimensions
     * @return a reshaped and cast {@link NDArray}
     * @throws IllegalArgumentException if the total element count does not match {@code newShape}
     */
    public NDArray reshape(DType dType, long... newShape) {
        return this.reshape(newShape).cast(dType);
    }

    /**
     * Returns true if elements are laid out contiguously in memory in standard C row-major order.
     *
     * <p>In a C-contiguous array, the stride of dimension {@code d} equals the product of subsequent
     * dimension sizes, allowing flat single-pointer SIMD vectorization.</p>
     *
     * @return true if C-contiguous, false if strided, sliced, or transposed
     */
    public boolean isContiguous() {
        return LayoutOps.isContiguous(this);
    }

    /**
     * Ensures this tensor is stored contiguously in memory, copying elements into a new auto-managed buffer
     * if strided or sliced.
     *
     * <p>If already contiguous, returns {@code this} without allocation.</p>
     *
     * @return a contiguous {@link NDArray}
     */
    public NDArray contiguous() {
        return contiguous(Arena.ofAuto());
    }

    /**
     * Ensures this tensor is stored contiguously in memory, allocated in the specified {@link Arena}.
     *
     * <p>If already contiguous, returns {@code this} without allocation.</p>
     *
     * @param arena memory arena governing the lifecycle of the materialized buffer
     * @return a contiguous {@link NDArray}
     */
    public NDArray contiguous(Arena arena) {
        return MemoryOps.contiguous(this, arena);
    }

    /**
     * Returns an independent deep copy of this tensor in a new contiguous memory segment.
     *
     * <p>Mutations to the returned tensor will not affect the original array.</p>
     *
     * @return a deep copy of this {@link NDArray}
     */
    public NDArray copy() {
        return MemoryOps.copy(this);
    }

    /**
     * Casts elements of this tensor to a target {@link DType}, returning a newly allocated array.
     *
     * @param target target data type descriptor
     * @return a newly allocated {@link NDArray} of data type {@code target}
     */
    public NDArray cast(DType target) {
        return MemoryOps.cast(this, target);
    }

    // =========================================================================
    // Mathematical Operations
    // =========================================================================

    // --- ARITHMETIC: ADD ---

    /**
     * Performs element-wise addition between this tensor and tensor {@code b}, supporting broadcasting.
     *
     * <p>Accelerated with hardware SIMD instructions via Java's Vector API. Allocates and returns
     * a newly materialized {@link NDArray}.</p>
     *
     * @param b the second tensor operand
     * @return a newly allocated {@link NDArray} containing the element-wise sum
     * @throws IllegalArgumentException if shapes cannot be aligned via broadcasting rules
     */
    public NDArray add(NDArray b) { return ArithmeticOps.add(this, b); }

    /**
     * Performs element-wise addition between this tensor and tensor {@code b}, writing results into
     * pre-allocated {@code resArray}.
     *
     * <p>Zero-allocation variant designed for high-performance loops and memory-constrained pipelines.</p>
     *
     * @param b the second tensor operand
     * @param resArray pre-allocated destination tensor to hold the results
     * @return {@code resArray} populated with the sum
     * @throws IllegalArgumentException if shapes cannot be aligned or {@code resArray} shape is invalid
     */
    public NDArray add(NDArray b, NDArray resArray) { return ArithmeticOps.add(this, b, resArray); }

    public NDArray add(NDArray b, Arena arena) { return ArithmeticOps.add(this, b, arena); }

    /**
     * Adds a scalar float to every element of this tensor, returning a new tensor.
     *
     * @param b scalar addend
     * @return a newly allocated {@link NDArray} containing the sum
     */
    public NDArray add(float b) { return ArithmeticOps.add(this, b); }

    /**
     * Adds a scalar integer to every element of this tensor, returning a new tensor.
     *
     * @param b scalar addend
     * @return a newly allocated {@link NDArray} containing the sum
     */
    public NDArray add(int b) { return ArithmeticOps.add(this, b); }

    /**
     * Adds a scalar double to every element of this tensor, returning a new tensor.
     *
     * @param b scalar addend
     * @return a newly allocated {@link NDArray} containing the sum
     */
    public NDArray add(double b) { return ArithmeticOps.add(this, b); }

    /**
     * Adds a scalar float to every element of this tensor, storing results in {@code resArray}.
     *
     * @param b scalar addend
     * @param resArray pre-allocated destination tensor
     * @return {@code resArray} populated with the sum
     */
    public NDArray add(float b, NDArray resArray) { return ArithmeticOps.add(this, b, resArray); }

    /**
     * Adds a scalar integer to every element of this tensor, storing results in {@code resArray}.
     *
     * @param b scalar addend
     * @param resArray pre-allocated destination tensor
     * @return {@code resArray} populated with the sum
     */
    public NDArray add(int b, NDArray resArray) { return ArithmeticOps.add(this, b, resArray); }

    /**
     * Adds a scalar double to every element of this tensor, storing results in {@code resArray}.
     *
     * @param b scalar addend
     * @param resArray pre-allocated destination tensor
     * @return {@code resArray} populated with the sum
     */
    public NDArray add(double b, NDArray resArray) { return ArithmeticOps.add(this, b, resArray); }

    // --- ARITHMETIC: SUBTRACT ---

    /**
     * Performs element-wise subtraction ({@code this - b}), supporting broadcasting.
     *
     * <p>Accelerated with hardware SIMD instructions via Java's Vector API. Allocates and returns
     * a newly materialized {@link NDArray}.</p>
     *
     * @param b the subtrahend tensor operand
     * @return a newly allocated {@link NDArray} containing the element-wise difference
     * @throws IllegalArgumentException if shapes cannot be aligned via broadcasting rules
     */
    public NDArray sub(NDArray b) { return ArithmeticOps.sub(this, b); }

    /**
     * Performs element-wise subtraction ({@code this - b}), writing results into pre-allocated {@code resArray}.
     *
     * @param b the subtrahend tensor operand
     * @param resArray pre-allocated destination tensor
     * @return {@code resArray} populated with the difference
     * @throws IllegalArgumentException if shapes cannot be aligned or {@code resArray} shape is invalid
     */
    public NDArray sub(NDArray b, NDArray resArray) { return ArithmeticOps.sub(this, b, resArray); }

    public NDArray sub(NDArray b, Arena arena) { return ArithmeticOps.sub(this, b, arena); }

    /**
     * Subtracts a scalar float from every element of this tensor, returning a new tensor.
     *
     * @param b scalar subtrahend
     * @return a newly allocated {@link NDArray} containing the difference
     */
    public NDArray sub(float b) { return ArithmeticOps.sub(this, b); }

    /**
     * Subtracts a scalar integer from every element of this tensor, returning a new tensor.
     *
     * @param b scalar subtrahend
     * @return a newly allocated {@link NDArray} containing the difference
     */
    public NDArray sub(int b) { return ArithmeticOps.sub(this, b); }

    /**
     * Subtracts a scalar double from every element of this tensor, returning a new tensor.
     *
     * @param b scalar subtrahend
     * @return a newly allocated {@link NDArray} containing the difference
     */
    public NDArray sub(double b) { return ArithmeticOps.sub(this, b); }

    /**
     * Subtracts a scalar float from every element of this tensor, storing results in {@code resArray}.
     *
     * @param b scalar subtrahend
     * @param resArray pre-allocated destination tensor
     * @return {@code resArray} populated with the difference
     */
    public NDArray sub(float b, NDArray resArray) { return ArithmeticOps.sub(this, b, resArray); }

    /**
     * Subtracts a scalar integer from every element of this tensor, storing results in {@code resArray}.
     *
     * @param b scalar subtrahend
     * @param resArray pre-allocated destination tensor
     * @return {@code resArray} populated with the difference
     */
    public NDArray sub(int b, NDArray resArray) { return ArithmeticOps.sub(this, b, resArray); }

    /**
     * Subtracts a scalar double from every element of this tensor, storing results in {@code resArray}.
     *
     * @param b scalar subtrahend
     * @param resArray pre-allocated destination tensor
     * @return {@code resArray} populated with the difference
     */
    public NDArray sub(double b, NDArray resArray) { return ArithmeticOps.sub(this, b, resArray); }

    // --- ARITHMETIC: MULTIPLY ---

    /**
     * Performs element-wise multiplication ({@code this * b}), supporting broadcasting (Hadamard product).
     *
     * <p>Accelerated with hardware SIMD instructions via Java's Vector API. Allocates and returns
     * a newly materialized {@link NDArray}.</p>
     *
     * @param b the multiplier tensor operand
     * @return a newly allocated {@link NDArray} containing the element-wise product
     * @throws IllegalArgumentException if shapes cannot be aligned via broadcasting rules
     */
    public NDArray mul(NDArray b) { return ArithmeticOps.mul(this, b); }

    /**
     * Performs element-wise multiplication ({@code this * b}), writing results into pre-allocated {@code resArray}.
     *
     * @param b the multiplier tensor operand
     * @param resArray pre-allocated destination tensor
     * @return {@code resArray} populated with the product
     * @throws IllegalArgumentException if shapes cannot be aligned or {@code resArray} shape is invalid
     */
    public NDArray mul(NDArray b, NDArray resArray) { return ArithmeticOps.mul(this, b, resArray); }

    public NDArray mul(NDArray b, Arena arena) { return ArithmeticOps.mul(this, b, arena); }

    /**
     * Multiplies every element of this tensor by a scalar float, returning a new tensor.
     *
     * @param b scalar multiplier
     * @return a newly allocated {@link NDArray} containing the product
     */
    public NDArray mul(float b) { return ArithmeticOps.mul(this, b); }

    /**
     * Multiplies every element of this tensor by a scalar integer, returning a new tensor.
     *
     * @param b scalar multiplier
     * @return a newly allocated {@link NDArray} containing the product
     */
    public NDArray mul(int b) { return ArithmeticOps.mul(this, b); }

    /**
     * Multiplies every element of this tensor by a scalar double, returning a new tensor.
     *
     * @param b scalar multiplier
     * @return a newly allocated {@link NDArray} containing the product
     */
    public NDArray mul(double b) { return ArithmeticOps.mul(this, b); }

    /**
     * Multiplies every element of this tensor by a scalar float, storing results in {@code resArray}.
     *
     * @param b scalar multiplier
     * @param resArray pre-allocated destination tensor
     * @return {@code resArray} populated with the product
     */
    public NDArray mul(float b, NDArray resArray) { return ArithmeticOps.mul(this, b, resArray); }

    /**
     * Multiplies every element of this tensor by a scalar integer, storing results in {@code resArray}.
     *
     * @param b scalar multiplier
     * @param resArray pre-allocated destination tensor
     * @return {@code resArray} populated with the product
     */
    public NDArray mul(int b, NDArray resArray) { return ArithmeticOps.mul(this, b, resArray); }

    /**
     * Multiplies every element of this tensor by a scalar double, storing results in {@code resArray}.
     *
     * @param b scalar multiplier
     * @param resArray pre-allocated destination tensor
     * @return {@code resArray} populated with the product
     */
    public NDArray mul(double b, NDArray resArray) { return ArithmeticOps.mul(this, b, resArray); }

    // --- ARITHMETIC: DIVIDE ---

    /**
     * Performs element-wise division ({@code this / b}), supporting broadcasting.
     *
     * <p>Accelerated with hardware SIMD instructions via Java's Vector API. Allocates and returns
     * a newly materialized {@link NDArray}.</p>
     *
     * @param b the divisor tensor operand
     * @return a newly allocated {@link NDArray} containing the element-wise quotient
     * @throws IllegalArgumentException if shapes cannot be aligned via broadcasting rules
     */
    public NDArray div(NDArray b) { return ArithmeticOps.div(this, b); }

    /**
     * Performs element-wise division ({@code this / b}), writing results into pre-allocated {@code resArray}.
     *
     * @param b the divisor tensor operand
     * @param resArray pre-allocated destination tensor
     * @return {@code resArray} populated with the quotient
     * @throws IllegalArgumentException if shapes cannot be aligned or {@code resArray} shape is invalid
     */
    public NDArray div(NDArray b, NDArray resArray) { return ArithmeticOps.div(this, b, resArray); }

    public NDArray div(NDArray b, Arena arena) { return ArithmeticOps.div(this, b, arena); }

    /**
     * Divides every element of this tensor by a scalar float, returning a new tensor.
     *
     * @param b scalar divisor
     * @return a newly allocated {@link NDArray} containing the quotient
     */
    public NDArray div(float b) { return ArithmeticOps.div(this, b); }

    /**
     * Divides every element of this tensor by a scalar integer, returning a new tensor.
     *
     * @param b scalar divisor
     * @return a newly allocated {@link NDArray} containing the quotient
     */
    public NDArray div(int b) { return ArithmeticOps.div(this, b); }

    /**
     * Divides every element of this tensor by a scalar double, returning a new tensor.
     *
     * @param b scalar divisor
     * @return a newly allocated {@link NDArray} containing the quotient
     */
    public NDArray div(double b) { return ArithmeticOps.div(this, b); }

    /**
     * Divides every element of this tensor by a scalar float, storing results in {@code resArray}.
     *
     * @param b scalar divisor
     * @param resArray pre-allocated destination tensor
     * @return {@code resArray} populated with the quotient
     */
    public NDArray div(float b, NDArray resArray) { return ArithmeticOps.div(this, b, resArray); }

    /**
     * Divides every element of this tensor by a scalar integer, storing results in {@code resArray}.
     *
     * @param b scalar divisor
     * @param resArray pre-allocated destination tensor
     * @return {@code resArray} populated with the quotient
     */
    public NDArray div(int b, NDArray resArray) { return ArithmeticOps.div(this, b, resArray); }

    /**
     * Divides every element of this tensor by a scalar double, storing results in {@code resArray}.
     *
     * @param b scalar divisor
     * @param resArray pre-allocated destination tensor
     * @return {@code resArray} populated with the quotient
     */
    public NDArray div(double b, NDArray resArray) { return ArithmeticOps.div(this, b, resArray); }

    // --- IN-PLACE ARITHMETIC ---

    /**
     * In-place element-wise addition ({@code this += b}), modifying this tensor directly without heap allocation.
     *
     * @param b the tensor operand to add
     * @return {@code this} tensor instance after in-place addition
     * @throws IllegalArgumentException if shapes are incompatible
     */
    public NDArray addi(NDArray b) { return this.add(b, this); }

    /**
     * In-place scalar addition ({@code this += b}), modifying this tensor directly without heap allocation.
     *
     * @param b scalar float to add
     * @return {@code this} tensor instance after mutation
     */
    public NDArray addi(float b) { return this.add(b, this); }

    /**
     * In-place scalar addition ({@code this += b}), modifying this tensor directly without heap allocation.
     *
     * @param b scalar int to add
     * @return {@code this} tensor instance after mutation
     */
    public NDArray addi(int b) { return this.add(b, this); }

    /**
     * In-place scalar addition ({@code this += b}), modifying this tensor directly without heap allocation.
     *
     * @param b scalar double to add
     * @return {@code this} tensor instance after mutation
     */
    public NDArray addi(double b) { return this.add(b, this); }

    /**
     * In-place element-wise subtraction ({@code this -= b}), modifying this tensor directly without heap allocation.
     *
     * @param b the tensor operand to subtract
     * @return {@code this} tensor instance after in-place subtraction
     * @throws IllegalArgumentException if shapes are incompatible
     */
    public NDArray subi(NDArray b) { return this.sub(b, this); }

    /**
     * In-place scalar subtraction ({@code this -= b}), modifying this tensor directly without heap allocation.
     *
     * @param b scalar float to subtract
     * @return {@code this} tensor instance after mutation
     */
    public NDArray subi(float b) { return this.sub(b, this); }

    /**
     * In-place scalar subtraction ({@code this -= b}), modifying this tensor directly without heap allocation.
     *
     * @param b scalar int to subtract
     * @return {@code this} tensor instance after mutation
     */
    public NDArray subi(int b) { return this.sub(b, this); }

    /**
     * In-place scalar subtraction ({@code this -= b}), modifying this tensor directly without heap allocation.
     *
     * @param b scalar double to subtract
     * @return {@code this} tensor instance after mutation
     */
    public NDArray subi(double b) { return this.sub(b, this); }

    /**
     * In-place element-wise multiplication ({@code this *= b}), modifying this tensor directly without heap allocation.
     *
     * @param b the tensor operand to multiply
     * @return {@code this} tensor instance after in-place multiplication
     * @throws IllegalArgumentException if shapes are incompatible
     */
    public NDArray muli(NDArray b) { return this.mul(b, this); }

    /**
     * In-place scalar multiplication ({@code this *= b}), modifying this tensor directly without heap allocation.
     *
     * @param b scalar float multiplier
     * @return {@code this} tensor instance after mutation
     */
    public NDArray muli(float b) { return this.mul(b, this); }

    /**
     * In-place scalar multiplication ({@code this *= b}), modifying this tensor directly without heap allocation.
     *
     * @param b scalar int multiplier
     * @return {@code this} tensor instance after mutation
     */
    public NDArray muli(int b) { return this.mul(b, this); }

    /**
     * In-place scalar multiplication ({@code this *= b}), modifying this tensor directly without heap allocation.
     *
     * @param b scalar double multiplier
     * @return {@code this} tensor instance after mutation
     */
    public NDArray muli(double b) { return this.mul(b, this); }

    /**
     * In-place element-wise division ({@code this /= b}), modifying this tensor directly without heap allocation.
     *
     * @param b the tensor operand to divide by
     * @return {@code this} tensor instance after in-place division
     * @throws IllegalArgumentException if shapes are incompatible
     */
    public NDArray divi(NDArray b) { return this.div(b, this); }

    /**
     * In-place scalar division ({@code this /= b}), modifying this tensor directly without heap allocation.
     *
     * @param b scalar float divisor
     * @return {@code this} tensor instance after mutation
     */
    public NDArray divi(float b) { return this.div(b, this); }

    /**
     * In-place scalar division ({@code this /= b}), modifying this tensor directly without heap allocation.
     *
     * @param b scalar int divisor
     * @return {@code this} tensor instance after mutation
     */
    public NDArray divi(int b) { return this.div(b, this); }

    /**
     * In-place scalar division ({@code this /= b}), modifying this tensor directly without heap allocation.
     *
     * @param b scalar double divisor
     * @return {@code this} tensor instance after mutation
     */
    public NDArray divi(double b) { return this.div(b, this); }

    // --- KOTLIN OPERATOR ALIASES ---

    /**
     * Operator overload alias for {@code +} in Kotlin, delegating to {@link #add(NDArray)}.
     *
     * @param b tensor operand to add
     * @return newly allocated sum tensor
     */
    public NDArray plus(NDArray b) { return this.add(b); }

    /**
     * Operator overload alias for {@code +} with double scalar in Kotlin, delegating to {@link #add(double)}.
     *
     * @param b scalar double to add
     * @return newly allocated sum tensor
     */
    public NDArray plus(double b) { return this.add(b); }

    /**
     * Operator overload alias for {@code +} with float scalar in Kotlin, delegating to {@link #add(float)}.
     *
     * @param b scalar float to add
     * @return newly allocated sum tensor
     */
    public NDArray plus(float b) { return this.add(b); }

    /**
     * Operator overload alias for {@code +} with int scalar in Kotlin, delegating to {@link #add(int)}.
     *
     * @param b scalar int to add
     * @return newly allocated sum tensor
     */
    public NDArray plus(int b) { return this.add(b); }

    /**
     * Operator overload alias for {@code -} in Kotlin, delegating to {@link #sub(NDArray)}.
     *
     * @param b tensor operand to subtract
     * @return newly allocated difference tensor
     */
    public NDArray minus(NDArray b) { return this.sub(b); }

    /**
     * Operator overload alias for {@code -} with double scalar in Kotlin, delegating to {@link #sub(double)}.
     *
     * @param b scalar double to subtract
     * @return newly allocated difference tensor
     */
    public NDArray minus(double b) { return this.sub(b); }

    /**
     * Operator overload alias for {@code -} with float scalar in Kotlin, delegating to {@link #sub(float)}.
     *
     * @param b scalar float to subtract
     * @return newly allocated difference tensor
     */
    public NDArray minus(float b) { return this.sub(b); }

    /**
     * Operator overload alias for {@code -} with int scalar in Kotlin, delegating to {@link #sub(int)}.
     *
     * @param b scalar int to subtract
     * @return newly allocated difference tensor
     */
    public NDArray minus(int b) { return this.sub(b); }

    /**
     * Operator overload alias for {@code *} in Kotlin, delegating to {@link #mul(NDArray)}.
     *
     * @param b tensor operand to multiply
     * @return newly allocated product tensor
     */
    public NDArray times(NDArray b) { return this.mul(b); }

    /**
     * Operator overload alias for {@code *} with double scalar in Kotlin, delegating to {@link #mul(double)}.
     *
     * @param b scalar double multiplier
     * @return newly allocated product tensor
     */
    public NDArray times(double b) { return this.mul(b); }

    /**
     * Operator overload alias for {@code *} with float scalar in Kotlin, delegating to {@link #mul(float)}.
     *
     * @param b scalar float multiplier
     * @return newly allocated product tensor
     */
    public NDArray times(float b) { return this.mul(b); }

    /**
     * Operator overload alias for {@code *} with int scalar in Kotlin, delegating to {@link #mul(int)}.
     *
     * @param b scalar int multiplier
     * @return newly allocated product tensor
     */
    public NDArray times(int b) { return this.mul(b); }

    // --- COMPARISONS ---

    /**
     * Computes the element-wise maximum between this tensor and tensor {@code b}.
     *
     * @param b tensor operand
     * @return a newly allocated {@link NDArray} containing the element-wise maximums
     */
    public NDArray maximum(NDArray b) { return CompareOps.maximum(this, b); }

    /**
     * Computes the element-wise maximum between this tensor and a float scalar.
     *
     * @param b float scalar
     * @return a newly allocated {@link NDArray} containing the element-wise maximums
     */
    public NDArray maximum(float b) { return CompareOps.maximum(this, b); }

    /**
     * Computes the element-wise maximum between this tensor and an int scalar.
     *
     * @param b int scalar
     * @return a newly allocated {@link NDArray} containing the element-wise maximums
     */
    public NDArray maximum(int b) { return CompareOps.maximum(this, b); }

    /**
     * Computes the element-wise maximum between this tensor and a double scalar.
     *
     * @param b double scalar
     * @return a newly allocated {@link NDArray} containing the element-wise maximums
     */
    public NDArray maximum(double b) { return CompareOps.maximum(this, b); }

    /**
     * Computes the element-wise minimum between this tensor and tensor {@code b}.
     *
     * @param b tensor operand
     * @return a newly allocated {@link NDArray} containing the element-wise minimums
     */
    public NDArray minimum(NDArray b) { return CompareOps.minimum(this, b); }

    /**
     * Computes the element-wise minimum between this tensor and a float scalar.
     *
     * @param b float scalar
     * @return a newly allocated {@link NDArray} containing the element-wise minimums
     */
    public NDArray minimum(float b) { return CompareOps.minimum(this, b); }

    /**
     * Computes the element-wise minimum between this tensor and an int scalar.
     *
     * @param b int scalar
     * @return a newly allocated {@link NDArray} containing the element-wise minimums
     */
    public NDArray minimum(int b) { return CompareOps.minimum(this, b); }

    /**
     * Computes the element-wise minimum between this tensor and a double scalar.
     *
     * @param b double scalar
     * @return a newly allocated {@link NDArray} containing the element-wise minimums
     */
    public NDArray minimum(double b) { return CompareOps.minimum(this, b); }

    // --- REDUCTIONS ---

    /**
     * Finds the global maximum value across all elements in this tensor, promoted to {@code double}.
     *
     * <p>Accelerated with SIMD vector max-reduction lanes when contiguous.</p>
     *
     * @return global maximum scalar value
     */
    public double max() { return ReduceOps.max(this); }

    /**
     * Finds the global minimum value across all elements in this tensor, promoted to {@code double}.
     *
     * <p>Accelerated with SIMD vector min-reduction lanes when contiguous.</p>
     *
     * @return global minimum scalar value
     */
    public double min() { return ReduceOps.min(this); }

    /**
     * Computes the global sum across all elements in this tensor, promoted to {@code double}.
     *
     * <p>Accelerated with SIMD vector sum-reduction lanes when contiguous.</p>
     *
     * @return global sum scalar value
     */
    public double sum() { return ReduceOps.sum(this); }

    /**
     * Computes the 1D vector dot product (inner product) between this tensor and tensor {@code b}.
     *
     * @param b the second 1D vector operand
     * @return scalar dot product value
     * @throws IllegalArgumentException if either tensor is not 1D or their lengths differ
     */
    public double dot(NDArray b) { return ReduceOps.dot(this, b); }

    /**
     * Computes the sum of elements along the specified {@code axis}, collapsing that dimension.
     *
     * @param axis dimension along which to sum; supports negative indexing (e.g. {@code -1} for the last axis)
     * @return a newly allocated {@link NDArray} with rank reduced by 1
     * @throws IllegalArgumentException if {@code axis} is outside {@code [-dim(), dim() - 1]}
     */
    public NDArray sum(int axis) { return ReduceOps.sum(this, axis); }

    /**
     * Finds the maximum values along the specified {@code axis}, collapsing that dimension.
     *
     * @param axis dimension along which to find maximums; supports negative indexing
     * @return a newly allocated {@link NDArray} with rank reduced by 1
     * @throws IllegalArgumentException if {@code axis} is outside {@code [-dim(), dim() - 1]}
     */
    public NDArray max(int axis) { return ReduceOps.max(this, axis); }

    /**
     * Finds the minimum values along the specified {@code axis}, collapsing that dimension.
     *
     * @param axis dimension along which to find minimums; supports negative indexing
     * @return a newly allocated {@link NDArray} with rank reduced by 1
     * @throws IllegalArgumentException if {@code axis} is outside {@code [-dim(), dim() - 1]}
     */
    public NDArray min(int axis) { return ReduceOps.min(this, axis); }

    /**
     * Computes the sum of elements along the specified {@code axis}, optionally retaining the reduced
     * dimension as size 1.
     *
     * @param axis dimension along which to sum; supports negative indexing
     * @param keepDims whether to retain the reduced dimension with size 1
     * @return reduced {@link NDArray}
     * @throws IllegalArgumentException if {@code axis} is outside {@code [-dim(), dim() - 1]}
     */
    public NDArray sum(int axis, boolean keepDims) {
        NDArray res = this.sum(axis);
        if (keepDims) {
            int targetAxis = axis < 0 ? axis + this.dim() : axis;
            return res.unsqueeze(targetAxis);
        }
        return res;
    }

    /**
     * Finds the maximum values along the specified {@code axis}, optionally retaining the reduced
     * dimension as size 1.
     *
     * @param axis dimension along which to find maximums; supports negative indexing
     * @param keepDims whether to retain the reduced dimension with size 1
     * @return reduced {@link NDArray}
     * @throws IllegalArgumentException if {@code axis} is outside {@code [-dim(), dim() - 1]}
     */
    public NDArray max(int axis, boolean keepDims) {
        NDArray res = this.max(axis);
        if (keepDims) {
            int targetAxis = axis < 0 ? axis + this.dim() : axis;
            return res.unsqueeze(targetAxis);
        }
        return res;
    }

    /**
     * Finds the minimum values along the specified {@code axis}, optionally retaining the reduced
     * dimension as size 1.
     *
     * @param axis dimension along which to find minimums; supports negative indexing
     * @param keepDims whether to retain the reduced dimension with size 1
     * @return reduced {@link NDArray}
     * @throws IllegalArgumentException if {@code axis} is outside {@code [-dim(), dim() - 1]}
     */
    public NDArray min(int axis, boolean keepDims) {
        NDArray res = this.min(axis);
        if (keepDims) {
            int targetAxis = axis < 0 ? axis + this.dim() : axis;
            return res.unsqueeze(targetAxis);
        }
        return res;
    }

    // --- UNARY MATH ---

    /**
     * Computes the element-wise square root of this tensor.
     *
     * @return a newly allocated {@link NDArray} containing the square root of each element
     */
    public NDArray sqrt() { return UnaryOps.sqrt(this); }
    public NDArray sqrt(NDArray res) { return UnaryOps.sqrt(this, res); }
    public NDArray sqrt(Arena arena) { return UnaryOps.sqrt(this, arena); }

    /**
     * Computes the element-wise absolute value of this tensor.
     *
     * @return a newly allocated {@link NDArray} containing the absolute values
     */
    public NDArray abs() { return UnaryOps.abs(this); }
    public NDArray abs(NDArray res) { return UnaryOps.abs(this, res); }
    public NDArray abs(Arena arena) { return UnaryOps.abs(this, arena); }

    /**
     * Computes the element-wise natural exponential ({@code e^x}) of this tensor.
     *
     * @return a newly allocated {@link NDArray} containing the exponential values
     */
    public NDArray exp() { return UnaryOps.exp(this); }
    public NDArray exp(NDArray res) { return UnaryOps.exp(this, res); }
    public NDArray exp(Arena arena) { return UnaryOps.exp(this, arena); }

    /**
     * Computes the element-wise natural logarithm ({@code ln(x)}) of this tensor.
     *
     * @return a newly allocated {@link NDArray} containing the natural logarithms
     */
    public NDArray log() { return UnaryOps.log(this); }
    public NDArray log(NDArray res) { return UnaryOps.log(this, res); }
    public NDArray log(Arena arena) { return UnaryOps.log(this, arena); }

    /**
     * Computes the element-wise base-10 logarithm ({@code log10(x)}) of this tensor.
     *
     * @return a newly allocated {@link NDArray} containing the base-10 logarithms
     */
    public NDArray log10() { return UnaryOps.log10(this); }
    public NDArray log10(NDArray res) { return UnaryOps.log10(this, res); }
    public NDArray log10(Arena arena) { return UnaryOps.log10(this, arena); }

    /**
     * Applies the logistic sigmoid activation function {@code 1 / (1 + e^-x)} element-wise.
     *
     * @return a newly allocated {@link NDArray} containing the sigmoid activations
     */
    public NDArray sigmoid() { return UnaryOps.sigmoid(this); }
    public NDArray sigmoid(NDArray res) { return UnaryOps.sigmoid(this, res); }
    public NDArray sigmoid(Arena arena) { return UnaryOps.sigmoid(this, arena); }

    // --- TRIGONOMETRY ---

    /**
     * Computes the element-wise trigonometric sine of this tensor.
     *
     * @return a newly allocated {@link NDArray} containing sine values
     */
    public NDArray sin() { return TrigOps.sin(this); }
    public NDArray sin(NDArray res) { return TrigOps.sin(this, res); }
    public NDArray sin(Arena arena) { return TrigOps.sin(this, arena); }

    /**
     * Computes the element-wise trigonometric cosine of this tensor.
     *
     * @return a newly allocated {@link NDArray} containing cosine values
     */
    public NDArray cos() { return TrigOps.cos(this); }
    public NDArray cos(NDArray res) { return TrigOps.cos(this, res); }
    public NDArray cos(Arena arena) { return TrigOps.cos(this, arena); }

    /**
     * Computes the element-wise trigonometric tangent of this tensor.
     *
     * @return a newly allocated {@link NDArray} containing tangent values
     */
    public NDArray tan() { return TrigOps.tan(this); }
    public NDArray tan(NDArray res) { return TrigOps.tan(this, res); }
    public NDArray tan(Arena arena) { return TrigOps.tan(this, arena); }

    /**
     * Computes the element-wise hyperbolic sine of this tensor.
     *
     * @return a newly allocated {@link NDArray} containing hyperbolic sine values
     */
    public NDArray sinh() { return TrigOps.sinh(this); }
    public NDArray sinh(NDArray res) { return TrigOps.sinh(this, res); }
    public NDArray sinh(Arena arena) { return TrigOps.sinh(this, arena); }

    /**
     * Computes the element-wise hyperbolic cosine of this tensor.
     *
     * @return a newly allocated {@link NDArray} containing hyperbolic cosine values
     */
    public NDArray cosh() { return TrigOps.cosh(this); }
    public NDArray cosh(NDArray res) { return TrigOps.cosh(this, res); }
    public NDArray cosh(Arena arena) { return TrigOps.cosh(this, arena); }

    /**
     * Computes the element-wise hyperbolic tangent of this tensor.
     *
     * @return a newly allocated {@link NDArray} containing hyperbolic tangent values
     */
    public NDArray tanh() { return TrigOps.tanh(this); }
    public NDArray tanh(NDArray res) { return TrigOps.tanh(this, res); }
    public NDArray tanh(Arena arena) { return TrigOps.tanh(this, arena); }

    // --- LINEAR ALGEBRA ---

    /**
     * Performs matrix multiplication ({@code this @ b}) or batched matrix multiplication.
     *
     * <p>Hardware-accelerated using tile-based SIMD vector kernels. Adheres to standard NumPy
     * batched matmul rules where leading dimensions are broadcasted.</p>
     *
     * @param b the right-hand matrix or batched tensor operand
     * @return a newly allocated {@link NDArray} containing the matrix product
     * @throws IllegalArgumentException if inner matrix dimensions do not match or ranks are invalid
     */
    public NDArray matmul(NDArray b) { return LinalgOps.matmul(this, b); }

    /**
     * Performs matrix multiplication, writing results into pre-allocated {@code resArray}.
     *
     * <p>Zero-allocation variant for high-throughput linear algebra loops.</p>
     *
     * @param b the right-hand matrix or batched tensor operand
     * @param resArray pre-allocated destination tensor to hold the product
     * @return {@code resArray} populated with the matrix product
     * @throws IllegalArgumentException if dimensions are incompatible
     */
    public NDArray matmul(NDArray b, NDArray resArray) { return LinalgOps.matmul(this, b, resArray); }

    /**
     * Computes the sum along the main diagonal of a 2D square matrix.
     *
     * @return scalar trace value
     * @throws IllegalArgumentException if tensor is not 2D
     */
    public double trace() { return LinalgOps.trace(this); }

    /**
     * Computes the sum along a diagonal offset from the main diagonal of a 2D matrix.
     *
     * @param offset diagonal offset (positive above main diagonal, negative below)
     * @return scalar trace value
     * @throws IllegalArgumentException if tensor is not 2D
     */
    public double trace(int offset) { return LinalgOps.trace(this, offset); }

    /**
     * Computes the matrix Frobenius norm or vector Euclidean 2-norm ({@code sqrt(sum(x_i^2))}).
     *
     * @return scalar norm value
     */
    public double norm() { return LinalgOps.norm(this); }

    /**
     * Computes the matrix or vector p-norm according to order {@code ord}.
     *
     * @param ord norm order (e.g. 1 for L1-norm, 2 for L2-norm, {@link Integer#MAX_VALUE} for inf-norm)
     * @return scalar norm value
     */
    public double norm(int ord) { return LinalgOps.norm(this, ord); }

    /**
     * Computes the determinant of a square 2D matrix via LU decomposition with partial pivoting.
     *
     * @return scalar determinant value
     * @throws IllegalArgumentException if tensor is not a 2D square matrix
     */
    public double det() { return LinalgOps.det(this); }

    /**
     * Computes the sign and natural logarithm of the determinant of a square 2D matrix.
     *
     * <p>Provides numerical stability against overflow or underflow for large matrices.</p>
     *
     * @return {@link SlogdetResult} containing the sign ({@code -1, 0, +1}) and log-absolute-determinant
     * @throws IllegalArgumentException if tensor is not a 2D square matrix
     */
    public SlogdetResult slogdet() { return LinalgOps.slogdet(this); }

    /**
     * Computes the multiplicative inverse of a square non-singular matrix ({@code A^-1}).
     *
     * @return a newly allocated {@link NDArray} representing the matrix inverse
     * @throws ArithmeticException if the matrix is singular (non-invertible)
     * @throws IllegalArgumentException if tensor is not a 2D square matrix
     */
    public NDArray inv() { return LinalgOps.inv(this); }

    /**
     * Computes the multiplicative inverse of a square non-singular matrix, allocating inside {@code arena}.
     *
     * @param arena memory arena governing lifecycle of the result
     * @return a newly allocated {@link NDArray} representing the matrix inverse
     * @throws ArithmeticException if the matrix is singular (non-invertible)
     * @throws IllegalArgumentException if tensor is not a 2D square matrix
     */
    public NDArray inv(Arena arena) { return LinalgOps.inv(this, arena); }

    /**
     * Solves the linear matrix equation {@code A * X = B} for {@code X}.
     *
     * @param b right-hand side vector or matrix
     * @return solution matrix or vector {@code X}
     * @throws ArithmeticException if {@code this} matrix is singular
     * @throws IllegalArgumentException if dimensions do not match
     */
    public NDArray solve(NDArray b) { return LinalgOps.solve(this, b); }

    /**
     * Solves the linear matrix equation {@code A * X = B} for {@code X}, allocating inside {@code arena}.
     *
     * @param b right-hand side vector or matrix
     * @param arena memory arena governing lifecycle of the result
     * @return solution matrix or vector {@code X}
     * @throws ArithmeticException if {@code this} matrix is singular
     * @throws IllegalArgumentException if dimensions do not match
     */
    public NDArray solve(NDArray b, Arena arena) { return LinalgOps.solve(this, b, arena); }

    /**
     * Computes the Cholesky decomposition of a symmetric/Hermitian positive-definite matrix ({@code A = L * L^T}).
     *
     * @return lower-triangular matrix factor {@code L}
     * @throws ArithmeticException if matrix is not positive-definite
     * @throws IllegalArgumentException if tensor is not 2D square
     */
    public NDArray cholesky() { return LinalgOps.cholesky(this); }

    /**
     * Computes the Cholesky decomposition, allocating inside {@code arena}.
     *
     * @param arena memory arena governing lifecycle of the result
     * @return lower-triangular matrix factor {@code L}
     * @throws ArithmeticException if matrix is not positive-definite
     * @throws IllegalArgumentException if tensor is not 2D square
     */
    public NDArray cholesky(Arena arena) { return LinalgOps.cholesky(this, arena); }

    /**
     * Computes the QR decomposition ({@code A = Q * R}) of this matrix.
     *
     * @return {@link QRResult} containing orthogonal matrix {@code Q} and upper-triangular matrix {@code R}
     * @throws IllegalArgumentException if tensor is not 2D
     */
    public QRResult qr() { return LinalgOps.qr(this); }

    /**
     * Computes the QR decomposition, allocating inside {@code arena}.
     *
     * @param arena memory arena governing lifecycle of the decomposition results
     * @return {@link QRResult} containing orthogonal matrix {@code Q} and upper-triangular matrix {@code R}
     * @throws IllegalArgumentException if tensor is not 2D
     */
    public QRResult qr(Arena arena) { return LinalgOps.qr(this, arena); }

    /**
     * Computes eigenvalues and eigenvectors for a real symmetric (or Hermitian) matrix.
     *
     * @return {@link EighResult} containing real eigenvalues and orthogonal eigenvectors
     * @throws IllegalArgumentException if tensor is not 2D square
     */
    public EighResult eigh() { return LinalgOps.eigh(this); }

    /**
     * Computes eigenvalues and eigenvectors for a symmetric matrix, allocating inside {@code arena}.
     *
     * @param arena memory arena governing lifecycle of the results
     * @return {@link EighResult} containing real eigenvalues and orthogonal eigenvectors
     * @throws IllegalArgumentException if tensor is not 2D square
     */
    public EighResult eigh(Arena arena) { return LinalgOps.eigh(this, arena); }

    /**
     * Computes the Singular Value Decomposition ({@code A = U * S * V^T}) of this matrix.
     *
     * @return {@link SVDResult} containing left singular vectors {@code U}, singular values {@code S}, and right vectors {@code V^T}
     * @throws IllegalArgumentException if tensor is not 2D
     */
    public SVDResult svd() { return LinalgOps.svd(this); }

    /**
     * Computes the Singular Value Decomposition, allocating inside {@code arena}.
     *
     * @param arena memory arena governing lifecycle of the decomposition results
     * @return {@link SVDResult} containing left singular vectors {@code U}, singular values {@code S}, and right vectors {@code V^T}
     * @throws IllegalArgumentException if tensor is not 2D
     */
    public SVDResult svd(Arena arena) { return LinalgOps.svd(this, arena); }

    /**
     * Computes the numerical rank of this matrix using SVD with default tolerance.
     *
     * @return integer matrix rank
     * @throws IllegalArgumentException if tensor is not 2D
     */
    public int matrixRank() { return LinalgOps.matrixRank(this); }

    /**
     * Computes the numerical rank of this matrix using SVD with specified tolerance {@code tol}.
     *
     * @param tol singular value threshold below which values are treated as zero
     * @return integer matrix rank
     * @throws IllegalArgumentException if tensor is not 2D
     */
    public int matrixRank(double tol) { return LinalgOps.matrixRank(this, tol); }

    /**
     * Computes the numerical rank of this matrix with specified tolerance and memory {@code arena}.
     *
     * @param tol singular value threshold below which values are treated as zero
     * @param arena memory arena for intermediate decomposition buffers
     * @return integer matrix rank
     * @throws IllegalArgumentException if tensor is not 2D
     */
    public int matrixRank(double tol, Arena arena) { return LinalgOps.matrixRank(this, tol, arena); }

    /**
     * Computes the 2-norm condition number of this matrix ({@code sigma_max / sigma_min}).
     *
     * @return condition number
     * @throws IllegalArgumentException if tensor is not 2D
     */
    public double cond() { return LinalgOps.cond(this); }

    /**
     * Computes the 2-norm condition number of this matrix, allocating inside {@code arena}.
     *
     * @param arena memory arena for intermediate buffers
     * @return condition number
     * @throws IllegalArgumentException if tensor is not 2D
     */
    public double cond(Arena arena) { return LinalgOps.cond(this, arena); }

    /**
     * Computes the Moore-Penrose pseudoinverse ({@code A^+}) of this matrix via SVD.
     *
     * @return a newly allocated {@link NDArray} representing the pseudoinverse
     * @throws IllegalArgumentException if tensor is not 2D
     */
    public NDArray pinv() { return LinalgOps.pinv(this); }

    /**
     * Computes the Moore-Penrose pseudoinverse with cutoff ratio {@code rcond}.
     *
     * @param rcond singular value cutoff threshold ratio
     * @return a newly allocated {@link NDArray} representing the pseudoinverse
     * @throws IllegalArgumentException if tensor is not 2D
     */
    public NDArray pinv(double rcond) { return LinalgOps.pinv(this, rcond); }

    /**
     * Computes the Moore-Penrose pseudoinverse with cutoff ratio {@code rcond}, allocating inside {@code arena}.
     *
     * @param rcond singular value cutoff threshold ratio
     * @param arena memory arena governing lifecycle of result
     * @return a newly allocated {@link NDArray} representing the pseudoinverse
     * @throws IllegalArgumentException if tensor is not 2D
     */
    public NDArray pinv(double rcond, Arena arena) { return LinalgOps.pinv(this, rcond, arena); }

    /**
     * Computes eigenvalues and eigenvectors for a general square matrix.
     *
     * @return {@link EigResult} containing eigenvalues and eigenvectors
     * @throws IllegalArgumentException if tensor is not 2D square
     */
    public EigResult eig() { return LinalgOps.eig(this); }

    /**
     * Computes eigenvalues and eigenvectors for a general square matrix, allocating inside {@code arena}.
     *
     * @param arena memory arena governing lifecycle of result
     * @return {@link EigResult} containing eigenvalues and eigenvectors
     * @throws IllegalArgumentException if tensor is not 2D square
     */
    public EigResult eig(Arena arena) { return LinalgOps.eig(this, arena); }

    /**
     * Computes the cosine similarity between this 1D vector and vector {@code b}.
     *
     * @param b the second 1D vector operand
     * @return scalar cosine similarity value in range {@code [-1.0, 1.0]}
     * @throws IllegalArgumentException if either tensor is not 1D or lengths differ
     */
    public double cosineSimilarity(NDArray b) { return LinalgOps.cosineSimilarity(this, b); }

    // --- BOOLEAN LOGIC ---

    /**
     * Performs element-wise logical AND ({@code this & b}) on boolean tensors.
     *
     * @param b the second boolean tensor operand
     * @return a newly allocated boolean {@link NDArray}
     * @throws IllegalArgumentException if shapes cannot be aligned
     */
    public NDArray and(NDArray b) { return BooleanOps.and(this, b); }

    /**
     * Performs element-wise logical AND on boolean tensors, storing results in {@code resArray}.
     *
     * @param b the second boolean tensor operand
     * @param resArray destination buffer
     * @return {@code resArray} populated with logical AND results
     * @throws IllegalArgumentException if shapes cannot be aligned
     */
    public NDArray and(NDArray b, NDArray resArray) { return BooleanOps.and(this, b, resArray); }

    /**
     * Performs element-wise logical OR ({@code this | b}) on boolean tensors.
     *
     * @param b the second boolean tensor operand
     * @return a newly allocated boolean {@link NDArray}
     * @throws IllegalArgumentException if shapes cannot be aligned
     */
    public NDArray or(NDArray b) { return BooleanOps.or(this, b); }

    /**
     * Performs element-wise logical OR on boolean tensors, storing results in {@code resArray}.
     *
     * @param b the second boolean tensor operand
     * @param resArray destination buffer
     * @return {@code resArray} populated with logical OR results
     * @throws IllegalArgumentException if shapes cannot be aligned
     */
    public NDArray or(NDArray b, NDArray resArray) { return BooleanOps.or(this, b, resArray); }

    /**
     * Performs element-wise logical XOR ({@code this ^ b}) on boolean tensors.
     *
     * @param b the second boolean tensor operand
     * @return a newly allocated boolean {@link NDArray}
     * @throws IllegalArgumentException if shapes cannot be aligned
     */
    public NDArray xor(NDArray b) { return BooleanOps.xor(this, b); }

    /**
     * Performs element-wise logical XOR on boolean tensors, storing results in {@code resArray}.
     *
     * @param b the second boolean tensor operand
     * @param resArray destination buffer
     * @return {@code resArray} populated with logical XOR results
     * @throws IllegalArgumentException if shapes cannot be aligned
     */
    public NDArray xor(NDArray b, NDArray resArray) { return BooleanOps.xor(this, b, resArray); }

    /**
     * Performs element-wise logical NOT ({@code !this}) on a boolean tensor.
     *
     * @return a newly allocated inverted boolean {@link NDArray}
     */
    public NDArray not() { return BooleanOps.not(this); }

    /**
     * Performs element-wise logical NOT on a boolean tensor, storing results in {@code resArray}.
     *
     * @param resArray destination buffer
     * @return {@code resArray} populated with logical NOT results
     */
    public NDArray not(NDArray resArray) { return BooleanOps.not(this, resArray); }

    /**
     * Evaluates whether at least one element in this tensor is truthy (non-zero).
     *
     * @return true if any element is non-zero, false otherwise
     */
    public boolean any() { return BooleanOps.any(this); }

    /**
     * Evaluates whether all elements in this tensor are truthy (non-zero).
     *
     * @return true if all elements are non-zero, false otherwise
     */
    public boolean all() { return BooleanOps.all(this); }

    // =========================================================================
    // Object Class Overrides
    // =========================================================================

    /**
     * Returns a human-readable string representation of this tensor summarizing dimensions and elements.
     *
     * @return formatted string describing tensor shape and values
     */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("NDArray" + shapeString() + " [");
        int maxPrint = 6;
        NDIter iter = new NDIter(this.internalShapeUnsafe());
        for (long i = 0; i < getSize(); i++) {
            if (i == maxPrint / 2 && getSize() > maxPrint) {
                sb.append("..., ");
            } else if (getSize() <= maxPrint || i < maxPrint / 2 || i >= getSize() - (maxPrint / 2)) {
                long byteOffset = ShapeUtil.getByteOffset(iter.coords, this.internalStridesUnsafe(), this.getDType());
                switch (this.getDType()) {
                    case i32 -> sb.append(getData().get(ValueLayout.JAVA_INT, byteOffset));
                    case f32 -> sb.append(getData().get(ValueLayout.JAVA_FLOAT, byteOffset));
                    case f64 -> sb.append(getData().get(ValueLayout.JAVA_DOUBLE, byteOffset));
                    case bool -> sb.append(getData().get(ValueLayout.JAVA_BYTE, byteOffset) != 0 ? "true" : "false");
                    default -> throw new UnsupportedOperationException("This dtype " + this.getDType() + " doesn't support this method");
                }
                if (i < getSize() - 1) sb.append(", ");
            }
            iter.next();
        }
        return sb.append("]").toString();
    }

    /**
     * Compares this tensor with another object for numerical equality.
     *
     * <p>If both tensors are C-contiguous with identical shape and dtype, equality is evaluated
     * via hardware-accelerated memory byte segment comparison ({@link MemorySegment#mismatch(MemorySegment)}).
     * Otherwise, elements are compared element-by-element across their physical strided offsets.</p>
     *
     * @param o the reference object with which to compare
     * @return true if both objects are tensors with matching dtype, shape, and element values
     */
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

            boolean match = switch (this.getDType()) {
                case f32 -> this.getData().getAtIndex(ValueLayout.JAVA_FLOAT, offsetThis) == other.getData().getAtIndex(ValueLayout.JAVA_FLOAT, offsetOther);
                case f64 -> this.getData().getAtIndex(ValueLayout.JAVA_DOUBLE, offsetThis) == other.getData().getAtIndex(ValueLayout.JAVA_DOUBLE, offsetOther);
                case i32 -> this.getData().getAtIndex(ValueLayout.JAVA_INT, offsetThis) == other.getData().getAtIndex(ValueLayout.JAVA_INT, offsetOther);
                case bool -> this.getData().getAtIndex(ValueLayout.JAVA_BYTE, offsetThis) == other.getData().getAtIndex(ValueLayout.JAVA_BYTE, offsetOther);
                default -> throw new UnsupportedOperationException("This dtype " + this.getDType() + " doesn't support this method");
            };
            if (!match) return false;
        }
        return true;
    }

    /**
     * Returns a hash code value for this tensor based on shape, dtype, and leading element values.
     *
     * @return an integer hash code value
     */
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
                default -> throw new UnsupportedOperationException("This dtype " + getDType() + " doesn't support this method");
            };
            result = 31 * result + valHash;
        }
        return result;
    }

    private static long getPhysicalOffset(long logicalIndex, long[] shape, long[] strides) {
        return GetOps.getPhysicalOffset(logicalIndex, shape, strides);
    }
}
