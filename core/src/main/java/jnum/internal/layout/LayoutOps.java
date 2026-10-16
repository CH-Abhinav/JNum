package jnum.internal.layout;

import jnum.NDArray;
import jnum.Slice;

import java.lang.foreign.MemorySegment;
import java.util.Arrays;

/**
 * Internal operations for tensor layouts, strided transformations, and zero-copy views.
 */
public final class LayoutOps {

    private LayoutOps() {
        throw new AssertionError("LayoutOps utility class cannot be instantiated.");
    }

    /**
     * Checks if the tensor is contiguous in standard C-order (row-major).
     */
    public static boolean isContiguous(NDArray a) {
        long[] shape = a.internalShapeUnsafe();
        long[] strides = a.internalStridesUnsafe();
        long expStride = 1L;
        for (int i = shape.length - 1; i >= 0; i--) {
            if (strides[i] != expStride) return false;
            expStride *= shape[i];
        }
        return true;
    }

    /**
     * Reverses the axes of a tensor, creating a zero-copy transposed view.
     */
    public static NDArray transpose(NDArray a) {
        if (a.dim() < 2) return a;
        long[] shape = a.internalShapeUnsafe();
        long[] strides = a.internalStridesUnsafe();
        int len = shape.length;
        long[] newShape = new long[len];
        long[] newStrides = new long[len];
        for (int i = 0; i < len; i++) {
            newShape[i] = shape[len - 1 - i];
            newStrides[i] = strides[len - 1 - i];
        }
        return NDArray.ofRaw(a.getData(), newShape, newStrides, a.getDType());
    }

    /**
     * Broadcasts a tensor to a larger compatible shape by setting stride to 0 on expanded dimensions.
     */
    public static NDArray broadcastTo(NDArray a, long... shape) {
        if (Arrays.equals(a.internalShapeUnsafe(), shape)) return a;
        int ndim = shape.length;
        if (ndim < a.dim()) {
            throw new IllegalArgumentException(
                    "Cannot broadcast array of shape " + Arrays.toString(a.internalShapeUnsafe()) +
                            " to target shape " + Arrays.toString(shape) +
                            " because the target has fewer dimensions."
            );
        }
        long[] newStrides = new long[ndim];
        long[] paddedShape = new long[ndim];
        long[] paddedStrides = new long[ndim];
        int offset = ndim - a.dim();
        for (int i = 0; i < ndim; i++) {
            if (i < offset) {
                paddedShape[i] = 1;
                paddedStrides[i] = 0;
            } else {
                paddedShape[i] = a.internalShapeUnsafe()[i - offset];
                paddedStrides[i] = a.internalStridesUnsafe()[i - offset];
            }
        }

        for (int i = 0; i < ndim; i++) {
            if (paddedShape[i] == shape[i]) {
                newStrides[i] = paddedStrides[i];
            } else if (paddedShape[i] == 1) {
                newStrides[i] = 0;
            } else {
                throw new IllegalArgumentException(
                        "Cannot broadcast array of shape " + Arrays.toString(a.internalShapeUnsafe()) +
                                " to target shape " + Arrays.toString(shape) +
                                " because dimension " + i + " is incompatible: source dimension " +
                                paddedShape[i] + " cannot expand to " + shape[i]
                );
            }
        }
        return NDArray.ofRaw(a.getData(), shape, newStrides, a.getDType());
    }

    /**
     * Extracts a sub-tensor by fixing the first dimension to {@code index}.
     */
    public static NDArray subview(NDArray a, long index) {
        if (a.dim() == 0) {
            throw new IllegalStateException("Cannot take a subview of a 0-dimensional scalar array.");
        }
        long[] shape = a.internalShapeUnsafe();
        long[] strides = a.internalStridesUnsafe();
        long dim0 = shape[0];
        if (index < 0) index += dim0;
        if (index < 0 || index >= dim0) {
            throw new IndexOutOfBoundsException("Index " + index + " out of bounds for axis 0 with size " + dim0);
        }

        long byteOffset = index * strides[0] * a.getDType().layout.byteSize();
        MemorySegment subSegment = a.getData().asSlice(byteOffset);

        long[] subShape = Arrays.copyOfRange(shape, 1, shape.length);
        long[] subStrides = Arrays.copyOfRange(strides, 1, strides.length);
        return NDArray.ofRaw(subSegment, subShape, subStrides, a.getDType());
    }

    /**
     * Extracts a sub-tensor by fixing leading dimensions to {@code indices}.
     */
    public static NDArray subview(NDArray a, long... indices) {
        if (indices.length >= a.dim()) {
            throw new IllegalArgumentException("Cannot strip " + indices.length + " dimensions from an array of rank " + a.dim() + ". Use scalar getters instead.");
        }
        long[] shape = a.internalShapeUnsafe();
        long[] strides = a.internalStridesUnsafe();
        long byteOffset = 0;
        for (int d = 0; d < indices.length; d++) {
            long idx = indices[d];
            long dimSize = shape[d];
            if (idx < 0) idx += dimSize;
            if (idx < 0 || idx >= dimSize) {
                throw new IndexOutOfBoundsException("Index " + idx + " out of bounds for axis " + d + " with size " + dimSize);
            }
            byteOffset += idx * strides[d];
        }

        byteOffset *= a.getDType().layout.byteSize();
        MemorySegment subSegment = a.getData().asSlice(byteOffset);

        long[] subShape = Arrays.copyOfRange(shape, indices.length, shape.length);
        long[] subStrides = Arrays.copyOfRange(strides, indices.length, strides.length);
        return NDArray.ofRaw(subSegment, subShape, subStrides, a.getDType());
    }

    /**
     * Slices an array across multiple dimensions using {@link Slice} ranges.
     */
    public static NDArray slice(NDArray a, Slice... slices) {
        int rank = a.dim();
        if (slices.length > rank) {
            throw new IllegalArgumentException("Too many slices: " + slices.length + " provided for array of rank " + rank);
        }

        long[] shape = a.internalShapeUnsafe();
        long[] strides = a.internalStridesUnsafe();
        long[] newShape = new long[rank];
        long[] newStrides = new long[rank];
        long baseElementOffset = 0;

        for (int d = 0; d < rank; d++) {
            Slice s = d < slices.length ? slices[d] : Slice.all();
            Slice.ResolvedSlice resolved = s.resolve(shape[d]);
            newShape[d] = resolved.length();
            newStrides[d] = strides[d] * resolved.step();
            baseElementOffset += resolved.start() * strides[d];
        }

        long byteOffset = baseElementOffset * a.getDType().layout.byteSize();
        MemorySegment slicedSegment = a.getData().asSlice(byteOffset);
        return NDArray.ofRaw(slicedSegment, newShape, newStrides, a.getDType());
    }

    /**
     * Slices an array using a NumPy-style slice string (e.g., {@code "0:5, :"}).
     */
    public static NDArray slice(NDArray a, String sliceExpr) {
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
        return slice(a, sliceObjs);
    }

    /**
     * Removes a dimension of size 1 at the specified axis.
     */
    public static NDArray squeeze(NDArray a, int axis) {
        int currentDim = a.dim();
        if (axis < 0) axis += currentDim;
        if (axis < 0 || axis >= currentDim) {
            throw new IllegalArgumentException("Axis " + axis + " is out of bounds for array of dimension " + currentDim);
        }
        long[] shape = a.internalShapeUnsafe();
        if (shape[axis] != 1) {
            throw new IllegalArgumentException("Cannot squeeze axis " + axis + " because its size is not 1. Shape is: " + a.shapeString());
        }

        long[] newShape = new long[currentDim - 1];
        for (int i = 0, j = 0; i < currentDim; i++) {
            if (i != axis) {
                newShape[j++] = shape[i];
            }
        }
        return a.reshape(newShape);
    }

    /**
     * Inserts a new dimension of size 1 at the specified axis.
     */
    public static NDArray unsqueeze(NDArray a, int axis) {
        int currentDim = a.dim();
        if (axis < 0) axis += (currentDim + 1);
        if (axis < 0 || axis > currentDim) {
            throw new IllegalArgumentException("Axis " + axis + " is out of bounds for array of dimension " + currentDim);
        }

        long[] shape = a.internalShapeUnsafe();
        long[] newShape = new long[currentDim + 1];
        for (int i = 0, j = 0; i < newShape.length; i++) {
            if (i == axis) {
                newShape[i] = 1;
            } else {
                newShape[i] = shape[j++];
            }
        }
        return a.reshape(newShape);
    }
}