package jnum.internal.layout;

/**
 * Fast multi-dimensional coordinate and linear offset iterator for arbitrary strided layouts.
 *
 * <p>Employs an odometer-style algorithm with precomputed "backstrides" to achieve amortized
 * O(1) step complexity. Instead of recomputing {@code sum(coord[d] * stride[d])} on every step
 * (which costs O(rank)), it increments the running {@link #offset} by {@code strides[d]}.
 * When coordinate {@code d} rolls over to 0, it subtracts {@code backstride[d] = (shape[d] - 1) * stride[d]}
 * to rewind the pointer to the start of that axis.</p>
 */
public class NDIter {
    /** Target tensor shape. */
    public final long[] shape;
    /** Dimensional strides in elements. */
    public final long[] strides;
    /** Number of dimensions. */
    public final int rank;
    /** Current multidimensional coordinate vector. */
    public long[] coords;
    /** Rewind distances applied when a dimension counter rolls over from (shape - 1) to 0. */
    public final long[] backstrides;
    /** Current linear element offset into the buffer. */
    public long offset;
    /** True while there are remaining elements to visit. */
    public boolean hasNext;

    /**
     * Creates an iterator over a contiguous (C-order) shape.
     */
    public NDIter(long[] shape) {
        this(shape, ShapeUtil.calculateDefaultStrides(shape));
    }

    /**
     * Creates an iterator over an arbitrary strided shape. Precomputes backstrides.
     */
    public NDIter(long[] shape, long[] strides) {
        this.shape = shape;
        this.strides = strides;
        this.rank = shape.length;
        this.coords = new long[rank];
        this.backstrides = new long[rank];
        this.offset = 0;
        this.hasNext = true;
        for (int i = 0; i < rank; i++) {
            this.backstrides[i] = (shape[i] - 1) * strides[i];
        }
    }

    /**
     * Advances to the next multidimensional coordinate, updating {@link #coords} and {@link #offset}.
     * Sets {@link #hasNext} to false once all coordinates have been traversed.
     */
    public void next() {
        int last = rank - 1;
        while (last >= 0) {
            coords[last]++;
            if (coords[last] == shape[last]) {
                coords[last] = 0;
                offset -= backstrides[last];
                last--;
            } else {
                offset += strides[last];
                break;
            }
        }
        if (last < 0) {
            hasNext = false;
        }
    }

    /**
     * Fills an index map with consecutive offsets for SIMD vector gather operations.
     */
    public int nextVector(long[] indexMap, int vl) {
        int count = 0;
        while (this.hasNext && count < vl) {
            indexMap[count++] = this.offset;
            this.next();
        }
        return count;
    }
}
