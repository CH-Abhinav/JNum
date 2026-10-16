package jnum;

/**
 * Representation of a multidimensional array slice range for strided views.
 *
 * <p>A {@code Slice} encapsulates a {@code [start, stop, step]} range specification mimicking
 * Python / NumPy array indexing syntax (e.g., {@code arr[1:10:2]}, {@code arr[::-1]}).</p>
 *
 * <p>Key features:
 * <ul>
 *   <li><b>Symbolic Unbounded Ends:</b> Slices can omit start or stop (e.g. {@link #all()}, {@link #to(long)},
 *       {@link #from(long)}), represented internally by sentinel values {@link #UNBOUNDED_START} and
 *       {@link #UNBOUNDED_STOP}. These are bound to a concrete dimension length during {@link #resolve(long)}.</li>
 *   <li><b>Python-Style Negative Indexing:</b> Negative values for {@code start} or {@code stop} count backward
 *       from the end of the dimension (e.g., {@code -1} refers to the last element).</li>
 *   <li><b>Strided and Inverted Stepping:</b> Positive steps select elements moving forward; negative steps
 *       traverse backward with negative physical strides, enabling zero-copy array reversal.</li>
 * </ul>
 * </p>
 *
 * <p><b>Valhalla Candidate:</b> In upcoming JDK versions (Project Valhalla), this record is designed
 * to be declared as a {@code value record} for zero-heap allocation and CPU register passing.</p>
 *
 * @param start the inclusive starting index (or {@link #UNBOUNDED_START})
 * @param stop the exclusive stopping boundary (or {@link #UNBOUNDED_STOP})
 * @param step the stride increment between elements (must be non-zero)
 */
public record Slice(long start, long stop, long step) {

    /** Sentinel value representing an omitted or unbounded start index (equivalent to {@code :stop} in Python). */
    public static final long UNBOUNDED_START = Long.MIN_VALUE;

    /** Sentinel value representing an omitted or unbounded stop index (equivalent to {@code start:} in Python). */
    public static final long UNBOUNDED_STOP = Long.MAX_VALUE;

    /**
     * Compact constructor validating that the slice step increment is non-zero.
     *
     * @param start inclusive start index or {@link #UNBOUNDED_START}
     * @param stop exclusive stop boundary or {@link #UNBOUNDED_STOP}
     * @param step stride step increment
     * @throws IllegalArgumentException if {@code step == 0}
     */
    public Slice {
        if (step == 0) {
            throw new IllegalArgumentException("Slice step cannot be zero.");
        }
    }

    /**
     * Creates a slice covering an entire dimension with step 1 (equivalent to {@code :} in Python).
     *
     * @return an unbounded full-dimension slice
     */
    public static Slice all() { return new Slice(UNBOUNDED_START, UNBOUNDED_STOP, 1); }

    /**
     * Creates a slice from index 0 up to {@code stop} (exclusive) with step 1 (equivalent to {@code :stop} in Python).
     *
     * @param stop exclusive stop boundary
     * @return slice from 0 to {@code stop}
     */
    public static Slice to(long stop) { return new Slice(0, stop, 1); }

    /**
     * Creates a slice from {@code start} (inclusive) through the end of the dimension with step 1 (equivalent to {@code start:} in Python).
     *
     * @param start inclusive start index
     * @return slice from {@code start} to the end of the dimension
     */
    public static Slice from(long start) { return new Slice(start, UNBOUNDED_STOP, 1); }

    /**
     * Creates a forward slice from {@code start} (inclusive) to {@code stop} (exclusive) with step 1 (equivalent to {@code start:stop} in Python).
     *
     * @param start inclusive start index
     * @param stop exclusive stop boundary
     * @return slice from {@code start} to {@code stop}
     */
    public static Slice range(long start, long stop) { return new Slice(start, stop, 1); }

    /**
     * Creates a strided slice from {@code start} (inclusive) to {@code stop} (exclusive) with arbitrary {@code step} (equivalent to {@code start:stop:step} in Python).
     *
     * @param start inclusive start index
     * @param stop exclusive stop boundary
     * @param step stride increment (positive or negative, non-zero)
     * @return strided slice
     * @throws IllegalArgumentException if {@code step == 0}
     */
    public static Slice range(long start, long stop, long step) { return new Slice(start, stop, step); }

    /**
     * Creates a strided slice across the entire dimension with arbitrary {@code step} (equivalent to {@code ::step} in Python).
     *
     * <p>For example, {@code Slice.step(-1)} reverses the entire dimension.</p>
     *
     * @param step stride increment (positive or negative, non-zero)
     * @return full-dimension slice with custom step
     * @throws IllegalArgumentException if {@code step == 0}
     */
    public static Slice step(long step) { return new Slice(UNBOUNDED_START, UNBOUNDED_STOP, step); }

    /**
     * Resolves symbolic bounds, negative indices, and clamping against a concrete dimension size.
     *
     * <p>Resolution rules:
     * <ul>
     *   <li><b>Positive Step ({@code step > 0}):</b>
     *     <ul>
     *       <li>Unbounded start defaults to {@code 0}.</li>
     *       <li>Unbounded stop defaults to {@code dimSize}.</li>
     *       <li>Negative indices are converted to {@code dimSize + index}.</li>
     *       <li>Bounds are clamped to the interval {@code [0, dimSize]}.</li>
     *       <li>Total length is computed using ceiling division: {@code ceil((stop - start) / step)}.</li>
     *     </ul>
     *   </li>
     *   <li><b>Negative Step ({@code step < 0}):</b>
     *     <ul>
     *       <li>Unbounded start defaults to {@code dimSize - 1}.</li>
     *       <li>Unbounded stop defaults to {@code -1}.</li>
     *       <li>Negative indices are converted to {@code dimSize + index}.</li>
     *       <li>Bounds are clamped to the interval {@code [-1, dimSize - 1]}.</li>
     *       <li>Total length is computed using ceiling division: {@code ceil((start - stop) / (-step))}.</li>
     *     </ul>
     *   </li>
     * </ul>
     * </p>
     *
     * @param dimSize the concrete length of the dimension being sliced
     * @return a {@link ResolvedSlice} containing physical start, stop, step, and element count
     * @throws IllegalArgumentException if {@code dimSize < 0}
     */
    public ResolvedSlice resolve(long dimSize) {
        if (dimSize < 0) {
            throw new IllegalArgumentException("Dimension size cannot be negative: " + dimSize);
        }
        long st = step, s, e;
        if (st > 0) {
            s = start == UNBOUNDED_START ? 0 : (start < 0 ? dimSize + start : start);
            s = Math.clamp(s, 0, dimSize);
            e = stop == UNBOUNDED_STOP ? dimSize : (stop < 0 ? dimSize + stop : stop);
            e = Math.clamp(e, 0, dimSize);
            long length = e > s ? (e - s + st - 1) / st : 0;
            return new ResolvedSlice(s, e, st, Math.max(0, length));
        } else {
            s = start == UNBOUNDED_START ? dimSize - 1 : (start < 0 ? dimSize + start : start);
            s = Math.clamp(s, -1, dimSize - 1);
            e = stop == UNBOUNDED_STOP ? -1 : (stop < 0 ? dimSize + stop : stop);
            e = Math.clamp(e, -1, dimSize - 1);
            long length = s > e ? (s - e - st - 1) / (-st) : 0;
            return new ResolvedSlice(s, e, st, Math.max(0, length));
        }
    }

    /**
     * Concrete, resolved slice parameters bound to a specific dimension length.
     *
     * @param start physical start index within {@code [0, dimSize - 1]}
     * @param stop physical boundary index
     * @param step stride increment (positive or negative)
     * @param length total number of elements yielded by this slice
     */
    public record ResolvedSlice(long start, long stop, long step, long length) {}
}