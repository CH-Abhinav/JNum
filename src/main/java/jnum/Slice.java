package jnum;

/**
 * Representation of a multidimensional array slice range.
 * Supports Python/NumPy-style negative indexing and strided stepping.
 *
 * <p><b>Valhalla Candidate:</b> In JDK 28+, this is designed to be declared
 * as a {@code value record} for zero-allocation register passing.</p>
 */
public record Slice(long start, long stop, long step) {
    public static final long UNBOUNDED_START = Long.MIN_VALUE;
    public static final long UNBOUNDED_STOP = Long.MAX_VALUE;

    public Slice {
        if (step == 0) {
            throw new IllegalArgumentException("Slice step cannot be zero.");
        }
    }

    public static Slice all() { return new Slice(UNBOUNDED_START, UNBOUNDED_STOP, 1); }
    public static Slice to(long stop) { return new Slice(0, stop, 1); }
    public static Slice from(long start) { return new Slice(start, UNBOUNDED_STOP, 1); }
    public static Slice range(long start, long stop) { return new Slice(start, stop, 1); }
    public static Slice range(long start, long stop, long step) { return new Slice(start, stop, step); }
    public static Slice step(long step) { return new Slice(UNBOUNDED_START, UNBOUNDED_STOP, step); }

    public ResolvedSlice resolve(long dimSize) {
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

    public record ResolvedSlice(long start, long stop, long step, long length) {}
}