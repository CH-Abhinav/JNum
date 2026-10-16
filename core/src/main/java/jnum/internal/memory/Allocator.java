package jnum.internal.memory;

import jnum.DType;
import jnum.NDArray;
import jnum.internal.layout.ShapeUtil;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Arrays;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Internal allocator and factory operations for raw buffer allocation,
 * array initialization, primitive array bridging, random generation, and sequence creation.
 */
public final class Allocator {

    private Allocator() {
        throw new AssertionError("Allocator utility class cannot be instantiated.");
    }

    /**
     * Allocates a contiguous, 64-byte aligned memory segment for an array of the given shape and data type.
     */
    public static NDArray allocateEmpty(Arena arena, DType dType, long... shape) {
        long size = 1;
        for (long dim : shape) size *= dim;
        long byteSize = size * dType.layout.byteSize();
        MemorySegment segment = arena.allocate(byteSize, 64);
        return NDArray.ofRaw(segment, shape, ShapeUtil.calculateDefaultStrides(shape), dType);
    }

    /**
     * Creates an array filled with zeros (native zero-initialized buffer).
     */
    public static NDArray zeros(Arena arena, DType dType, long... shape) {
        return allocateEmpty(arena, dType, shape);
    }

    /**
     * Creates an array filled with ones.
     */
    public static NDArray ones(Arena arena, DType dType, long... shape) {
        NDArray result = allocateEmpty(arena, dType, shape);
        long size = result.getSize();
        MemorySegment segment = result.getData();
        switch (dType) {
            case i32 -> {
                for (long i = 0; i < size; i++) {
                    segment.setAtIndex(ValueLayout.JAVA_INT, i, 1);
                }
            }
            case f32 -> {
                for (long i = 0; i < size; i++) {
                    segment.setAtIndex(ValueLayout.JAVA_FLOAT, i, 1.0f);
                }
            }
            case f64 -> {
                for (long i = 0; i < size; i++) {
                    segment.setAtIndex(ValueLayout.JAVA_DOUBLE, i, 1.0);
                }
            }
            case bool -> {
                for (long i = 0; i < size; i++) {
                    segment.setAtIndex(ValueLayout.JAVA_BYTE, i, (byte) 1);
                }
            }
            default -> throw new UnsupportedOperationException("This dtype " + dType + " doesn't support this method");
        }
        return result;
    }

    /**
     * Creates an array from a primitive float[] array.
     */
    public static NDArray from(Arena arena, float[] data, long... shape) {
        long calcSize = 1;
        for (long dim : shape) calcSize *= dim;
        if (calcSize != data.length) {
            throw new IllegalArgumentException(
                "Requested shape " + Arrays.toString(shape) +
                " requires size " + calcSize +
                ", but the provided array has length " + data.length + "."
            );
        }
        DType dType = DType.f32;
        long byteSize = calcSize * dType.layout.byteSize();
        MemorySegment segment = arena.allocate(byteSize, 64);
        MemorySegment.copy(data, 0, segment, dType.layout, 0, data.length);
        return NDArray.ofRaw(segment, shape, ShapeUtil.calculateDefaultStrides(shape), dType);
    }

    /**
     * Creates an array from a primitive int[] array.
     */
    public static NDArray from(Arena arena, int[] data, long... shape) {
        long calcSize = 1;
        for (long dim : shape) calcSize *= dim;
        if (calcSize != data.length) {
            throw new IllegalArgumentException(
                "Requested shape " + Arrays.toString(shape) +
                " requires size " + calcSize +
                ", but the provided array has length " + data.length + "."
            );
        }
        DType dType = DType.i32;
        long byteSize = calcSize * dType.layout.byteSize();
        MemorySegment segment = arena.allocate(byteSize, 64);
        MemorySegment.copy(data, 0, segment, dType.layout, 0, data.length);
        return NDArray.ofRaw(segment, shape, ShapeUtil.calculateDefaultStrides(shape), dType);
    }

    /**
     * Creates an array from a primitive double[] array.
     */
    public static NDArray from(Arena arena, double[] data, long... shape) {
        long calcSize = 1;
        for (long dim : shape) calcSize *= dim;
        if (calcSize != data.length) {
            throw new IllegalArgumentException(
                "Requested shape " + Arrays.toString(shape) +
                " requires size " + calcSize +
                ", but the provided array has length " + data.length + "."
            );
        }
        DType dType = DType.f64;
        long byteSize = calcSize * dType.layout.byteSize();
        MemorySegment segment = arena.allocate(byteSize, 64);
        MemorySegment.copy(data, 0, segment, dType.layout, 0, data.length);
        return NDArray.ofRaw(segment, shape, ShapeUtil.calculateDefaultStrides(shape), dType);
    }

    /**
     * Creates an array from a primitive boolean[] array.
     */
    public static NDArray from(Arena arena, boolean[] data, long... shape) {
        long calcSize = 1;
        for (long dim : shape) calcSize *= dim;
        if (calcSize != data.length) {
            throw new IllegalArgumentException(
                "Requested shape " + Arrays.toString(shape) +
                " requires size " + calcSize +
                ", but the provided array has length " + data.length + "."
            );
        }
        DType dType = DType.bool;
        long byteSize = calcSize * dType.layout.byteSize();
        MemorySegment segment = arena.allocate(byteSize, 64);
        for (int i = 0; i < data.length; i++) {
            segment.setAtIndex(ValueLayout.JAVA_BYTE, i, (byte) (data[i] ? 1 : 0));
        }
        return NDArray.ofRaw(segment, shape, ShapeUtil.calculateDefaultStrides(shape), dType);
    }

    /**
     * Creates an array filled with pseudo-random values.
     */
    public static NDArray rand(Arena arena, DType dType, long... shape) {
        NDArray resArray = zeros(arena, dType, shape);
        switch (dType) {
            case f32 -> {
                for (long i = 0; i < resArray.getSize(); i++) {
                    resArray.getData().setAtIndex(ValueLayout.JAVA_FLOAT, i, ThreadLocalRandom.current().nextFloat());
                }
            }
            case i32 -> {
                for (long i = 0; i < resArray.getSize(); i++) {
                    resArray.getData().setAtIndex(ValueLayout.JAVA_INT, i, ThreadLocalRandom.current().nextInt());
                }
            }
            case f64 -> {
                for (long i = 0; i < resArray.getSize(); i++) {
                    resArray.getData().setAtIndex(ValueLayout.JAVA_DOUBLE, i, ThreadLocalRandom.current().nextDouble());
                }
            }
            case bool -> {
                for (long i = 0; i < resArray.getSize(); i++) {
                    resArray.getData().setAtIndex(ValueLayout.JAVA_BYTE, i, (byte) (ThreadLocalRandom.current().nextBoolean() ? 1 : 0));
                }
            }
            default -> throw new UnsupportedOperationException("This dtype " + dType + " doesn't support this method");
        }
        return resArray;
    }

    /**
     * Creates an array with pseudo-random values bounded by an int max.
     */
    public static NDArray rand(Arena arena, int max, DType dType, long... shape) {
        NDArray resArray = zeros(arena, dType, shape);
        switch (dType) {
            case f32 -> {
                for (long i = 0; i < resArray.getSize(); i++) {
                    resArray.getData().setAtIndex(ValueLayout.JAVA_FLOAT, i, ThreadLocalRandom.current().nextFloat(max));
                }
            }
            case i32 -> {
                for (long i = 0; i < resArray.getSize(); i++) {
                    resArray.getData().setAtIndex(ValueLayout.JAVA_INT, i, ThreadLocalRandom.current().nextInt(max));
                }
            }
            case f64 -> {
                for (long i = 0; i < resArray.getSize(); i++) {
                    resArray.getData().setAtIndex(ValueLayout.JAVA_DOUBLE, i, ThreadLocalRandom.current().nextDouble(max));
                }
            }
            default -> throw new UnsupportedOperationException("This dtype " + dType + " doesn't support this method");
        }
        return resArray;
    }

    /**
     * Creates an array with pseudo-random values bounded by a float max.
     */
    public static NDArray rand(Arena arena, float max, DType dType, long... shape) {
        NDArray resArray = zeros(arena, dType, shape);
        switch (dType) {
            case f32 -> {
                for (long i = 0; i < resArray.getSize(); i++) {
                    resArray.getData().setAtIndex(ValueLayout.JAVA_FLOAT, i, ThreadLocalRandom.current().nextFloat(max));
                }
            }
            case i32 -> throw new IllegalArgumentException(
                "rand(float max, DType, shape) cannot generate FLOAT random values into dtype " +
                dType + " for shape " + Arrays.toString(shape)
            );
            case f64 -> {
                for (long i = 0; i < resArray.getSize(); i++) {
                    resArray.getData().setAtIndex(ValueLayout.JAVA_DOUBLE, i, ThreadLocalRandom.current().nextDouble(max));
                }
            }
            default -> throw new UnsupportedOperationException("This dtype " + dType + " doesn't support this method");
        }
        return resArray;
    }

    /**
     * Creates an array with pseudo-random values bounded by a double max.
     */
    public static NDArray rand(Arena arena, double max, DType dType, long... shape) {
        NDArray resArray = zeros(arena, dType, shape);
        switch (dType) {
            case f32 -> throw new IllegalArgumentException(
                "rand(double max, DType, shape) cannot generate DOUBLE random values into dtype " +
                dType + " for shape " + Arrays.toString(shape)
            );
            case i32 -> throw new IllegalArgumentException(
                "rand(double max, DType, shape) cannot generate DOUBLE random values into dtype " +
                dType + " for shape " + Arrays.toString(shape)
            );
            case f64 -> {
                for (long i = 0; i < resArray.getSize(); i++) {
                    resArray.getData().setAtIndex(ValueLayout.JAVA_DOUBLE, i, ThreadLocalRandom.current().nextDouble(max));
                }
            }
            default -> throw new UnsupportedOperationException("This dtype " + dType + " doesn't support this method");
        }
        return resArray;
    }

    /**
     * Creates an array with pseudo-random values in [min, max) for int bounds.
     */
    public static NDArray rand(Arena arena, int min, int max, DType dType, long... shape) {
        NDArray resArray = zeros(arena, dType, shape);
        switch (dType) {
            case f32 -> {
                for (long i = 0; i < resArray.getSize(); i++) {
                    resArray.getData().setAtIndex(ValueLayout.JAVA_FLOAT, i, ThreadLocalRandom.current().nextFloat(min, max));
                }
            }
            case i32 -> {
                for (long i = 0; i < resArray.getSize(); i++) {
                    resArray.getData().setAtIndex(ValueLayout.JAVA_INT, i, ThreadLocalRandom.current().nextInt(min, max));
                }
            }
            case f64 -> {
                for (long i = 0; i < resArray.getSize(); i++) {
                    resArray.getData().setAtIndex(ValueLayout.JAVA_DOUBLE, i, ThreadLocalRandom.current().nextDouble(min, max));
                }
            }
            default -> throw new UnsupportedOperationException("This dtype " + dType + " doesn't support this method");
        }
        return resArray;
    }

    /**
     * Creates an array with pseudo-random values in [min, max) for float bounds.
     */
    public static NDArray rand(Arena arena, float min, float max, DType dType, long... shape) {
        NDArray resArray = zeros(arena, dType, shape);
        switch (dType) {
            case f32 -> {
                for (long i = 0; i < resArray.getSize(); i++) {
                    resArray.getData().setAtIndex(ValueLayout.JAVA_FLOAT, i, ThreadLocalRandom.current().nextFloat(min, max));
                }
            }
            case i32 -> throw new IllegalArgumentException(
                "rand(float min, float max, DType, shape) cannot generate FLOAT random values into dtype " +
                dType + " for shape " + Arrays.toString(shape)
            );
            case f64 -> {
                for (long i = 0; i < resArray.getSize(); i++) {
                    resArray.getData().setAtIndex(ValueLayout.JAVA_DOUBLE, i, ThreadLocalRandom.current().nextDouble(min, max));
                }
            }
            default -> throw new UnsupportedOperationException("This dtype " + dType + " doesn't support this method");
        }
        return resArray;
    }

    /**
     * Creates an array with pseudo-random values in [min, max) for double bounds.
     */
    public static NDArray rand(Arena arena, double min, double max, DType dType, long... shape) {
        NDArray resArray = zeros(arena, dType, shape);
        switch (dType) {
            case f32 -> throw new IllegalArgumentException(
                "rand(double min, double max, DType, shape) cannot generate DOUBLE random values into dtype " +
                dType + " for shape " + Arrays.toString(shape)
            );
            case i32 -> throw new IllegalArgumentException(
                "rand(double min, double max, DType, shape) cannot generate DOUBLE random values into dtype " +
                dType + " for shape " + Arrays.toString(shape)
            );
            case f64 -> {
                for (long i = 0; i < resArray.getSize(); i++) {
                    resArray.getData().setAtIndex(ValueLayout.JAVA_DOUBLE, i, ThreadLocalRandom.current().nextDouble(min, max));
                }
            }
            default -> throw new UnsupportedOperationException("This dtype " + dType + " doesn't support this method");
        }
        return resArray;
    }

    /**
     * Creates a 1D sequence array within [start, stop) with given step.
     */
    public static NDArray arange(Arena arena, DType dType, double start, double stop, double step) {
        if (step == 0) {
            throw new IllegalArgumentException("Step cannot be zero.");
        }

        long size = (long) Math.ceil((stop - start) / step);
        if (size <= 0) {
            return NDArray.ofRaw(arena.allocate(0), new long[]{0}, new long[]{1}, dType);
        }

        long byteSize = size * dType.layout.byteSize();
        MemorySegment segment = arena.allocate(byteSize, 64);

        switch (dType) {
            case f32 -> {
                for (long i = 0; i < size; i++) {
                    segment.setAtIndex(ValueLayout.JAVA_FLOAT, i, (float) (start + i * step));
                }
            }
            case f64 -> {
                for (long i = 0; i < size; i++) {
                    segment.setAtIndex(ValueLayout.JAVA_DOUBLE, i, start + i * step);
                }
            }
            case i32 -> {
                for (long i = 0; i < size; i++) {
                    segment.setAtIndex(ValueLayout.JAVA_INT, i, (int) (start + i * step));
                }
            }
            case bool -> throw new IllegalArgumentException("arange is not supported for boolean type.");
            default -> throw new UnsupportedOperationException("Unsupported dtype: " + dType);
        }

        return NDArray.ofRaw(segment, new long[]{size}, new long[]{1}, dType);
    }

    /**
     * Creates an n x n 2D identity matrix.
     */
    public static NDArray eye(Arena arena, DType dtype, int n) {
        NDArray result = zeros(arena, dtype, n, n);
        for (int i = 0; i < n; i++) {
            switch (dtype) {
                case f32 -> result.setFloat(1.0f, i, i);
                case f64 -> result.setDouble(1.0, i, i);
                case i32 -> result.setInt(1, i, i);
                case bool -> result.setBoolean(true, i, i);
            }
        }
        return result;
    }
}
