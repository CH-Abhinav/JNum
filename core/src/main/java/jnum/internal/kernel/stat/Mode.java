package jnum.internal.kernel.stat;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Arrays;

import jnum.NDArray;
import jnum.internal.layout.NDIter;
import jnum.internal.layout.ShapeUtil;

/**
 * Kernels for Mode (most frequent value).
 * When frequencies are tied, returns the smallest value among the ties.
 */
public final class Mode {

    private Mode() {
        throw new AssertionError("Mode kernel cannot be instantiated.");
    }

    // =========================================================================
    // SCALAR MODE
    // =========================================================================

    public static double modeFloat(NDArray a) {
        int n = (int) a.getSize();
        if (n == 0) return Double.NaN;
        if (n == 1) return a.getData().getAtIndex(ValueLayout.JAVA_FLOAT, 0);

        float[] buf = new float[n];
        if (a.isContiguous()) {
            MemorySegment.copy(a.getData(), ValueLayout.JAVA_FLOAT, 0, buf, 0, n);
        } else {
            NDIter iter = new NDIter(a.internalShapeUnsafe());
            int idx = 0;
            while (iter.hasNext) {
                long offset = ShapeUtil.getByteOffset(iter.coords, a.internalStridesUnsafe(), a.getDType());
                buf[idx++] = a.getData().get(ValueLayout.JAVA_FLOAT, offset);
                iter.next();
            }
        }

        Arrays.sort(buf);
        return findModeFloat(buf, n);
    }

    private static float findModeFloat(float[] buf, int n) {
        float mode = buf[0];
        int maxCount = 1;
        float currentVal = buf[0];
        int currentCount = 1;

        for (int i = 1; i < n; i++) {
            if (buf[i] == currentVal) {
                currentCount++;
            } else {
                if (currentCount > maxCount) {
                    maxCount = currentCount;
                    mode = currentVal;
                }
                currentVal = buf[i];
                currentCount = 1;
            }
        }
        if (currentCount > maxCount) {
            mode = currentVal;
        }
        return mode;
    }

    public static double modeDouble(NDArray a) {
        int n = (int) a.getSize();
        if (n == 0) return Double.NaN;
        if (n == 1) return a.getData().getAtIndex(ValueLayout.JAVA_DOUBLE, 0);

        double[] buf = new double[n];
        if (a.isContiguous()) {
            MemorySegment.copy(a.getData(), ValueLayout.JAVA_DOUBLE, 0, buf, 0, n);
        } else {
            NDIter iter = new NDIter(a.internalShapeUnsafe());
            int idx = 0;
            while (iter.hasNext) {
                long offset = ShapeUtil.getByteOffset(iter.coords, a.internalStridesUnsafe(), a.getDType());
                buf[idx++] = a.getData().get(ValueLayout.JAVA_DOUBLE, offset);
                iter.next();
            }
        }

        Arrays.sort(buf);
        return findModeDouble(buf, n);
    }

    private static double findModeDouble(double[] buf, int n) {
        double mode = buf[0];
        int maxCount = 1;
        double currentVal = buf[0];
        int currentCount = 1;

        for (int i = 1; i < n; i++) {
            if (buf[i] == currentVal) {
                currentCount++;
            } else {
                if (currentCount > maxCount) {
                    maxCount = currentCount;
                    mode = currentVal;
                }
                currentVal = buf[i];
                currentCount = 1;
            }
        }
        if (currentCount > maxCount) {
            mode = currentVal;
        }
        return mode;
    }

    public static double modeInt(NDArray a) {
        int n = (int) a.getSize();
        if (n == 0) return Double.NaN;
        if (n == 1) return (double) a.getData().getAtIndex(ValueLayout.JAVA_INT, 0);

        int[] buf = new int[n];
        if (a.isContiguous()) {
            MemorySegment.copy(a.getData(), ValueLayout.JAVA_INT, 0, buf, 0, n);
        } else {
            NDIter iter = new NDIter(a.internalShapeUnsafe());
            int idx = 0;
            while (iter.hasNext) {
                long offset = ShapeUtil.getByteOffset(iter.coords, a.internalStridesUnsafe(), a.getDType());
                buf[idx++] = a.getData().get(ValueLayout.JAVA_INT, offset);
                iter.next();
            }
        }

        Arrays.sort(buf);
        return (double) findModeInt(buf, n);
    }

    private static int findModeInt(int[] buf, int n) {
        int mode = buf[0];
        int maxCount = 1;
        int currentVal = buf[0];
        int currentCount = 1;

        for (int i = 1; i < n; i++) {
            if (buf[i] == currentVal) {
                currentCount++;
            } else {
                if (currentCount > maxCount) {
                    maxCount = currentCount;
                    mode = currentVal;
                }
                currentVal = buf[i];
                currentCount = 1;
            }
        }
        if (currentCount > maxCount) {
            mode = currentVal;
        }
        return mode;
    }

    // =========================================================================
    // AXIS MODE
    // =========================================================================

    public static NDArray modeFloatAxis(NDArray a, int axis, NDArray resArray) {
        int size = (int) a.internalShapeUnsafe()[axis];
        long strideA = a.internalStridesUnsafe()[axis];
        long resSize = resArray.getSize();
        MemorySegment aSeg = a.getData();
        MemorySegment resSeg = resArray.getData();

        float[] buf = new float[size];

        for (long idx = 0; idx < resSize; idx++) {
            long tempIndex = idx;
            long baseOffset = 0;
            long offsetRes = 0;
            for (int d = resArray.internalShapeUnsafe().length - 1; d >= 0; d--) {
                long coord = tempIndex % resArray.internalShapeUnsafe()[d];
                tempIndex /= resArray.internalShapeUnsafe()[d];
                offsetRes += coord * resArray.internalStridesUnsafe()[d];

                int aDim = (d >= axis) ? d + 1 : d;
                baseOffset += coord * a.internalStridesUnsafe()[aDim];
            }

            for (int k = 0; k < size; k++) {
                buf[k] = aSeg.getAtIndex(ValueLayout.JAVA_FLOAT, baseOffset + (long) k * strideA);
            }

            Arrays.sort(buf);
            float mVal = findModeFloat(buf, size);
            resSeg.setAtIndex(ValueLayout.JAVA_FLOAT, offsetRes, mVal);
        }
        return resArray;
    }

    public static NDArray modeDoubleAxis(NDArray a, int axis, NDArray resArray) {
        int size = (int) a.internalShapeUnsafe()[axis];
        long strideA = a.internalStridesUnsafe()[axis];
        long resSize = resArray.getSize();
        MemorySegment aSeg = a.getData();
        MemorySegment resSeg = resArray.getData();

        double[] buf = new double[size];

        for (long idx = 0; idx < resSize; idx++) {
            long tempIndex = idx;
            long baseOffset = 0;
            long offsetRes = 0;
            for (int d = resArray.internalShapeUnsafe().length - 1; d >= 0; d--) {
                long coord = tempIndex % resArray.internalShapeUnsafe()[d];
                tempIndex /= resArray.internalShapeUnsafe()[d];
                offsetRes += coord * resArray.internalStridesUnsafe()[d];

                int aDim = (d >= axis) ? d + 1 : d;
                baseOffset += coord * a.internalStridesUnsafe()[aDim];
            }

            for (int k = 0; k < size; k++) {
                buf[k] = aSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, baseOffset + (long) k * strideA);
            }

            Arrays.sort(buf);
            double mVal = findModeDouble(buf, size);
            resSeg.setAtIndex(ValueLayout.JAVA_DOUBLE, offsetRes, mVal);
        }
        return resArray;
    }

    public static NDArray modeIntAxis(NDArray a, int axis, NDArray resArray) {
        int size = (int) a.internalShapeUnsafe()[axis];
        long strideA = a.internalStridesUnsafe()[axis];
        long resSize = resArray.getSize();
        MemorySegment aSeg = a.getData();
        MemorySegment resSeg = resArray.getData();

        int[] buf = new int[size];

        for (long idx = 0; idx < resSize; idx++) {
            long tempIndex = idx;
            long baseOffset = 0;
            long offsetRes = 0;
            for (int d = resArray.internalShapeUnsafe().length - 1; d >= 0; d--) {
                long coord = tempIndex % resArray.internalShapeUnsafe()[d];
                tempIndex /= resArray.internalShapeUnsafe()[d];
                offsetRes += coord * resArray.internalStridesUnsafe()[d];

                int aDim = (d >= axis) ? d + 1 : d;
                baseOffset += coord * a.internalStridesUnsafe()[aDim];
            }

            for (int k = 0; k < size; k++) {
                buf[k] = aSeg.getAtIndex(ValueLayout.JAVA_INT, baseOffset + (long) k * strideA);
            }

            Arrays.sort(buf);
            int mVal = findModeInt(buf, size);
            resSeg.setAtIndex(ValueLayout.JAVA_INT, offsetRes, mVal);
        }
        return resArray;
    }
}
