package jnum.internal.kernel.stat;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Arrays;

import jnum.NDArray;
import jnum.internal.layout.NDIter;
import jnum.internal.layout.ShapeUtil;
import jnum.stat.QuantileMethod;

/**
 * High-performance Order Statistics and Quantile selection via O(N) Introselect.
 * Replaces O(N log N) sorting with linear expected and worst-case selection.
 */
public final class Quantile {

    private Quantile() {
        throw new AssertionError("Quantile kernel cannot be instantiated.");
    }

    // =========================================================================
    // INTROSELECT ALGORITHMS (O(N) SELECTION)
    // =========================================================================

    public static void selectFloat(float[] a, int left, int right, int k) {
        int maxDepth = 2 * (31 - Integer.numberOfLeadingZeros(Math.max(1, right - left + 1)));
        while (right > left) {
            if (right - left <= 16) {
                insertionSortFloat(a, left, right);
                return;
            }
            if (maxDepth == 0) {
                Arrays.sort(a, left, right + 1);
                return;
            }
            maxDepth--;

            int mid = left + (right - left) / 2;
            int pivotIndex = medianOf3Float(a, left, mid, right);
            pivotIndex = partitionFloat(a, left, right, pivotIndex);

            if (k == pivotIndex) {
                return;
            } else if (k < pivotIndex) {
                right = pivotIndex - 1;
            } else {
                left = pivotIndex + 1;
            }
        }
    }

    private static int medianOf3Float(float[] a, int i, int j, int k) {
        float ai = a[i], aj = a[j], ak = a[k];
        if (ai < aj) {
            if (aj < ak) return j;
            return (ai < ak) ? k : i;
        } else {
            if (ai < ak) return i;
            return (aj < ak) ? k : j;
        }
    }

    private static int partitionFloat(float[] a, int left, int right, int pivotIndex) {
        float pivot = a[pivotIndex];
        swapFloat(a, pivotIndex, right);
        int storeIndex = left;
        for (int i = left; i < right; i++) {
            if (a[i] < pivot) {
                swapFloat(a, i, storeIndex);
                storeIndex++;
            }
        }
        swapFloat(a, storeIndex, right);
        return storeIndex;
    }

    private static void swapFloat(float[] a, int i, int j) {
        float tmp = a[i];
        a[i] = a[j];
        a[j] = tmp;
    }

    private static void insertionSortFloat(float[] a, int left, int right) {
        for (int i = left + 1; i <= right; i++) {
            float val = a[i];
            int j = i - 1;
            while (j >= left && a[j] > val) {
                a[j + 1] = a[j];
                j--;
            }
            a[j + 1] = val;
        }
    }

    public static void selectDouble(double[] a, int left, int right, int k) {
        int maxDepth = 2 * (31 - Integer.numberOfLeadingZeros(Math.max(1, right - left + 1)));
        while (right > left) {
            if (right - left <= 16) {
                insertionSortDouble(a, left, right);
                return;
            }
            if (maxDepth == 0) {
                Arrays.sort(a, left, right + 1);
                return;
            }
            maxDepth--;

            int mid = left + (right - left) / 2;
            int pivotIndex = medianOf3Double(a, left, mid, right);
            pivotIndex = partitionDouble(a, left, right, pivotIndex);

            if (k == pivotIndex) {
                return;
            } else if (k < pivotIndex) {
                right = pivotIndex - 1;
            } else {
                left = pivotIndex + 1;
            }
        }
    }

    private static int medianOf3Double(double[] a, int i, int j, int k) {
        double ai = a[i], aj = a[j], ak = a[k];
        if (ai < aj) {
            if (aj < ak) return j;
            return (ai < ak) ? k : i;
        } else {
            if (ai < ak) return i;
            return (aj < ak) ? k : j;
        }
    }

    private static int partitionDouble(double[] a, int left, int right, int pivotIndex) {
        double pivot = a[pivotIndex];
        swapDouble(a, pivotIndex, right);
        int storeIndex = left;
        for (int i = left; i < right; i++) {
            if (a[i] < pivot) {
                swapDouble(a, i, storeIndex);
                storeIndex++;
            }
        }
        swapDouble(a, storeIndex, right);
        return storeIndex;
    }

    private static void swapDouble(double[] a, int i, int j) {
        double tmp = a[i];
        a[i] = a[j];
        a[j] = tmp;
    }

    private static void insertionSortDouble(double[] a, int left, int right) {
        for (int i = left + 1; i <= right; i++) {
            double val = a[i];
            int j = i - 1;
            while (j >= left && a[j] > val) {
                a[j + 1] = a[j];
                j--;
            }
            a[j + 1] = val;
        }
    }

    public static void selectInt(int[] a, int left, int right, int k) {
        int maxDepth = 2 * (31 - Integer.numberOfLeadingZeros(Math.max(1, right - left + 1)));
        while (right > left) {
            if (right - left <= 16) {
                insertionSortInt(a, left, right);
                return;
            }
            if (maxDepth == 0) {
                Arrays.sort(a, left, right + 1);
                return;
            }
            maxDepth--;

            int mid = left + (right - left) / 2;
            int pivotIndex = medianOf3Int(a, left, mid, right);
            pivotIndex = partitionInt(a, left, right, pivotIndex);

            if (k == pivotIndex) {
                return;
            } else if (k < pivotIndex) {
                right = pivotIndex - 1;
            } else {
                left = pivotIndex + 1;
            }
        }
    }

    private static int medianOf3Int(int[] a, int i, int j, int k) {
        int ai = a[i], aj = a[j], ak = a[k];
        if (ai < aj) {
            if (aj < ak) return j;
            return (ai < ak) ? k : i;
        } else {
            if (ai < ak) return i;
            return (aj < ak) ? k : j;
        }
    }

    private static int partitionInt(int[] a, int left, int right, int pivotIndex) {
        int pivot = a[pivotIndex];
        swapInt(a, pivotIndex, right);
        int storeIndex = left;
        for (int i = left; i < right; i++) {
            if (a[i] < pivot) {
                swapInt(a, i, storeIndex);
                storeIndex++;
            }
        }
        swapInt(a, storeIndex, right);
        return storeIndex;
    }

    private static void swapInt(int[] a, int i, int j) {
        int tmp = a[i];
        a[i] = a[j];
        a[j] = tmp;
    }

    private static void insertionSortInt(int[] a, int left, int right) {
        for (int i = left + 1; i <= right; i++) {
            int val = a[i];
            int j = i - 1;
            while (j >= left && a[j] > val) {
                a[j + 1] = a[j];
                j--;
            }
            a[j + 1] = val;
        }
    }

    // =========================================================================
    // SCALAR QUANTILE
    // =========================================================================

    public static double quantileFloat(NDArray a, double q, QuantileMethod method) {
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

        return evaluateQuantileFloat(buf, n, q, method);
    }

    public static double evaluateQuantileFloat(float[] buf, int n, double q, QuantileMethod method) {
        double v = q * (n - 1);
        int i = (int) Math.floor(v);
        int j = Math.min(i + 1, n - 1);
        double f = v - i;

        selectFloat(buf, 0, n - 1, i);
        double valI = buf[i];

        if (i == j || f == 0.0) {
            return valI;
        }

        float valJ = buf[i + 1];
        for (int k = i + 2; k < n; k++) {
            if (buf[k] < valJ) valJ = buf[k];
        }

        return interpolate(valI, valJ, f, method, i);
    }

    public static double quantileDouble(NDArray a, double q, QuantileMethod method) {
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

        return evaluateQuantileDouble(buf, n, q, method);
    }

    public static double evaluateQuantileDouble(double[] buf, int n, double q, QuantileMethod method) {
        double v = q * (n - 1);
        int i = (int) Math.floor(v);
        int j = Math.min(i + 1, n - 1);
        double f = v - i;

        selectDouble(buf, 0, n - 1, i);
        double valI = buf[i];

        if (i == j || f == 0.0) {
            return valI;
        }

        double valJ = buf[i + 1];
        for (int k = i + 2; k < n; k++) {
            if (buf[k] < valJ) valJ = buf[k];
        }

        return interpolate(valI, valJ, f, method, i);
    }

    public static double quantileInt(NDArray a, double q, QuantileMethod method) {
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

        return evaluateQuantileInt(buf, n, q, method);
    }

    public static double evaluateQuantileInt(int[] buf, int n, double q, QuantileMethod method) {
        double v = q * (n - 1);
        int i = (int) Math.floor(v);
        int j = Math.min(i + 1, n - 1);
        double f = v - i;

        selectInt(buf, 0, n - 1, i);
        double valI = (double) buf[i];

        if (i == j || f == 0.0) {
            return valI;
        }

        int valJ = buf[i + 1];
        for (int k = i + 2; k < n; k++) {
            if (buf[k] < valJ) valJ = buf[k];
        }

        return interpolate(valI, (double) valJ, f, method, i);
    }

    private static double interpolate(double valI, double valJ, double f, QuantileMethod method, int i) {
        return switch (method) {
            case LINEAR -> valI + f * (valJ - valI);
            case LOWER -> valI;
            case HIGHER -> valJ;
            case MIDPOINT -> (valI + valJ) / 2.0;
            case NEAREST -> {
                if (f < 0.5) yield valI;
                if (f > 0.5) yield valJ;
                yield (i % 2 == 0) ? valI : valJ;
            }
        };
    }

    // =========================================================================
    // AXIS QUANTILE
    // =========================================================================

    public static NDArray quantileFloatAxis(NDArray a, int axis, double q, QuantileMethod method, NDArray resArray) {
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

            double qVal = evaluateQuantileFloat(buf, size, q, method);
            resSeg.setAtIndex(ValueLayout.JAVA_FLOAT, offsetRes, (float) qVal);
        }
        return resArray;
    }

    public static NDArray quantileDoubleAxis(NDArray a, int axis, double q, QuantileMethod method, NDArray resArray) {
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

            double qVal = evaluateQuantileDouble(buf, size, q, method);
            resSeg.setAtIndex(ValueLayout.JAVA_DOUBLE, offsetRes, qVal);
        }
        return resArray;
    }

    public static NDArray quantileIntAxis(NDArray a, int axis, double q, QuantileMethod method, NDArray resArray) {
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

            double qVal = evaluateQuantileInt(buf, size, q, method);
            resSeg.setAtIndex(ValueLayout.JAVA_DOUBLE, offsetRes, qVal);
        }
        return resArray;
    }
}
