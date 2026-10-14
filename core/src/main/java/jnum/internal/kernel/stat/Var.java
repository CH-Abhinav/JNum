package jnum.internal.kernel.stat;

import static jnum.internal.Constants.*;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.concurrent.RecursiveAction;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jnum.NDArray;
import jnum.internal.layout.NDIter;
import jnum.internal.layout.ShapeUtil;

/**
 * Numerically stable, SIMD-accelerated variance kernels with arbitrary degrees of freedom (ddof).
 */
public final class Var {

    private Var() {
        throw new AssertionError("Var kernel cannot be instantiated.");
    }

    // =========================================================================
    // SCALAR VARIANCE
    // =========================================================================

    public static double varFloat(NDArray a, int ddof) {
        long size = a.getSize();
        if (size == 0) return Double.NaN;
        double divisor = (double) (size - ddof);
        if (divisor <= 0) return Double.NaN;
        if (size == 1) return 0.0;

        double m = Mean.meanFloat(a);

        if (!a.isContiguous()) {
            double sumSq = 0.0;
            NDIter iter = new NDIter(a.internalShapeUnsafe());
            while (iter.hasNext) {
                long byteOffset = ShapeUtil.getByteOffset(iter.coords, a.internalStridesUnsafe(), a.getDType());
                double diff = (double) a.getData().get(ValueLayout.JAVA_FLOAT, byteOffset) - m;
                sumSq += diff * diff;
                iter.next();
            }
            return sumSq / divisor;
        }

        if (size >= PARALLEL_ELEMENT_THRESHOLD && AVAILABLE_CORES > 1) {
            int nChunks = Math.min(AVAILABLE_CORES, (int) Math.max(1, size / PARALLEL_ELEMENT_THRESHOLD));
            long chunkSize = (size + nChunks - 1) / nChunks;
            double[] partials = new double[nChunks];
            RecursiveAction[] tasks = new RecursiveAction[nChunks];
            MemorySegment seg = a.getData();

            for (int t = 0; t < nChunks; t++) {
                final int idx = t;
                final long start = t * chunkSize;
                final long end = Math.min(start + chunkSize, size);
                tasks[t] = new RecursiveAction() {
                    @Override
                    protected void compute() {
                        partials[idx] = computeVarFloatChunk(seg, m, start, end);
                    }
                };
            }
            POOL.invoke(new RecursiveAction() {
                @Override
                protected void compute() {
                    invokeAll(tasks);
                }
            });
            double sumSq = 0.0;
            for (double p : partials) sumSq += p;
            return sumSq / divisor;
        }

        return computeVarFloatChunk(a.getData(), m, 0, size) / divisor;
    }

    private static double computeVarFloatChunk(MemorySegment seg, double m, long start, long end) {
        long len = end - start;
        long loopbound4 = start + ((len) / (VL_F32 * 4)) * (VL_F32 * 4);
        FloatVector vMean = FloatVector.broadcast(SPECIES_F32, (float) m);
        var vSum1 = FloatVector.zero(SPECIES_F32);
        var vSum2 = FloatVector.zero(SPECIES_F32);
        var vSum3 = FloatVector.zero(SPECIES_F32);
        var vSum4 = FloatVector.zero(SPECIES_F32);

        long i = start;
        for (; i < loopbound4; i += VL_F32 * 4) {
            var d1 = FloatVector.fromMemorySegment(SPECIES_F32, seg, i * BYTES_F32, NATIVE_ORDER).sub(vMean);
            var d2 = FloatVector.fromMemorySegment(SPECIES_F32, seg, (i + VL_F32) * BYTES_F32, NATIVE_ORDER).sub(vMean);
            var d3 = FloatVector.fromMemorySegment(SPECIES_F32, seg, (i + VL_F32 * 2) * BYTES_F32, NATIVE_ORDER).sub(vMean);
            var d4 = FloatVector.fromMemorySegment(SPECIES_F32, seg, (i + VL_F32 * 3) * BYTES_F32, NATIVE_ORDER).sub(vMean);
            vSum1 = d1.fma(d1, vSum1);
            vSum2 = d2.fma(d2, vSum2);
            vSum3 = d3.fma(d3, vSum3);
            vSum4 = d4.fma(d4, vSum4);
        }

        var vCombined = vSum1.add(vSum2).add(vSum3).add(vSum4);
        double total = (double) vCombined.reduceLanes(VectorOperators.ADD);

        long loopbound = i + SPECIES_F32.loopBound(end - i);
        var vTail = FloatVector.zero(SPECIES_F32);
        for (; i < loopbound; i += VL_F32) {
            var d = FloatVector.fromMemorySegment(SPECIES_F32, seg, i * BYTES_F32, NATIVE_ORDER).sub(vMean);
            vTail = d.fma(d, vTail);
        }
        total += (double) vTail.reduceLanes(VectorOperators.ADD);

        for (; i < end; i++) {
            double diff = (double) seg.getAtIndex(ValueLayout.JAVA_FLOAT, i) - m;
            total += diff * diff;
        }
        return total;
    }

    public static double varDouble(NDArray a, int ddof) {
        long size = a.getSize();
        if (size == 0) return Double.NaN;
        double divisor = (double) (size - ddof);
        if (divisor <= 0) return Double.NaN;
        if (size == 1) return 0.0;

        double m = Mean.meanDouble(a);

        if (!a.isContiguous()) {
            double sumSq = 0.0;
            NDIter iter = new NDIter(a.internalShapeUnsafe());
            while (iter.hasNext) {
                long byteOffset = ShapeUtil.getByteOffset(iter.coords, a.internalStridesUnsafe(), a.getDType());
                double diff = a.getData().get(ValueLayout.JAVA_DOUBLE, byteOffset) - m;
                sumSq += diff * diff;
                iter.next();
            }
            return sumSq / divisor;
        }

        if (size >= PARALLEL_ELEMENT_THRESHOLD && AVAILABLE_CORES > 1) {
            int nChunks = Math.min(AVAILABLE_CORES, (int) Math.max(1, size / PARALLEL_ELEMENT_THRESHOLD));
            long chunkSize = (size + nChunks - 1) / nChunks;
            double[] partials = new double[nChunks];
            RecursiveAction[] tasks = new RecursiveAction[nChunks];
            MemorySegment seg = a.getData();

            for (int t = 0; t < nChunks; t++) {
                final int idx = t;
                final long start = t * chunkSize;
                final long end = Math.min(start + chunkSize, size);
                tasks[t] = new RecursiveAction() {
                    @Override
                    protected void compute() {
                        partials[idx] = computeVarDoubleChunk(seg, m, start, end);
                    }
                };
            }
            POOL.invoke(new RecursiveAction() {
                @Override
                protected void compute() {
                    invokeAll(tasks);
                }
            });
            double sumSq = 0.0;
            for (double p : partials) sumSq += p;
            return sumSq / divisor;
        }

        return computeVarDoubleChunk(a.getData(), m, 0, size) / divisor;
    }

    private static double computeVarDoubleChunk(MemorySegment seg, double m, long start, long end) {
        long len = end - start;
        long loopbound4 = start + ((len) / (VL_F64 * 4)) * (VL_F64 * 4);
        DoubleVector vMean = DoubleVector.broadcast(SPECIES_F64, m);
        var vSum1 = DoubleVector.zero(SPECIES_F64);
        var vSum2 = DoubleVector.zero(SPECIES_F64);
        var vSum3 = DoubleVector.zero(SPECIES_F64);
        var vSum4 = DoubleVector.zero(SPECIES_F64);

        long i = start;
        for (; i < loopbound4; i += VL_F64 * 4) {
            var d1 = DoubleVector.fromMemorySegment(SPECIES_F64, seg, i * BYTES_F64, NATIVE_ORDER).sub(vMean);
            var d2 = DoubleVector.fromMemorySegment(SPECIES_F64, seg, (i + VL_F64) * BYTES_F64, NATIVE_ORDER).sub(vMean);
            var d3 = DoubleVector.fromMemorySegment(SPECIES_F64, seg, (i + VL_F64 * 2) * BYTES_F64, NATIVE_ORDER).sub(vMean);
            var d4 = DoubleVector.fromMemorySegment(SPECIES_F64, seg, (i + VL_F64 * 3) * BYTES_F64, NATIVE_ORDER).sub(vMean);
            vSum1 = d1.fma(d1, vSum1);
            vSum2 = d2.fma(d2, vSum2);
            vSum3 = d3.fma(d3, vSum3);
            vSum4 = d4.fma(d4, vSum4);
        }

        var vCombined = vSum1.add(vSum2).add(vSum3).add(vSum4);
        double total = vCombined.reduceLanes(VectorOperators.ADD);

        long loopbound = i + SPECIES_F64.loopBound(end - i);
        var vTail = DoubleVector.zero(SPECIES_F64);
        for (; i < loopbound; i += VL_F64) {
            var d = DoubleVector.fromMemorySegment(SPECIES_F64, seg, i * BYTES_F64, NATIVE_ORDER).sub(vMean);
            vTail = d.fma(d, vTail);
        }
        total += vTail.reduceLanes(VectorOperators.ADD);

        for (; i < end; i++) {
            double diff = seg.getAtIndex(ValueLayout.JAVA_DOUBLE, i) - m;
            total += diff * diff;
        }
        return total;
    }

    public static double varInt(NDArray a, int ddof) {
        long size = a.getSize();
        if (size == 0) return Double.NaN;
        double divisor = (double) (size - ddof);
        if (divisor <= 0) return Double.NaN;
        if (size == 1) return 0.0;

        double m = Mean.meanInt(a);
        double sumSq = 0.0;
        if (!a.isContiguous()) {
            NDIter iter = new NDIter(a.internalShapeUnsafe());
            while (iter.hasNext) {
                long byteOffset = ShapeUtil.getByteOffset(iter.coords, a.internalStridesUnsafe(), a.getDType());
                double diff = (double) a.getData().get(ValueLayout.JAVA_INT, byteOffset) - m;
                sumSq += diff * diff;
                iter.next();
            }
            return sumSq / divisor;
        }

        MemorySegment seg = a.getData();
        for (long i = 0; i < size; i++) {
            double diff = (double) seg.getAtIndex(ValueLayout.JAVA_INT, i) - m;
            sumSq += diff * diff;
        }
        return sumSq / divisor;
    }

    // =========================================================================
    // AXIS VARIANCE
    // =========================================================================

    public static NDArray varFloatAxis(NDArray a, int axis, int ddof, NDArray resArray) {
        long size = a.internalShapeUnsafe()[axis];
        long strideA = a.internalStridesUnsafe()[axis];
        long resSize = resArray.getSize();
        double divisor = (double) (size - ddof);
        float invDivisor = divisor <= 0 ? Float.NaN : (float) (1.0 / divisor);
        float invSize = 1.0f / (float) size;

        if (resSize * size >= PARALLEL_ELEMENT_THRESHOLD && AVAILABLE_CORES > 1) {
            int nChunks = Math.min(AVAILABLE_CORES, (int) Math.max(1, (resSize * size) / PARALLEL_ELEMENT_THRESHOLD));
            long chunkSize = (resSize + nChunks - 1) / nChunks;
            RecursiveAction[] tasks = new RecursiveAction[nChunks];
            for (int t = 0; t < nChunks; t++) {
                final long start = t * chunkSize;
                final long end = Math.min(start + chunkSize, resSize);
                tasks[t] = new RecursiveAction() {
                    @Override
                    protected void compute() {
                        computeVarFloatAxisRange(a, axis, size, strideA, invSize, invDivisor, resArray, start, end);
                    }
                };
            }
            POOL.invoke(new RecursiveAction() {
                @Override
                protected void compute() {
                    invokeAll(tasks);
                }
            });
            return resArray;
        }

        computeVarFloatAxisRange(a, axis, size, strideA, invSize, invDivisor, resArray, 0, resSize);
        return resArray;
    }

    private static void computeVarFloatAxisRange(NDArray a, int axis, long size, long strideA, float invSize,
                                                 float invDivisor, NDArray resArray, long start, long end) {
        MemorySegment aSeg = a.getData();
        MemorySegment resSeg = resArray.getData();

        for (long i = start; i < end; i++) {
            long tempIndex = i;
            long baseOffset = 0;
            long offsetRes = 0;
            for (int d = resArray.internalShapeUnsafe().length - 1; d >= 0; d--) {
                long coord = tempIndex % resArray.internalShapeUnsafe()[d];
                tempIndex /= resArray.internalShapeUnsafe()[d];
                offsetRes += coord * resArray.internalStridesUnsafe()[d];

                int aDim = (d >= axis) ? d + 1 : d;
                baseOffset += coord * a.internalStridesUnsafe()[aDim];
            }

            // Pass 1: compute mean
            float sum = 0f;
            if (strideA == 1) {
                var vAcc = FloatVector.zero(SPECIES_F32);
                long k = 0;
                long loopbound = SPECIES_F32.loopBound(size);
                for (; k < loopbound; k += VL_F32) {
                    vAcc = vAcc.add(FloatVector.fromMemorySegment(SPECIES_F32, aSeg, (baseOffset + k) * BYTES_F32, NATIVE_ORDER));
                }
                sum = vAcc.reduceLanes(VectorOperators.ADD);
                for (; k < size; k++) {
                    sum += aSeg.getAtIndex(ValueLayout.JAVA_FLOAT, baseOffset + k);
                }
            } else {
                for (long k = 0; k < size; k++) {
                    sum += aSeg.getAtIndex(ValueLayout.JAVA_FLOAT, baseOffset + k * strideA);
                }
            }
            float m = sum * invSize;

            // Pass 2: compute sum of squared diffs
            float sumSq = 0f;
            if (strideA == 1) {
                FloatVector vM = FloatVector.broadcast(SPECIES_F32, m);
                var vSumSq = FloatVector.zero(SPECIES_F32);
                long k = 0;
                long loopbound = SPECIES_F32.loopBound(size);
                for (; k < loopbound; k += VL_F32) {
                    var diff = FloatVector.fromMemorySegment(SPECIES_F32, aSeg, (baseOffset + k) * BYTES_F32, NATIVE_ORDER).sub(vM);
                    vSumSq = diff.fma(diff, vSumSq);
                }
                sumSq = vSumSq.reduceLanes(VectorOperators.ADD);
                for (; k < size; k++) {
                    float diff = aSeg.getAtIndex(ValueLayout.JAVA_FLOAT, baseOffset + k) - m;
                    sumSq += diff * diff;
                }
            } else {
                for (long k = 0; k < size; k++) {
                    float diff = aSeg.getAtIndex(ValueLayout.JAVA_FLOAT, baseOffset + k * strideA) - m;
                    sumSq += diff * diff;
                }
            }

            resSeg.setAtIndex(ValueLayout.JAVA_FLOAT, offsetRes, sumSq * invDivisor);
        }
    }

    public static NDArray varDoubleAxis(NDArray a, int axis, int ddof, NDArray resArray) {
        long size = a.internalShapeUnsafe()[axis];
        long strideA = a.internalStridesUnsafe()[axis];
        long resSize = resArray.getSize();
        double divisor = (double) (size - ddof);
        double invDivisor = divisor <= 0 ? Double.NaN : 1.0 / divisor;
        double invSize = 1.0 / (double) size;

        if (resSize * size >= PARALLEL_ELEMENT_THRESHOLD && AVAILABLE_CORES > 1) {
            int nChunks = Math.min(AVAILABLE_CORES, (int) Math.max(1, (resSize * size) / PARALLEL_ELEMENT_THRESHOLD));
            long chunkSize = (resSize + nChunks - 1) / nChunks;
            RecursiveAction[] tasks = new RecursiveAction[nChunks];
            for (int t = 0; t < nChunks; t++) {
                final long start = t * chunkSize;
                final long end = Math.min(start + chunkSize, resSize);
                tasks[t] = new RecursiveAction() {
                    @Override
                    protected void compute() {
                        computeVarDoubleAxisRange(a, axis, size, strideA, invSize, invDivisor, resArray, start, end);
                    }
                };
            }
            POOL.invoke(new RecursiveAction() {
                @Override
                protected void compute() {
                    invokeAll(tasks);
                }
            });
            return resArray;
        }

        computeVarDoubleAxisRange(a, axis, size, strideA, invSize, invDivisor, resArray, 0, resSize);
        return resArray;
    }

    private static void computeVarDoubleAxisRange(NDArray a, int axis, long size, long strideA, double invSize,
                                                  double invDivisor, NDArray resArray, long start, long end) {
        MemorySegment aSeg = a.getData();
        MemorySegment resSeg = resArray.getData();

        for (long i = start; i < end; i++) {
            long tempIndex = i;
            long baseOffset = 0;
            long offsetRes = 0;
            for (int d = resArray.internalShapeUnsafe().length - 1; d >= 0; d--) {
                long coord = tempIndex % resArray.internalShapeUnsafe()[d];
                tempIndex /= resArray.internalShapeUnsafe()[d];
                offsetRes += coord * resArray.internalStridesUnsafe()[d];

                int aDim = (d >= axis) ? d + 1 : d;
                baseOffset += coord * a.internalStridesUnsafe()[aDim];
            }

            // Pass 1: mean
            double sum = 0.0;
            if (strideA == 1) {
                var vAcc = DoubleVector.zero(SPECIES_F64);
                long k = 0;
                long loopbound = SPECIES_F64.loopBound(size);
                for (; k < loopbound; k += VL_F64) {
                    vAcc = vAcc.add(DoubleVector.fromMemorySegment(SPECIES_F64, aSeg, (baseOffset + k) * BYTES_F64, NATIVE_ORDER));
                }
                sum = vAcc.reduceLanes(VectorOperators.ADD);
                for (; k < size; k++) {
                    sum += aSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, baseOffset + k);
                }
            } else {
                for (long k = 0; k < size; k++) {
                    sum += aSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, baseOffset + k * strideA);
                }
            }
            double m = sum * invSize;

            // Pass 2: sum of squared diffs
            double sumSq = 0.0;
            if (strideA == 1) {
                DoubleVector vM = DoubleVector.broadcast(SPECIES_F64, m);
                var vSumSq = DoubleVector.zero(SPECIES_F64);
                long k = 0;
                long loopbound = SPECIES_F64.loopBound(size);
                for (; k < loopbound; k += VL_F64) {
                    var diff = DoubleVector.fromMemorySegment(SPECIES_F64, aSeg, (baseOffset + k) * BYTES_F64, NATIVE_ORDER).sub(vM);
                    vSumSq = diff.fma(diff, vSumSq);
                }
                sumSq = vSumSq.reduceLanes(VectorOperators.ADD);
                for (; k < size; k++) {
                    double diff = aSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, baseOffset + k) - m;
                    sumSq += diff * diff;
                }
            } else {
                for (long k = 0; k < size; k++) {
                    double diff = aSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, baseOffset + k * strideA) - m;
                    sumSq += diff * diff;
                }
            }

            resSeg.setAtIndex(ValueLayout.JAVA_DOUBLE, offsetRes, sumSq * invDivisor);
        }
    }

    public static NDArray varIntAxis(NDArray a, int axis, int ddof, NDArray resArray) {
        long size = a.internalShapeUnsafe()[axis];
        long strideA = a.internalStridesUnsafe()[axis];
        long resSize = resArray.getSize();
        double divisor = (double) (size - ddof);
        double invDivisor = divisor <= 0 ? Double.NaN : 1.0 / divisor;
        double invSize = 1.0 / (double) size;

        MemorySegment aSeg = a.getData();
        MemorySegment resSeg = resArray.getData();

        for (long i = 0; i < resSize; i++) {
            long tempIndex = i;
            long baseOffset = 0;
            long offsetRes = 0;
            for (int d = resArray.internalShapeUnsafe().length - 1; d >= 0; d--) {
                long coord = tempIndex % resArray.internalShapeUnsafe()[d];
                tempIndex /= resArray.internalShapeUnsafe()[d];
                offsetRes += coord * resArray.internalStridesUnsafe()[d];

                int aDim = (d >= axis) ? d + 1 : d;
                baseOffset += coord * a.internalStridesUnsafe()[aDim];
            }

            double sum = 0.0;
            for (long k = 0; k < size; k++) {
                sum += (double) aSeg.getAtIndex(ValueLayout.JAVA_INT, baseOffset + k * strideA);
            }
            double m = sum * invSize;

            double sumSq = 0.0;
            for (long k = 0; k < size; k++) {
                double diff = (double) aSeg.getAtIndex(ValueLayout.JAVA_INT, baseOffset + k * strideA) - m;
                sumSq += diff * diff;
            }

            resSeg.setAtIndex(ValueLayout.JAVA_DOUBLE, offsetRes, sumSq * invDivisor);
        }
        return resArray;
    }
}
