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
package jnum.internal.kernel.stat;

import static jnum.internal.Constants.*;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.concurrent.RecursiveAction;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jnum.NDArray;
import jnum.internal.layout.NDIter;
import jnum.internal.layout.ShapeUtil;
import jnum.internal.ops.StatOps;

/**
 * Numerically stable, SIMD-accelerated covariance kernels.
 * Supports scalar pairwise covariance (1D vectors) and full 2D covariance matrices.
 */
public final class Cov {

    private Cov() {
        throw new AssertionError("Cov kernel cannot be instantiated.");
    }

    // =========================================================================
    // SCALAR PAIRWISE COVARIANCE
    // =========================================================================

    public static double covFloat(NDArray x, NDArray y, int ddof) {
        long size = x.getSize();
        if (size != y.getSize()) {
            throw new IllegalArgumentException("Arrays must have the same size for covariance: " +
                    size + " vs " + y.getSize());
        }
        if (size == 0) return Double.NaN;
        double divisor = (double) (size - ddof);
        if (divisor <= 0) return Double.NaN;
        if (size == 1) return 0.0;

        double mx = Mean.meanFloat(x);
        double my = Mean.meanFloat(y);

        if (!x.isContiguous() || !y.isContiguous()) {
            double sum = 0.0;
            NDIter iterX = new NDIter(x.internalShapeUnsafe());
            NDIter iterY = new NDIter(y.internalShapeUnsafe());
            MemorySegment segX = x.getData();
            MemorySegment segY = y.getData();

            while (iterX.hasNext && iterY.hasNext) {
                long offX = ShapeUtil.getByteOffset(iterX.coords, x.internalStridesUnsafe(), x.getDType());
                long offY = ShapeUtil.getByteOffset(iterY.coords, y.internalStridesUnsafe(), y.getDType());
                double dx = (double) segX.get(ValueLayout.JAVA_FLOAT, offX) - mx;
                double dy = (double) segY.get(ValueLayout.JAVA_FLOAT, offY) - my;
                sum += dx * dy;
                iterX.next();
                iterY.next();
            }
            return sum / divisor;
        }

        if (size >= PARALLEL_ELEMENT_THRESHOLD && AVAILABLE_CORES > 1) {
            int nChunks = Math.min(AVAILABLE_CORES, (int) Math.max(1, size / PARALLEL_ELEMENT_THRESHOLD));
            long chunkSize = (size + nChunks - 1) / nChunks;
            double[] partials = new double[nChunks];
            RecursiveAction[] tasks = new RecursiveAction[nChunks];
            MemorySegment segX = x.getData();
            MemorySegment segY = y.getData();

            for (int t = 0; t < nChunks; t++) {
                final int idx = t;
                final long start = t * chunkSize;
                final long end = Math.min(start + chunkSize, size);
                tasks[t] = new RecursiveAction() {
                    @Override
                    protected void compute() {
                        partials[idx] = computeCovFloatChunk(segX, segY, mx, my, start, end);
                    }
                };
            }
            POOL.invoke(new RecursiveAction() {
                @Override
                protected void compute() {
                    invokeAll(tasks);
                }
            });
            double sum = 0.0;
            for (double p : partials) sum += p;
            return sum / divisor;
        }

        return computeCovFloatChunk(x.getData(), y.getData(), mx, my, 0, size) / divisor;
    }

    private static double computeCovFloatChunk(MemorySegment segX, MemorySegment segY, double mx, double my, long start, long end) {
        long len = end - start;
        long loopbound4 = start + ((len) / (VL_F32 * 4)) * (VL_F32 * 4);
        FloatVector vMx = FloatVector.broadcast(SPECIES_F32, (float) mx);
        FloatVector vMy = FloatVector.broadcast(SPECIES_F32, (float) my);
        var vSum1 = FloatVector.zero(SPECIES_F32);
        var vSum2 = FloatVector.zero(SPECIES_F32);
        var vSum3 = FloatVector.zero(SPECIES_F32);
        var vSum4 = FloatVector.zero(SPECIES_F32);

        long i = start;
        for (; i < loopbound4; i += VL_F32 * 4) {
            var dx1 = FloatVector.fromMemorySegment(SPECIES_F32, segX, i * 4L, NATIVE_ORDER).sub(vMx);
            var dy1 = FloatVector.fromMemorySegment(SPECIES_F32, segY, i * 4L, NATIVE_ORDER).sub(vMy);
            var dx2 = FloatVector.fromMemorySegment(SPECIES_F32, segX, (i + VL_F32) * 4L, NATIVE_ORDER).sub(vMx);
            var dy2 = FloatVector.fromMemorySegment(SPECIES_F32, segY, (i + VL_F32) * 4L, NATIVE_ORDER).sub(vMy);
            var dx3 = FloatVector.fromMemorySegment(SPECIES_F32, segX, (i + VL_F32 * 2) * 4L, NATIVE_ORDER).sub(vMx);
            var dy3 = FloatVector.fromMemorySegment(SPECIES_F32, segY, (i + VL_F32 * 2) * 4L, NATIVE_ORDER).sub(vMy);
            var dx4 = FloatVector.fromMemorySegment(SPECIES_F32, segX, (i + VL_F32 * 3) * 4L, NATIVE_ORDER).sub(vMx);
            var dy4 = FloatVector.fromMemorySegment(SPECIES_F32, segY, (i + VL_F32 * 3) * 4L, NATIVE_ORDER).sub(vMy);

            vSum1 = dx1.fma(dy1, vSum1);
            vSum2 = dx2.fma(dy2, vSum2);
            vSum3 = dx3.fma(dy3, vSum3);
            vSum4 = dx4.fma(dy4, vSum4);
        }

        double total = vSum1.add(vSum2).add(vSum3).add(vSum4).reduceLanes(VectorOperators.ADD);

        long loopbound = start + SPECIES_F32.loopBound(len);
        for (; i < loopbound; i += VL_F32) {
            var dx = FloatVector.fromMemorySegment(SPECIES_F32, segX, i * 4L, NATIVE_ORDER).sub(vMx);
            var dy = FloatVector.fromMemorySegment(SPECIES_F32, segY, i * 4L, NATIVE_ORDER).sub(vMy);
            total += dx.mul(dy).reduceLanes(VectorOperators.ADD);
        }

        for (; i < end; i++) {
            double dx = (double) segX.getAtIndex(ValueLayout.JAVA_FLOAT, i) - mx;
            double dy = (double) segY.getAtIndex(ValueLayout.JAVA_FLOAT, i) - my;
            total += dx * dy;
        }

        return total;
    }

    public static double covDouble(NDArray x, NDArray y, int ddof) {
        long size = x.getSize();
        if (size != y.getSize()) {
            throw new IllegalArgumentException("Arrays must have the same size for covariance: " +
                    size + " vs " + y.getSize());
        }
        if (size == 0) return Double.NaN;
        double divisor = (double) (size - ddof);
        if (divisor <= 0) return Double.NaN;
        if (size == 1) return 0.0;

        double mx = Mean.meanDouble(x);
        double my = Mean.meanDouble(y);

        if (!x.isContiguous() || !y.isContiguous()) {
            double sum = 0.0;
            NDIter iterX = new NDIter(x.internalShapeUnsafe());
            NDIter iterY = new NDIter(y.internalShapeUnsafe());
            MemorySegment segX = x.getData();
            MemorySegment segY = y.getData();

            while (iterX.hasNext && iterY.hasNext) {
                long offX = ShapeUtil.getByteOffset(iterX.coords, x.internalStridesUnsafe(), x.getDType());
                long offY = ShapeUtil.getByteOffset(iterY.coords, y.internalStridesUnsafe(), y.getDType());
                double dx = segX.get(ValueLayout.JAVA_DOUBLE, offX) - mx;
                double dy = segY.get(ValueLayout.JAVA_DOUBLE, offY) - my;
                sum += dx * dy;
                iterX.next();
                iterY.next();
            }
            return sum / divisor;
        }

        if (size >= PARALLEL_ELEMENT_THRESHOLD && AVAILABLE_CORES > 1) {
            int nChunks = Math.min(AVAILABLE_CORES, (int) Math.max(1, size / PARALLEL_ELEMENT_THRESHOLD));
            long chunkSize = (size + nChunks - 1) / nChunks;
            double[] partials = new double[nChunks];
            RecursiveAction[] tasks = new RecursiveAction[nChunks];
            MemorySegment segX = x.getData();
            MemorySegment segY = y.getData();

            for (int t = 0; t < nChunks; t++) {
                final int idx = t;
                final long start = t * chunkSize;
                final long end = Math.min(start + chunkSize, size);
                tasks[t] = new RecursiveAction() {
                    @Override
                    protected void compute() {
                        partials[idx] = computeCovDoubleChunk(segX, segY, mx, my, start, end);
                    }
                };
            }
            POOL.invoke(new RecursiveAction() {
                @Override
                protected void compute() {
                    invokeAll(tasks);
                }
            });
            double sum = 0.0;
            for (double p : partials) sum += p;
            return sum / divisor;
        }

        return computeCovDoubleChunk(x.getData(), y.getData(), mx, my, 0, size) / divisor;
    }

    private static double computeCovDoubleChunk(MemorySegment segX, MemorySegment segY, double mx, double my, long start, long end) {
        long len = end - start;
        long loopbound4 = start + ((len) / (VL_F64 * 4)) * (VL_F64 * 4);
        DoubleVector vMx = DoubleVector.broadcast(SPECIES_F64, mx);
        DoubleVector vMy = DoubleVector.broadcast(SPECIES_F64, my);
        var vSum1 = DoubleVector.zero(SPECIES_F64);
        var vSum2 = DoubleVector.zero(SPECIES_F64);
        var vSum3 = DoubleVector.zero(SPECIES_F64);
        var vSum4 = DoubleVector.zero(SPECIES_F64);

        long i = start;
        for (; i < loopbound4; i += VL_F64 * 4) {
            var dx1 = DoubleVector.fromMemorySegment(SPECIES_F64, segX, i * 8L, NATIVE_ORDER).sub(vMx);
            var dy1 = DoubleVector.fromMemorySegment(SPECIES_F64, segY, i * 8L, NATIVE_ORDER).sub(vMy);
            var dx2 = DoubleVector.fromMemorySegment(SPECIES_F64, segX, (i + VL_F64) * 8L, NATIVE_ORDER).sub(vMx);
            var dy2 = DoubleVector.fromMemorySegment(SPECIES_F64, segY, (i + VL_F64) * 8L, NATIVE_ORDER).sub(vMy);
            var dx3 = DoubleVector.fromMemorySegment(SPECIES_F64, segX, (i + VL_F64 * 2) * 8L, NATIVE_ORDER).sub(vMx);
            var dy3 = DoubleVector.fromMemorySegment(SPECIES_F64, segY, (i + VL_F64 * 2) * 8L, NATIVE_ORDER).sub(vMy);
            var dx4 = DoubleVector.fromMemorySegment(SPECIES_F64, segX, (i + VL_F64 * 3) * 8L, NATIVE_ORDER).sub(vMx);
            var dy4 = DoubleVector.fromMemorySegment(SPECIES_F64, segY, (i + VL_F64 * 3) * 8L, NATIVE_ORDER).sub(vMy);

            vSum1 = dx1.fma(dy1, vSum1);
            vSum2 = dx2.fma(dy2, vSum2);
            vSum3 = dx3.fma(dy3, vSum3);
            vSum4 = dx4.fma(dy4, vSum4);
        }

        double total = vSum1.add(vSum2).add(vSum3).add(vSum4).reduceLanes(VectorOperators.ADD);

        long loopbound = start + SPECIES_F64.loopBound(len);
        for (; i < loopbound; i += VL_F64) {
            var dx = DoubleVector.fromMemorySegment(SPECIES_F64, segX, i * 8L, NATIVE_ORDER).sub(vMx);
            var dy = DoubleVector.fromMemorySegment(SPECIES_F64, segY, i * 8L, NATIVE_ORDER).sub(vMy);
            total += dx.mul(dy).reduceLanes(VectorOperators.ADD);
        }

        for (; i < end; i++) {
            double dx = segX.getAtIndex(ValueLayout.JAVA_DOUBLE, i) - mx;
            double dy = segY.getAtIndex(ValueLayout.JAVA_DOUBLE, i) - my;
            total += dx * dy;
        }

        return total;
    }

    public static double covInt(NDArray x, NDArray y, int ddof) {
        long size = x.getSize();
        if (size != y.getSize()) {
            throw new IllegalArgumentException("Arrays must have the same size for covariance: " +
                    size + " vs " + y.getSize());
        }
        if (size == 0) return Double.NaN;
        double divisor = (double) (size - ddof);
        if (divisor <= 0) return Double.NaN;
        if (size == 1) return 0.0;

        double mx = Mean.meanInt(x);
        double my = Mean.meanInt(y);

        double sum = 0.0;
        if (x.isContiguous() && y.isContiguous()) {
            MemorySegment segX = x.getData();
            MemorySegment segY = y.getData();
            for (long i = 0; i < size; i++) {
                double dx = (double) segX.getAtIndex(ValueLayout.JAVA_INT, i) - mx;
                double dy = (double) segY.getAtIndex(ValueLayout.JAVA_INT, i) - my;
                sum += dx * dy;
            }
        } else {
            NDIter iterX = new NDIter(x.internalShapeUnsafe());
            NDIter iterY = new NDIter(y.internalShapeUnsafe());
            MemorySegment segX = x.getData();
            MemorySegment segY = y.getData();
            while (iterX.hasNext && iterY.hasNext) {
                long offX = ShapeUtil.getByteOffset(iterX.coords, x.internalStridesUnsafe(), x.getDType());
                long offY = ShapeUtil.getByteOffset(iterY.coords, y.internalStridesUnsafe(), y.getDType());
                double dx = (double) segX.get(ValueLayout.JAVA_INT, offX) - mx;
                double dy = (double) segY.get(ValueLayout.JAVA_INT, offY) - my;
                sum += dx * dy;
                iterX.next();
                iterY.next();
            }
        }
        return sum / divisor;
    }

    // =========================================================================
    // 2D COVARIANCE MATRIX
    // =========================================================================

    public static NDArray covMatrix(NDArray m, boolean rowvar, int ddof, Arena arena) {
        if (m.dim() != 2) {
            throw new IllegalArgumentException("Covariance matrix input must be a 2D NDArray, but got dim: " + m.dim());
        }

        long[] shape = m.internalShapeUnsafe();
        long vars = rowvar ? shape[0] : shape[1];
        long observations = rowvar ? shape[1] : shape[0];

        if (observations - ddof <= 0) {
            throw new IllegalArgumentException("Degrees of freedom (" + (observations - ddof) + ") must be > 0");
        }

        int axis = rowvar ? 1 : 0;
        NDArray mu = StatOps.mean(m, axis, true, arena);
        NDArray xc = m.sub(mu);

        NDArray prod;
        if (rowvar) {
            prod = xc.matmul(xc.transpose());
        } else {
            prod = xc.transpose().matmul(xc);
        }

        if (m.getDType() == jnum.DType.f32) {
            return prod.div((float) (observations - ddof));
        } else {
            return prod.div((double) (observations - ddof));
        }
    }
}
