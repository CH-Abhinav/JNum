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

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.concurrent.RecursiveAction;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorOperators;
import jnum.NDArray;
import jnum.internal.layout.NDIter;
import jnum.internal.layout.ShapeUtil;

/**
 * Hardware-accelerated, numerically stable kernels for arithmetic mean.
 * Utilizes Panama Vector API SIMD and ForkJoin parallelism.
 */
public final class Mean {

    private Mean() {
        throw new AssertionError("Mean kernel cannot be instantiated.");
    }

    // =========================================================================
    // SCALAR MEAN
    // =========================================================================

    public static double meanFloat(NDArray a) {
        long size = a.getSize();
        if (size == 0) return Double.NaN;
        if (!a.isContiguous()) {
            double total = 0.0;
            NDIter iter = new NDIter(a.internalShapeUnsafe());
            while (iter.hasNext) {
                long byteOffset = ShapeUtil.getByteOffset(iter.coords, a.internalStridesUnsafe(), a.getDType());
                total += a.getData().get(ValueLayout.JAVA_FLOAT, byteOffset);
                iter.next();
            }
            return total / (double) size;
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
                        partials[idx] = computeFloatChunk(seg, start, end);
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
            return sum / (double) size;
        }

        return computeFloatChunk(a.getData(), 0, size) / (double) size;
    }

    private static double computeFloatChunk(MemorySegment seg, long start, long end) {
        long len = end - start;
        long loopbound4 = start + ((len) / (VL_F32 * 4)) * (VL_F32 * 4);
        var vSum1 = FloatVector.zero(SPECIES_F32);
        var vSum2 = FloatVector.zero(SPECIES_F32);
        var vSum3 = FloatVector.zero(SPECIES_F32);
        var vSum4 = FloatVector.zero(SPECIES_F32);

        long i = start;
        for (; i < loopbound4; i += VL_F32 * 4) {
            vSum1 = vSum1.add(FloatVector.fromMemorySegment(SPECIES_F32, seg, i * BYTES_F32, NATIVE_ORDER));
            vSum2 = vSum2.add(FloatVector.fromMemorySegment(SPECIES_F32, seg, (i + VL_F32) * BYTES_F32, NATIVE_ORDER));
            vSum3 = vSum3.add(FloatVector.fromMemorySegment(SPECIES_F32, seg, (i + VL_F32 * 2) * BYTES_F32, NATIVE_ORDER));
            vSum4 = vSum4.add(FloatVector.fromMemorySegment(SPECIES_F32, seg, (i + VL_F32 * 3) * BYTES_F32, NATIVE_ORDER));
        }

        var vCombined = vSum1.add(vSum2).add(vSum3).add(vSum4);
        double total = (double) vCombined.reduceLanes(VectorOperators.ADD);

        long loopbound = i + SPECIES_F32.loopBound(end - i);
        var vTail = FloatVector.zero(SPECIES_F32);
        for (; i < loopbound; i += VL_F32) {
            vTail = vTail.add(FloatVector.fromMemorySegment(SPECIES_F32, seg, i * BYTES_F32, NATIVE_ORDER));
        }
        total += (double) vTail.reduceLanes(VectorOperators.ADD);

        for (; i < end; i++) {
            total += seg.getAtIndex(ValueLayout.JAVA_FLOAT, i);
        }
        return total;
    }

    public static double meanDouble(NDArray a) {
        long size = a.getSize();
        if (size == 0) return Double.NaN;
        if (!a.isContiguous()) {
            double total = 0.0;
            NDIter iter = new NDIter(a.internalShapeUnsafe());
            while (iter.hasNext) {
                long byteOffset = ShapeUtil.getByteOffset(iter.coords, a.internalStridesUnsafe(), a.getDType());
                total += a.getData().get(ValueLayout.JAVA_DOUBLE, byteOffset);
                iter.next();
            }
            return total / (double) size;
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
                        partials[idx] = computeDoubleChunk(seg, start, end);
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
            return sum / (double) size;
        }

        return computeDoubleChunk(a.getData(), 0, size) / (double) size;
    }

    private static double computeDoubleChunk(MemorySegment seg, long start, long end) {
        long len = end - start;
        long loopbound4 = start + ((len) / (VL_F64 * 4)) * (VL_F64 * 4);
        var vSum1 = DoubleVector.zero(SPECIES_F64);
        var vSum2 = DoubleVector.zero(SPECIES_F64);
        var vSum3 = DoubleVector.zero(SPECIES_F64);
        var vSum4 = DoubleVector.zero(SPECIES_F64);

        long i = start;
        for (; i < loopbound4; i += VL_F64 * 4) {
            vSum1 = vSum1.add(DoubleVector.fromMemorySegment(SPECIES_F64, seg, i * BYTES_F64, NATIVE_ORDER));
            vSum2 = vSum2.add(DoubleVector.fromMemorySegment(SPECIES_F64, seg, (i + VL_F64) * BYTES_F64, NATIVE_ORDER));
            vSum3 = vSum3.add(DoubleVector.fromMemorySegment(SPECIES_F64, seg, (i + VL_F64 * 2) * BYTES_F64, NATIVE_ORDER));
            vSum4 = vSum4.add(DoubleVector.fromMemorySegment(SPECIES_F64, seg, (i + VL_F64 * 3) * BYTES_F64, NATIVE_ORDER));
        }

        var vCombined = vSum1.add(vSum2).add(vSum3).add(vSum4);
        double total = vCombined.reduceLanes(VectorOperators.ADD);

        long loopbound = i + SPECIES_F64.loopBound(end - i);
        var vTail = DoubleVector.zero(SPECIES_F64);
        for (; i < loopbound; i += VL_F64) {
            vTail = vTail.add(DoubleVector.fromMemorySegment(SPECIES_F64, seg, i * BYTES_F64, NATIVE_ORDER));
        }
        total += vTail.reduceLanes(VectorOperators.ADD);

        for (; i < end; i++) {
            total += seg.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
        }
        return total;
    }

    public static double meanInt(NDArray a) {
        long size = a.getSize();
        if (size == 0) return Double.NaN;
        if (!a.isContiguous()) {
            double total = 0.0;
            NDIter iter = new NDIter(a.internalShapeUnsafe());
            while (iter.hasNext) {
                long byteOffset = ShapeUtil.getByteOffset(iter.coords, a.internalStridesUnsafe(), a.getDType());
                total += a.getData().get(ValueLayout.JAVA_INT, byteOffset);
                iter.next();
            }
            return total / (double) size;
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
                        partials[idx] = computeIntChunk(seg, start, end);
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
            return sum / (double) size;
        }

        return computeIntChunk(a.getData(), 0, size) / (double) size;
    }

    private static double computeIntChunk(MemorySegment seg, long start, long end) {
        long len = end - start;
        long loopbound = start + SPECIES_I32.loopBound(len);
        var vSum = IntVector.zero(SPECIES_I32);
        long i = start;
        double total = 0.0;
        long blockStep = 1024L;

        while (i < loopbound) {
            long blockEnd = Math.min(i + blockStep, loopbound);
            for (; i < blockEnd; i += VL_I32) {
                vSum = vSum.add(IntVector.fromMemorySegment(SPECIES_I32, seg, i * BYTES_I32, NATIVE_ORDER));
            }
            total += (double) vSum.reduceLanes(VectorOperators.ADD);
            vSum = IntVector.zero(SPECIES_I32);
        }

        for (; i < end; i++) {
            total += (double) seg.getAtIndex(ValueLayout.JAVA_INT, i);
        }
        return total;
    }

    public static double meanBool(NDArray a) {
        long size = a.getSize();
        if (size == 0) return Double.NaN;
        long count = 0;
        if (!a.isContiguous()) {
            NDIter iter = new NDIter(a.internalShapeUnsafe());
            while (iter.hasNext) {
                long byteOffset = ShapeUtil.getByteOffset(iter.coords, a.internalStridesUnsafe(), a.getDType());
                if (a.getData().get(ValueLayout.JAVA_BYTE, byteOffset) != 0) {
                    count++;
                }
                iter.next();
            }
            return (double) count / (double) size;
        }
        MemorySegment seg = a.getData();
        for (long i = 0; i < size; i++) {
            if (seg.getAtIndex(ValueLayout.JAVA_BYTE, i) != 0) {
                count++;
            }
        }
        return (double) count / (double) size;
    }

    // =========================================================================
    // AXIS MEAN
    // =========================================================================

    public static NDArray meanFloatAxis(NDArray a, int axis, NDArray resArray) {
        long size = a.internalShapeUnsafe()[axis];
        long strideA = a.internalStridesUnsafe()[axis];
        long resSize = resArray.getSize();
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
                        computeFloatAxisRange(a, axis, size, strideA, invSize, resArray, start, end);
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

        computeFloatAxisRange(a, axis, size, strideA, invSize, resArray, 0, resSize);
        return resArray;
    }

    private static void computeFloatAxisRange(NDArray a, int axis, long size, long strideA, float invSize,
                                              NDArray resArray, long start, long end) {
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

            if (strideA == 1) {
                var vAcc = FloatVector.zero(SPECIES_F32);
                long k = 0;
                long loopbound = SPECIES_F32.loopBound(size);
                for (; k < loopbound; k += VL_F32) {
                    var vVal = FloatVector.fromMemorySegment(SPECIES_F32, aSeg, (baseOffset + k) * BYTES_F32, NATIVE_ORDER);
                    vAcc = vAcc.add(vVal);
                }
                float acc = vAcc.reduceLanes(VectorOperators.ADD);
                for (; k < size; k++) {
                    acc += aSeg.getAtIndex(ValueLayout.JAVA_FLOAT, baseOffset + k);
                }
                resSeg.setAtIndex(ValueLayout.JAVA_FLOAT, offsetRes, acc * invSize);
            } else {
                float acc = 0f;
                for (long k = 0; k < size; k++) {
                    acc += aSeg.getAtIndex(ValueLayout.JAVA_FLOAT, baseOffset + k * strideA);
                }
                resSeg.setAtIndex(ValueLayout.JAVA_FLOAT, offsetRes, acc * invSize);
            }
        }
    }

    public static NDArray meanDoubleAxis(NDArray a, int axis, NDArray resArray) {
        long size = a.internalShapeUnsafe()[axis];
        long strideA = a.internalStridesUnsafe()[axis];
        long resSize = resArray.getSize();
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
                        computeDoubleAxisRange(a, axis, size, strideA, invSize, resArray, start, end);
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

        computeDoubleAxisRange(a, axis, size, strideA, invSize, resArray, 0, resSize);
        return resArray;
    }

    private static void computeDoubleAxisRange(NDArray a, int axis, long size, long strideA, double invSize,
                                               NDArray resArray, long start, long end) {
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

            if (strideA == 1) {
                var vAcc = DoubleVector.zero(SPECIES_F64);
                long k = 0;
                long loopbound = SPECIES_F64.loopBound(size);
                for (; k < loopbound; k += VL_F64) {
                    var vVal = DoubleVector.fromMemorySegment(SPECIES_F64, aSeg, (baseOffset + k) * BYTES_F64, NATIVE_ORDER);
                    vAcc = vAcc.add(vVal);
                }
                double acc = vAcc.reduceLanes(VectorOperators.ADD);
                for (; k < size; k++) {
                    acc += aSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, baseOffset + k);
                }
                resSeg.setAtIndex(ValueLayout.JAVA_DOUBLE, offsetRes, acc * invSize);
            } else {
                double acc = 0.0;
                for (long k = 0; k < size; k++) {
                    acc += aSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, baseOffset + k * strideA);
                }
                resSeg.setAtIndex(ValueLayout.JAVA_DOUBLE, offsetRes, acc * invSize);
            }
        }
    }

    public static NDArray meanIntAxis(NDArray a, int axis, NDArray resArray) {
        long size = a.internalShapeUnsafe()[axis];
        long strideA = a.internalStridesUnsafe()[axis];
        long resSize = resArray.getSize();
        double invSize = 1.0 / (double) size;

        computeIntAxisRange(a, axis, size, strideA, invSize, resArray, 0, resSize);
        return resArray;
    }

    private static void computeIntAxisRange(NDArray a, int axis, long size, long strideA, double invSize,
                                            NDArray resArray, long start, long end) {
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

            double acc = 0.0;
            for (long k = 0; k < size; k++) {
                acc += (double) aSeg.getAtIndex(ValueLayout.JAVA_INT, baseOffset + k * strideA);
            }
            resSeg.setAtIndex(ValueLayout.JAVA_DOUBLE, offsetRes, acc * invSize);
        }
    }
}
