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

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jnum.NDArray;
import jnum.internal.layout.NDIter;
import jnum.internal.layout.ShapeUtil;

/**
 * High-performance, SIMD-accelerated cumulative product (cumprod) kernels.
 * Zero-intermediate allocations with cache-aligned sequential memory traversals.
 */
public final class CumProd {

    private CumProd() {
        throw new AssertionError("CumProd kernel cannot be instantiated.");
    }

    // =========================================================================
    // FLAT 1D CUMPROD
    // =========================================================================

    public static NDArray cumprodFloatFlat(NDArray a, NDArray res) {
        long total = a.getSize();
        if (total == 0) return res;

        MemorySegment segRes = res.getData();
        if (a.isContiguous()) {
            MemorySegment.copy(a.getData(), 0, segRes, 0, total * 4L);
            float acc = segRes.getAtIndex(ValueLayout.JAVA_FLOAT, 0);
            for (long i = 1; i < total; i++) {
                acc *= segRes.getAtIndex(ValueLayout.JAVA_FLOAT, i);
                segRes.setAtIndex(ValueLayout.JAVA_FLOAT, i, acc);
            }
        } else {
            NDIter iter = new NDIter(a.internalShapeUnsafe());
            MemorySegment segA = a.getData();
            float acc = 1.0f;
            long idx = 0;
            while (iter.hasNext) {
                long off = ShapeUtil.getByteOffset(iter.coords, a.internalStridesUnsafe(), a.getDType());
                acc *= segA.get(ValueLayout.JAVA_FLOAT, off);
                segRes.setAtIndex(ValueLayout.JAVA_FLOAT, idx++, acc);
                iter.next();
            }
        }
        return res;
    }

    public static NDArray cumprodDoubleFlat(NDArray a, NDArray res) {
        long total = a.getSize();
        if (total == 0) return res;

        MemorySegment segRes = res.getData();
        if (a.isContiguous()) {
            MemorySegment.copy(a.getData(), 0, segRes, 0, total * 8L);
            double acc = segRes.getAtIndex(ValueLayout.JAVA_DOUBLE, 0);
            for (long i = 1; i < total; i++) {
                acc *= segRes.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
                segRes.setAtIndex(ValueLayout.JAVA_DOUBLE, i, acc);
            }
        } else {
            NDIter iter = new NDIter(a.internalShapeUnsafe());
            MemorySegment segA = a.getData();
            double acc = 1.0;
            long idx = 0;
            while (iter.hasNext) {
                long off = ShapeUtil.getByteOffset(iter.coords, a.internalStridesUnsafe(), a.getDType());
                acc *= segA.get(ValueLayout.JAVA_DOUBLE, off);
                segRes.setAtIndex(ValueLayout.JAVA_DOUBLE, idx++, acc);
                iter.next();
            }
        }
        return res;
    }

    public static NDArray cumprodIntFlat(NDArray a, NDArray res) {
        long total = a.getSize();
        if (total == 0) return res;

        MemorySegment segRes = res.getData();
        if (a.isContiguous()) {
            MemorySegment.copy(a.getData(), 0, segRes, 0, total * 4L);
            int acc = segRes.getAtIndex(ValueLayout.JAVA_INT, 0);
            for (long i = 1; i < total; i++) {
                acc *= segRes.getAtIndex(ValueLayout.JAVA_INT, i);
                segRes.setAtIndex(ValueLayout.JAVA_INT, i, acc);
            }
        } else {
            NDIter iter = new NDIter(a.internalShapeUnsafe());
            MemorySegment segA = a.getData();
            int acc = 1;
            long idx = 0;
            while (iter.hasNext) {
                long off = ShapeUtil.getByteOffset(iter.coords, a.internalStridesUnsafe(), a.getDType());
                acc *= segA.get(ValueLayout.JAVA_INT, off);
                segRes.setAtIndex(ValueLayout.JAVA_INT, idx++, acc);
                iter.next();
            }
        }
        return res;
    }

    // =========================================================================
    // AXIS CUMPROD
    // =========================================================================

    public static NDArray cumprodFloatAxis(NDArray a, int axis, NDArray res) {
        long total = a.getSize();
        if (total == 0) return res;

        int rank = a.dim();
        long[] shape = a.internalShapeUnsafe();
        long axisLen = shape[axis];
        MemorySegment seg = res.getData();

        copyToResFloat(a, res);

        if (rank == 1 || axis == rank - 1) {
            long innerLen = shape[rank - 1];
            long outerTotal = total / innerLen;
            for (long out = 0; out < outerTotal; out++) {
                long base = out * innerLen;
                float acc = seg.getAtIndex(ValueLayout.JAVA_FLOAT, base);
                for (long i = 1; i < innerLen; i++) {
                    acc *= seg.getAtIndex(ValueLayout.JAVA_FLOAT, base + i);
                    seg.setAtIndex(ValueLayout.JAVA_FLOAT, base + i, acc);
                }
            }
            return res;
        }

        long outerLen = 1;
        for (int d = 0; d < axis; d++) outerLen *= shape[d];
        long innerLen = 1;
        for (int d = axis + 1; d < rank; d++) innerLen *= shape[d];

        for (long out = 0; out < outerLen; out++) {
            long outBase = out * axisLen * innerLen;
            for (long k = 1; k < axisLen; k++) {
                long prevBase = outBase + (k - 1) * innerLen;
                long currBase = outBase + k * innerLen;

                long j = 0;
                long loopbound = SPECIES_F32.loopBound(innerLen);
                for (; j < loopbound; j += VL_F32) {
                    var vPrev = FloatVector.fromMemorySegment(SPECIES_F32, seg, (prevBase + j) * 4L, NATIVE_ORDER);
                    var vCurr = FloatVector.fromMemorySegment(SPECIES_F32, seg, (currBase + j) * 4L, NATIVE_ORDER);
                    vCurr.mul(vPrev).intoMemorySegment(seg, (currBase + j) * 4L, NATIVE_ORDER);
                }
                for (; j < innerLen; j++) {
                    float val = seg.getAtIndex(ValueLayout.JAVA_FLOAT, currBase + j)
                            * seg.getAtIndex(ValueLayout.JAVA_FLOAT, prevBase + j);
                    seg.setAtIndex(ValueLayout.JAVA_FLOAT, currBase + j, val);
                }
            }
        }
        return res;
    }

    public static NDArray cumprodDoubleAxis(NDArray a, int axis, NDArray res) {
        long total = a.getSize();
        if (total == 0) return res;

        int rank = a.dim();
        long[] shape = a.internalShapeUnsafe();
        long axisLen = shape[axis];
        MemorySegment seg = res.getData();

        copyToResDouble(a, res);

        if (rank == 1 || axis == rank - 1) {
            long innerLen = shape[rank - 1];
            long outerTotal = total / innerLen;
            for (long out = 0; out < outerTotal; out++) {
                long base = out * innerLen;
                double acc = seg.getAtIndex(ValueLayout.JAVA_DOUBLE, base);
                for (long i = 1; i < innerLen; i++) {
                    acc *= seg.getAtIndex(ValueLayout.JAVA_DOUBLE, base + i);
                    seg.setAtIndex(ValueLayout.JAVA_DOUBLE, base + i, acc);
                }
            }
            return res;
        }

        long outerLen = 1;
        for (int d = 0; d < axis; d++) outerLen *= shape[d];
        long innerLen = 1;
        for (int d = axis + 1; d < rank; d++) innerLen *= shape[d];

        for (long out = 0; out < outerLen; out++) {
            long outBase = out * axisLen * innerLen;
            for (long k = 1; k < axisLen; k++) {
                long prevBase = outBase + (k - 1) * innerLen;
                long currBase = outBase + k * innerLen;

                long j = 0;
                long loopbound = SPECIES_F64.loopBound(innerLen);
                for (; j < loopbound; j += VL_F64) {
                    var vPrev = DoubleVector.fromMemorySegment(SPECIES_F64, seg, (prevBase + j) * 8L, NATIVE_ORDER);
                    var vCurr = DoubleVector.fromMemorySegment(SPECIES_F64, seg, (currBase + j) * 8L, NATIVE_ORDER);
                    vCurr.mul(vPrev).intoMemorySegment(seg, (currBase + j) * 8L, NATIVE_ORDER);
                }
                for (; j < innerLen; j++) {
                    double val = seg.getAtIndex(ValueLayout.JAVA_DOUBLE, currBase + j)
                            * seg.getAtIndex(ValueLayout.JAVA_DOUBLE, prevBase + j);
                    seg.setAtIndex(ValueLayout.JAVA_DOUBLE, currBase + j, val);
                }
            }
        }
        return res;
    }

    public static NDArray cumprodIntAxis(NDArray a, int axis, NDArray res) {
        long total = a.getSize();
        if (total == 0) return res;

        int rank = a.dim();
        long[] shape = a.internalShapeUnsafe();
        long axisLen = shape[axis];
        MemorySegment seg = res.getData();

        copyToResInt(a, res);

        if (rank == 1 || axis == rank - 1) {
            long innerLen = shape[rank - 1];
            long outerTotal = total / innerLen;
            for (long out = 0; out < outerTotal; out++) {
                long base = out * innerLen;
                int acc = seg.getAtIndex(ValueLayout.JAVA_INT, base);
                for (long i = 1; i < innerLen; i++) {
                    acc *= seg.getAtIndex(ValueLayout.JAVA_INT, base + i);
                    seg.setAtIndex(ValueLayout.JAVA_INT, base + i, acc);
                }
            }
            return res;
        }

        long outerLen = 1;
        for (int d = 0; d < axis; d++) outerLen *= shape[d];
        long innerLen = 1;
        for (int d = axis + 1; d < rank; d++) innerLen *= shape[d];

        for (long out = 0; out < outerLen; out++) {
            long outBase = out * axisLen * innerLen;
            for (long k = 1; k < axisLen; k++) {
                long prevBase = outBase + (k - 1) * innerLen;
                long currBase = outBase + k * innerLen;

                long j = 0;
                long loopbound = SPECIES_I32.loopBound(innerLen);
                for (; j < loopbound; j += VL_I32) {
                    var vPrev = IntVector.fromMemorySegment(SPECIES_I32, seg, (prevBase + j) * 4L, NATIVE_ORDER);
                    var vCurr = IntVector.fromMemorySegment(SPECIES_I32, seg, (currBase + j) * 4L, NATIVE_ORDER);
                    vCurr.mul(vPrev).intoMemorySegment(seg, (currBase + j) * 4L, NATIVE_ORDER);
                }
                for (; j < innerLen; j++) {
                    int val = seg.getAtIndex(ValueLayout.JAVA_INT, currBase + j)
                            * seg.getAtIndex(ValueLayout.JAVA_INT, prevBase + j);
                    seg.setAtIndex(ValueLayout.JAVA_INT, currBase + j, val);
                }
            }
        }
        return res;
    }

    private static void copyToResFloat(NDArray a, NDArray res) {
        if (a.isContiguous()) {
            MemorySegment.copy(a.getData(), 0, res.getData(), 0, a.getSize() * 4L);
        } else {
            NDIter iter = new NDIter(a.internalShapeUnsafe());
            MemorySegment segA = a.getData();
            MemorySegment segRes = res.getData();
            long idx = 0;
            while (iter.hasNext) {
                long off = ShapeUtil.getByteOffset(iter.coords, a.internalStridesUnsafe(), a.getDType());
                segRes.setAtIndex(ValueLayout.JAVA_FLOAT, idx++, segA.get(ValueLayout.JAVA_FLOAT, off));
                iter.next();
            }
        }
    }

    private static void copyToResDouble(NDArray a, NDArray res) {
        if (a.isContiguous()) {
            MemorySegment.copy(a.getData(), 0, res.getData(), 0, a.getSize() * 8L);
        } else {
            NDIter iter = new NDIter(a.internalShapeUnsafe());
            MemorySegment segA = a.getData();
            MemorySegment segRes = res.getData();
            long idx = 0;
            while (iter.hasNext) {
                long off = ShapeUtil.getByteOffset(iter.coords, a.internalStridesUnsafe(), a.getDType());
                segRes.setAtIndex(ValueLayout.JAVA_DOUBLE, idx++, segA.get(ValueLayout.JAVA_DOUBLE, off));
                iter.next();
            }
        }
    }

    private static void copyToResInt(NDArray a, NDArray res) {
        if (a.isContiguous()) {
            MemorySegment.copy(a.getData(), 0, res.getData(), 0, a.getSize() * 4L);
        } else {
            NDIter iter = new NDIter(a.internalShapeUnsafe());
            MemorySegment segA = a.getData();
            MemorySegment segRes = res.getData();
            long idx = 0;
            while (iter.hasNext) {
                long off = ShapeUtil.getByteOffset(iter.coords, a.internalStridesUnsafe(), a.getDType());
                segRes.setAtIndex(ValueLayout.JAVA_INT, idx++, segA.get(ValueLayout.JAVA_INT, off));
                iter.next();
            }
        }
    }
}
