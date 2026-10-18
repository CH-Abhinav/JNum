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

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jnum.JNum;
import jnum.NDArray;

/**
 * High-performance, SIMD-accelerated discrete difference (diff) kernels.
 * Zero-intermediate allocations with cache-aligned sequential memory traversals.
 */
public final class Diff {

    private Diff() {
        throw new AssertionError("Diff kernel cannot be instantiated.");
    }

    public static NDArray diff(NDArray a, int n, int axis, Arena arena) {
        if (n < 0) {
            throw new IllegalArgumentException("order must be non-negative but got " + n);
        }
        int rank = a.dim();
        if (axis < 0) axis += rank;
        if (axis < 0 || axis >= rank) {
            throw new IllegalArgumentException("Axis " + axis + " is out of bounds for array of dimension " + rank);
        }

        if (n == 0) {
            return a.copy();
        }

        long[] shape = a.internalShapeUnsafe();
        if (shape[axis] <= n) {
            long[] emptyShape = shape.clone();
            emptyShape[axis] = 0;
            return JNum.zeros(arena, a.getDType(), emptyShape);
        }

        NDArray current = diff1(a, axis, arena);
        for (int step = 2; step <= n; step++) {
            if (current.internalShapeUnsafe()[axis] <= 1) {
                long[] emptyShape = current.internalShapeUnsafe().clone();
                emptyShape[axis] = 0;
                return JNum.zeros(arena, a.getDType(), emptyShape);
            }
            current = diff1(current, axis, arena);
        }
        return current;
    }

    private static NDArray diff1(NDArray a, int axis, Arena arena) {
        long[] inShape = a.internalShapeUnsafe();
        long[] outShape = inShape.clone();
        outShape[axis] = inShape[axis] - 1;

        NDArray res = JNum.zeros(arena, a.getDType(), outShape);
        NDArray in = a.isContiguous() ? a : a.copy();

        return switch (a.getDType()) {
            case f32 -> diff1Float(in, axis, res);
            case f64 -> diff1Double(in, axis, res);
            case i32 -> diff1Int(in, axis, res);
            default -> throw new UnsupportedOperationException("DType " + a.getDType() + " is not supported for diff.");
        };
    }

    private static NDArray diff1Float(NDArray a, int axis, NDArray res) {
        int rank = a.dim();
        long[] shape = a.internalShapeUnsafe();
        long axisLen = shape[axis];
        long outAxisLen = axisLen - 1;
        MemorySegment segA = a.getData();
        MemorySegment segRes = res.getData();

        if (rank == 1 || axis == rank - 1) {
            long outerTotal = a.getSize() / axisLen;
            for (long out = 0; out < outerTotal; out++) {
                long inBase = out * axisLen;
                long outBase = out * outAxisLen;
                long j = 0;
                long loopbound = SPECIES_F32.loopBound(outAxisLen);
                for (; j < loopbound; j += VL_F32) {
                    var v1 = FloatVector.fromMemorySegment(SPECIES_F32, segA, (inBase + j) * 4L, NATIVE_ORDER);
                    var v2 = FloatVector.fromMemorySegment(SPECIES_F32, segA, (inBase + j + 1) * 4L, NATIVE_ORDER);
                    v2.sub(v1).intoMemorySegment(segRes, (outBase + j) * 4L, NATIVE_ORDER);
                }
                for (; j < outAxisLen; j++) {
                    float diff = segA.getAtIndex(ValueLayout.JAVA_FLOAT, inBase + j + 1)
                               - segA.getAtIndex(ValueLayout.JAVA_FLOAT, inBase + j);
                    segRes.setAtIndex(ValueLayout.JAVA_FLOAT, outBase + j, diff);
                }
            }
            return res;
        }

        long outerLen = 1;
        for (int d = 0; d < axis; d++) outerLen *= shape[d];
        long innerLen = 1;
        for (int d = axis + 1; d < rank; d++) innerLen *= shape[d];

        for (long out = 0; out < outerLen; out++) {
            long inOutBase = out * axisLen * innerLen;
            long resOutBase = out * outAxisLen * innerLen;

            for (long k = 0; k < outAxisLen; k++) {
                long inSlice1 = inOutBase + k * innerLen;
                long inSlice2 = inOutBase + (k + 1) * innerLen;
                long resSlice = resOutBase + k * innerLen;

                long j = 0;
                long loopbound = SPECIES_F32.loopBound(innerLen);
                for (; j < loopbound; j += VL_F32) {
                    var v1 = FloatVector.fromMemorySegment(SPECIES_F32, segA, (inSlice1 + j) * 4L, NATIVE_ORDER);
                    var v2 = FloatVector.fromMemorySegment(SPECIES_F32, segA, (inSlice2 + j) * 4L, NATIVE_ORDER);
                    v2.sub(v1).intoMemorySegment(segRes, (resSlice + j) * 4L, NATIVE_ORDER);
                }
                for (; j < innerLen; j++) {
                    float diff = segA.getAtIndex(ValueLayout.JAVA_FLOAT, inSlice2 + j)
                               - segA.getAtIndex(ValueLayout.JAVA_FLOAT, inSlice1 + j);
                    segRes.setAtIndex(ValueLayout.JAVA_FLOAT, resSlice + j, diff);
                }
            }
        }
        return res;
    }

    private static NDArray diff1Double(NDArray a, int axis, NDArray res) {
        int rank = a.dim();
        long[] shape = a.internalShapeUnsafe();
        long axisLen = shape[axis];
        long outAxisLen = axisLen - 1;
        MemorySegment segA = a.getData();
        MemorySegment segRes = res.getData();

        if (rank == 1 || axis == rank - 1) {
            long outerTotal = a.getSize() / axisLen;
            for (long out = 0; out < outerTotal; out++) {
                long inBase = out * axisLen;
                long outBase = out * outAxisLen;
                long j = 0;
                long loopbound = SPECIES_F64.loopBound(outAxisLen);
                for (; j < loopbound; j += VL_F64) {
                    var v1 = DoubleVector.fromMemorySegment(SPECIES_F64, segA, (inBase + j) * 8L, NATIVE_ORDER);
                    var v2 = DoubleVector.fromMemorySegment(SPECIES_F64, segA, (inBase + j + 1) * 8L, NATIVE_ORDER);
                    v2.sub(v1).intoMemorySegment(segRes, (outBase + j) * 8L, NATIVE_ORDER);
                }
                for (; j < outAxisLen; j++) {
                    double diff = segA.getAtIndex(ValueLayout.JAVA_DOUBLE, inBase + j + 1)
                                - segA.getAtIndex(ValueLayout.JAVA_DOUBLE, inBase + j);
                    segRes.setAtIndex(ValueLayout.JAVA_DOUBLE, outBase + j, diff);
                }
            }
            return res;
        }

        long outerLen = 1;
        for (int d = 0; d < axis; d++) outerLen *= shape[d];
        long innerLen = 1;
        for (int d = axis + 1; d < rank; d++) innerLen *= shape[d];

        for (long out = 0; out < outerLen; out++) {
            long inOutBase = out * axisLen * innerLen;
            long resOutBase = out * outAxisLen * innerLen;

            for (long k = 0; k < outAxisLen; k++) {
                long inSlice1 = inOutBase + k * innerLen;
                long inSlice2 = inOutBase + (k + 1) * innerLen;
                long resSlice = resOutBase + k * innerLen;

                long j = 0;
                long loopbound = SPECIES_F64.loopBound(innerLen);
                for (; j < loopbound; j += VL_F64) {
                    var v1 = DoubleVector.fromMemorySegment(SPECIES_F64, segA, (inSlice1 + j) * 8L, NATIVE_ORDER);
                    var v2 = DoubleVector.fromMemorySegment(SPECIES_F64, segA, (inSlice2 + j) * 8L, NATIVE_ORDER);
                    v2.sub(v1).intoMemorySegment(segRes, (resSlice + j) * 8L, NATIVE_ORDER);
                }
                for (; j < innerLen; j++) {
                    double diff = segA.getAtIndex(ValueLayout.JAVA_DOUBLE, inSlice2 + j)
                                - segA.getAtIndex(ValueLayout.JAVA_DOUBLE, inSlice1 + j);
                    segRes.setAtIndex(ValueLayout.JAVA_DOUBLE, resSlice + j, diff);
                }
            }
        }
        return res;
    }

    private static NDArray diff1Int(NDArray a, int axis, NDArray res) {
        int rank = a.dim();
        long[] shape = a.internalShapeUnsafe();
        long axisLen = shape[axis];
        long outAxisLen = axisLen - 1;
        MemorySegment segA = a.getData();
        MemorySegment segRes = res.getData();

        if (rank == 1 || axis == rank - 1) {
            long outerTotal = a.getSize() / axisLen;
            for (long out = 0; out < outerTotal; out++) {
                long inBase = out * axisLen;
                long outBase = out * outAxisLen;
                long j = 0;
                long loopbound = SPECIES_I32.loopBound(outAxisLen);
                for (; j < loopbound; j += VL_I32) {
                    var v1 = IntVector.fromMemorySegment(SPECIES_I32, segA, (inBase + j) * 4L, NATIVE_ORDER);
                    var v2 = IntVector.fromMemorySegment(SPECIES_I32, segA, (inBase + j + 1) * 4L, NATIVE_ORDER);
                    v2.sub(v1).intoMemorySegment(segRes, (outBase + j) * 4L, NATIVE_ORDER);
                }
                for (; j < outAxisLen; j++) {
                    int diff = segA.getAtIndex(ValueLayout.JAVA_INT, inBase + j + 1)
                             - segA.getAtIndex(ValueLayout.JAVA_INT, inBase + j);
                    segRes.setAtIndex(ValueLayout.JAVA_INT, outBase + j, diff);
                }
            }
            return res;
        }

        long outerLen = 1;
        for (int d = 0; d < axis; d++) outerLen *= shape[d];
        long innerLen = 1;
        for (int d = axis + 1; d < rank; d++) innerLen *= shape[d];

        for (long out = 0; out < outerLen; out++) {
            long inOutBase = out * axisLen * innerLen;
            long resOutBase = out * outAxisLen * innerLen;

            for (long k = 0; k < outAxisLen; k++) {
                long inSlice1 = inOutBase + k * innerLen;
                long inSlice2 = inOutBase + (k + 1) * innerLen;
                long resSlice = resOutBase + k * innerLen;

                long j = 0;
                long loopbound = SPECIES_I32.loopBound(innerLen);
                for (; j < loopbound; j += VL_I32) {
                    var v1 = IntVector.fromMemorySegment(SPECIES_I32, segA, (inSlice1 + j) * 4L, NATIVE_ORDER);
                    var v2 = IntVector.fromMemorySegment(SPECIES_I32, segA, (inSlice2 + j) * 4L, NATIVE_ORDER);
                    v2.sub(v1).intoMemorySegment(segRes, (resSlice + j) * 4L, NATIVE_ORDER);
                }
                for (; j < innerLen; j++) {
                    int diff = segA.getAtIndex(ValueLayout.JAVA_INT, inSlice2 + j)
                             - segA.getAtIndex(ValueLayout.JAVA_INT, inSlice1 + j);
                    segRes.setAtIndex(ValueLayout.JAVA_INT, resSlice + j, diff);
                }
            }
        }
        return res;
    }
}
