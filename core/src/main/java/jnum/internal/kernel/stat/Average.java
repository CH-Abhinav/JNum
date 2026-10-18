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
import java.util.Arrays;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jnum.NDArray;
import jnum.internal.layout.NDIter;
import jnum.internal.layout.ShapeUtil;

/**
 * Kernels for weighted arithmetic average: sum(a * w) / sum(w).
 */
public final class Average {

    private Average() {
        throw new AssertionError("Average kernel cannot be instantiated.");
    }

    public static double averageFloat(NDArray a, NDArray weights) {
        long size = a.getSize();
        if (size == 0) return Double.NaN;
        if (weights.getSize() != size) {
            throw new IllegalArgumentException(
                "Weights size (" + weights.getSize() + ") must match array size (" + size + ")"
            );
        }

        if (a.isContiguous() && weights.isContiguous() && weights.getDType() == a.getDType()) {
            MemorySegment aSeg = a.getData();
            MemorySegment wSeg = weights.getData();

            long loopbound4 = ((size) / (VL_F32 * 4)) * (VL_F32 * 4);
            var vAW1 = FloatVector.zero(SPECIES_F32);
            var vAW2 = FloatVector.zero(SPECIES_F32);
            var vAW3 = FloatVector.zero(SPECIES_F32);
            var vAW4 = FloatVector.zero(SPECIES_F32);

            var vW1 = FloatVector.zero(SPECIES_F32);
            var vW2 = FloatVector.zero(SPECIES_F32);
            var vW3 = FloatVector.zero(SPECIES_F32);
            var vW4 = FloatVector.zero(SPECIES_F32);

            long i = 0;
            for (; i < loopbound4; i += VL_F32 * 4) {
                var a1 = FloatVector.fromMemorySegment(SPECIES_F32, aSeg, i * BYTES_F32, NATIVE_ORDER);
                var w1 = FloatVector.fromMemorySegment(SPECIES_F32, wSeg, i * BYTES_F32, NATIVE_ORDER);
                vAW1 = a1.fma(w1, vAW1);
                vW1 = vW1.add(w1);

                var a2 = FloatVector.fromMemorySegment(SPECIES_F32, aSeg, (i + VL_F32) * BYTES_F32, NATIVE_ORDER);
                var w2 = FloatVector.fromMemorySegment(SPECIES_F32, wSeg, (i + VL_F32) * BYTES_F32, NATIVE_ORDER);
                vAW2 = a2.fma(w2, vAW2);
                vW2 = vW2.add(w2);

                var a3 = FloatVector.fromMemorySegment(SPECIES_F32, aSeg, (i + VL_F32 * 2) * BYTES_F32, NATIVE_ORDER);
                var w3 = FloatVector.fromMemorySegment(SPECIES_F32, wSeg, (i + VL_F32 * 2) * BYTES_F32, NATIVE_ORDER);
                vAW3 = a3.fma(w3, vAW3);
                vW3 = vW3.add(w3);

                var a4 = FloatVector.fromMemorySegment(SPECIES_F32, aSeg, (i + VL_F32 * 3) * BYTES_F32, NATIVE_ORDER);
                var w4 = FloatVector.fromMemorySegment(SPECIES_F32, wSeg, (i + VL_F32 * 3) * BYTES_F32, NATIVE_ORDER);
                vAW4 = a4.fma(w4, vAW4);
                vW4 = vW4.add(w4);
            }

            double sumAW = (double) vAW1.add(vAW2).add(vAW3).add(vAW4).reduceLanes(VectorOperators.ADD);
            double sumW = (double) vW1.add(vW2).add(vW3).add(vW4).reduceLanes(VectorOperators.ADD);

            for (; i < size; i++) {
                float aVal = aSeg.getAtIndex(ValueLayout.JAVA_FLOAT, i);
                float wVal = wSeg.getAtIndex(ValueLayout.JAVA_FLOAT, i);
                sumAW += (double) aVal * (double) wVal;
                sumW += (double) wVal;
            }

            if (sumW == 0.0) return Double.NaN;
            return sumAW / sumW;
        }

        // Generic / non-contiguous fallback
        double sumAW = 0.0;
        double sumW = 0.0;
        NDIter aIter = new NDIter(a.internalShapeUnsafe());
        NDIter wIter = new NDIter(weights.internalShapeUnsafe());
        while (aIter.hasNext && wIter.hasNext) {
            long aOffset = ShapeUtil.getByteOffset(aIter.coords, a.internalStridesUnsafe(), a.getDType());
            long wOffset = ShapeUtil.getByteOffset(wIter.coords, weights.internalStridesUnsafe(), weights.getDType());
            double aVal = a.getData().get(ValueLayout.JAVA_FLOAT, aOffset);
            double wVal = weights.getData().get(ValueLayout.JAVA_FLOAT, wOffset);
            sumAW += aVal * wVal;
            sumW += wVal;
            aIter.next();
            wIter.next();
        }
        if (sumW == 0.0) return Double.NaN;
        return sumAW / sumW;
    }

    public static double averageDouble(NDArray a, NDArray weights) {
        long size = a.getSize();
        if (size == 0) return Double.NaN;
        if (weights.getSize() != size) {
            throw new IllegalArgumentException(
                "Weights size (" + weights.getSize() + ") must match array size (" + size + ")"
            );
        }

        if (a.isContiguous() && weights.isContiguous() && weights.getDType() == a.getDType()) {
            MemorySegment aSeg = a.getData();
            MemorySegment wSeg = weights.getData();

            long loopbound4 = ((size) / (VL_F64 * 4)) * (VL_F64 * 4);
            var vAW1 = DoubleVector.zero(SPECIES_F64);
            var vAW2 = DoubleVector.zero(SPECIES_F64);
            var vAW3 = DoubleVector.zero(SPECIES_F64);
            var vAW4 = DoubleVector.zero(SPECIES_F64);

            var vW1 = DoubleVector.zero(SPECIES_F64);
            var vW2 = DoubleVector.zero(SPECIES_F64);
            var vW3 = DoubleVector.zero(SPECIES_F64);
            var vW4 = DoubleVector.zero(SPECIES_F64);

            long i = 0;
            for (; i < loopbound4; i += VL_F64 * 4) {
                var a1 = DoubleVector.fromMemorySegment(SPECIES_F64, aSeg, i * BYTES_F64, NATIVE_ORDER);
                var w1 = DoubleVector.fromMemorySegment(SPECIES_F64, wSeg, i * BYTES_F64, NATIVE_ORDER);
                vAW1 = a1.fma(w1, vAW1);
                vW1 = vW1.add(w1);

                var a2 = DoubleVector.fromMemorySegment(SPECIES_F64, aSeg, (i + VL_F64) * BYTES_F64, NATIVE_ORDER);
                var w2 = DoubleVector.fromMemorySegment(SPECIES_F64, wSeg, (i + VL_F64) * BYTES_F64, NATIVE_ORDER);
                vAW2 = a2.fma(w2, vAW2);
                vW2 = vW2.add(w2);

                var a3 = DoubleVector.fromMemorySegment(SPECIES_F64, aSeg, (i + VL_F64 * 2) * BYTES_F64, NATIVE_ORDER);
                var w3 = DoubleVector.fromMemorySegment(SPECIES_F64, wSeg, (i + VL_F64 * 2) * BYTES_F64, NATIVE_ORDER);
                vAW3 = a3.fma(w3, vAW3);
                vW3 = vW3.add(w3);

                var a4 = DoubleVector.fromMemorySegment(SPECIES_F64, aSeg, (i + VL_F64 * 3) * BYTES_F64, NATIVE_ORDER);
                var w4 = DoubleVector.fromMemorySegment(SPECIES_F64, wSeg, (i + VL_F64 * 3) * BYTES_F64, NATIVE_ORDER);
                vAW4 = a4.fma(w4, vAW4);
                vW4 = vW4.add(w4);
            }

            double sumAW = vAW1.add(vAW2).add(vAW3).add(vAW4).reduceLanes(VectorOperators.ADD);
            double sumW = vW1.add(vW2).add(vW3).add(vW4).reduceLanes(VectorOperators.ADD);

            for (; i < size; i++) {
                double aVal = aSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
                double wVal = wSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
                sumAW += aVal * wVal;
                sumW += wVal;
            }

            if (sumW == 0.0) return Double.NaN;
            return sumAW / sumW;
        }

        // Generic fallback
        double sumAW = 0.0;
        double sumW = 0.0;
        NDIter aIter = new NDIter(a.internalShapeUnsafe());
        NDIter wIter = new NDIter(weights.internalShapeUnsafe());
        while (aIter.hasNext && wIter.hasNext) {
            long aOffset = ShapeUtil.getByteOffset(aIter.coords, a.internalStridesUnsafe(), a.getDType());
            long wOffset = ShapeUtil.getByteOffset(wIter.coords, weights.internalStridesUnsafe(), weights.getDType());
            double aVal = a.getData().get(ValueLayout.JAVA_DOUBLE, aOffset);
            double wVal = weights.getData().get(ValueLayout.JAVA_DOUBLE, wOffset);
            sumAW += aVal * wVal;
            sumW += wVal;
            aIter.next();
            wIter.next();
        }
        if (sumW == 0.0) return Double.NaN;
        return sumAW / sumW;
    }

    public static NDArray averageFloatAxis(NDArray a, int axis, NDArray weights, NDArray resArray) {
        long size = a.internalShapeUnsafe()[axis];
        long strideA = a.internalStridesUnsafe()[axis];
        long resSize = resArray.getSize();

        boolean is1DWeights = weights.dim() == 1 && weights.getSize() == size;
        if (!is1DWeights && !Arrays.equals(weights.internalShapeUnsafe(), a.internalShapeUnsafe())) {
            throw new IllegalArgumentException(
                "Weights shape " + Arrays.toString(weights.internalShapeUnsafe()) +
                " is incompatible with array shape " + Arrays.toString(a.internalShapeUnsafe()) + " along axis " + axis
            );
        }

        MemorySegment aSeg = a.getData();
        MemorySegment wSeg = weights.getData();
        MemorySegment resSeg = resArray.getData();

        if (is1DWeights) {
            double totalWeight = 0.0;
            for (long k = 0; k < size; k++) {
                totalWeight += wSeg.getAtIndex(ValueLayout.JAVA_FLOAT, k * weights.internalStridesUnsafe()[0]);
            }
            float invTotalWeight = (float) (1.0 / totalWeight);

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

                float acc = 0f;
                long wStride = weights.internalStridesUnsafe()[0];
                for (long k = 0; k < size; k++) {
                    float aVal = aSeg.getAtIndex(ValueLayout.JAVA_FLOAT, baseOffset + k * strideA);
                    float wVal = wSeg.getAtIndex(ValueLayout.JAVA_FLOAT, k * wStride);
                    acc += aVal * wVal;
                }
                resSeg.setAtIndex(ValueLayout.JAVA_FLOAT, offsetRes, acc * invTotalWeight);
            }
        } else {
            long strideW = weights.internalStridesUnsafe()[axis];
            for (long i = 0; i < resSize; i++) {
                long tempIndex = i;
                long baseOffsetA = 0;
                long baseOffsetW = 0;
                long offsetRes = 0;
                for (int d = resArray.internalShapeUnsafe().length - 1; d >= 0; d--) {
                    long coord = tempIndex % resArray.internalShapeUnsafe()[d];
                    tempIndex /= resArray.internalShapeUnsafe()[d];
                    offsetRes += coord * resArray.internalStridesUnsafe()[d];

                    int aDim = (d >= axis) ? d + 1 : d;
                    baseOffsetA += coord * a.internalStridesUnsafe()[aDim];
                    baseOffsetW += coord * weights.internalStridesUnsafe()[aDim];
                }

                float acc = 0f;
                float sumW = 0f;
                for (long k = 0; k < size; k++) {
                    float aVal = aSeg.getAtIndex(ValueLayout.JAVA_FLOAT, baseOffsetA + k * strideA);
                    float wVal = wSeg.getAtIndex(ValueLayout.JAVA_FLOAT, baseOffsetW + k * strideW);
                    acc += aVal * wVal;
                    sumW += wVal;
                }
                resSeg.setAtIndex(ValueLayout.JAVA_FLOAT, offsetRes, sumW == 0f ? Float.NaN : acc / sumW);
            }
        }
        return resArray;
    }

    public static NDArray averageDoubleAxis(NDArray a, int axis, NDArray weights, NDArray resArray) {
        long size = a.internalShapeUnsafe()[axis];
        long strideA = a.internalStridesUnsafe()[axis];
        long resSize = resArray.getSize();

        boolean is1DWeights = weights.dim() == 1 && weights.getSize() == size;
        if (!is1DWeights && !Arrays.equals(weights.internalShapeUnsafe(), a.internalShapeUnsafe())) {
            throw new IllegalArgumentException(
                "Weights shape " + Arrays.toString(weights.internalShapeUnsafe()) +
                " is incompatible with array shape " + Arrays.toString(a.internalShapeUnsafe()) + " along axis " + axis
            );
        }

        MemorySegment aSeg = a.getData();
        MemorySegment wSeg = weights.getData();
        MemorySegment resSeg = resArray.getData();

        if (is1DWeights) {
            double totalWeight = 0.0;
            for (long k = 0; k < size; k++) {
                totalWeight += wSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, k * weights.internalStridesUnsafe()[0]);
            }
            double invTotalWeight = 1.0 / totalWeight;

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

                double acc = 0.0;
                long wStride = weights.internalStridesUnsafe()[0];
                for (long k = 0; k < size; k++) {
                    double aVal = aSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, baseOffset + k * strideA);
                    double wVal = wSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, k * wStride);
                    acc += aVal * wVal;
                }
                resSeg.setAtIndex(ValueLayout.JAVA_DOUBLE, offsetRes, acc * invTotalWeight);
            }
        } else {
            long strideW = weights.internalStridesUnsafe()[axis];
            for (long i = 0; i < resSize; i++) {
                long tempIndex = i;
                long baseOffsetA = 0;
                long baseOffsetW = 0;
                long offsetRes = 0;
                for (int d = resArray.internalShapeUnsafe().length - 1; d >= 0; d--) {
                    long coord = tempIndex % resArray.internalShapeUnsafe()[d];
                    tempIndex /= resArray.internalShapeUnsafe()[d];
                    offsetRes += coord * resArray.internalStridesUnsafe()[d];

                    int aDim = (d >= axis) ? d + 1 : d;
                    baseOffsetA += coord * a.internalStridesUnsafe()[aDim];
                    baseOffsetW += coord * weights.internalStridesUnsafe()[aDim];
                }

                double acc = 0.0;
                double sumW = 0.0;
                for (long k = 0; k < size; k++) {
                    double aVal = aSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, baseOffsetA + k * strideA);
                    double wVal = wSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, baseOffsetW + k * strideW);
                    acc += aVal * wVal;
                    sumW += wVal;
                }
                resSeg.setAtIndex(ValueLayout.JAVA_DOUBLE, offsetRes, sumW == 0.0 ? Double.NaN : acc / sumW);
            }
        }
        return resArray;
    }
}
