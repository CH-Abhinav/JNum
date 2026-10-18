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
import jdk.incubator.vector.VectorOperators;
import jnum.NDArray;
import jnum.internal.layout.NDIter;
import jnum.internal.layout.ShapeUtil;

/**
 * Fused Peak-to-Peak (ptp: max - min) kernels.
 * Evaluates minimum and maximum simultaneously in a single memory pass.
 */
public final class Ptp {

    private Ptp() {
        throw new AssertionError("Ptp kernel cannot be instantiated.");
    }

    // =========================================================================
    // SCALAR PTP
    // =========================================================================

    public static double ptpFloat(NDArray a) {
        long size = a.getSize();
        if (size == 0) return Double.NaN;
        if (!a.isContiguous()) {
            float minVal = Float.POSITIVE_INFINITY;
            float maxVal = Float.NEGATIVE_INFINITY;
            NDIter iter = new NDIter(a.internalShapeUnsafe());
            while (iter.hasNext) {
                long byteOffset = ShapeUtil.getByteOffset(iter.coords, a.internalStridesUnsafe(), a.getDType());
                float val = a.getData().get(ValueLayout.JAVA_FLOAT, byteOffset);
                if (val < minVal) minVal = val;
                if (val > maxVal) maxVal = val;
                iter.next();
            }
            return (double) maxVal - (double) minVal;
        }

        MemorySegment seg = a.getData();
        long loopbound = SPECIES_F32.loopBound(size);
        var vMin = FloatVector.broadcast(SPECIES_F32, Float.POSITIVE_INFINITY);
        var vMax = FloatVector.broadcast(SPECIES_F32, Float.NEGATIVE_INFINITY);

        long i = 0;
        for (; i < loopbound; i += VL_F32) {
            var v = FloatVector.fromMemorySegment(SPECIES_F32, seg, i * BYTES_F32, NATIVE_ORDER);
            vMin = vMin.min(v);
            vMax = vMax.max(v);
        }

        float minVal = vMin.reduceLanes(VectorOperators.MIN);
        float maxVal = vMax.reduceLanes(VectorOperators.MAX);

        for (; i < size; i++) {
            float val = seg.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            if (val < minVal) minVal = val;
            if (val > maxVal) maxVal = val;
        }

        return (double) maxVal - (double) minVal;
    }

    public static double ptpDouble(NDArray a) {
        long size = a.getSize();
        if (size == 0) return Double.NaN;
        if (!a.isContiguous()) {
            double minVal = Double.POSITIVE_INFINITY;
            double maxVal = Double.NEGATIVE_INFINITY;
            NDIter iter = new NDIter(a.internalShapeUnsafe());
            while (iter.hasNext) {
                long byteOffset = ShapeUtil.getByteOffset(iter.coords, a.internalStridesUnsafe(), a.getDType());
                double val = a.getData().get(ValueLayout.JAVA_DOUBLE, byteOffset);
                if (val < minVal) minVal = val;
                if (val > maxVal) maxVal = val;
                iter.next();
            }
            return maxVal - minVal;
        }

        MemorySegment seg = a.getData();
        long loopbound = SPECIES_F64.loopBound(size);
        var vMin = DoubleVector.broadcast(SPECIES_F64, Double.POSITIVE_INFINITY);
        var vMax = DoubleVector.broadcast(SPECIES_F64, Double.NEGATIVE_INFINITY);

        long i = 0;
        for (; i < loopbound; i += VL_F64) {
            var v = DoubleVector.fromMemorySegment(SPECIES_F64, seg, i * BYTES_F64, NATIVE_ORDER);
            vMin = vMin.min(v);
            vMax = vMax.max(v);
        }

        double minVal = vMin.reduceLanes(VectorOperators.MIN);
        double maxVal = vMax.reduceLanes(VectorOperators.MAX);

        for (; i < size; i++) {
            double val = seg.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
            if (val < minVal) minVal = val;
            if (val > maxVal) maxVal = val;
        }

        return maxVal - minVal;
    }

    public static double ptpInt(NDArray a) {
        long size = a.getSize();
        if (size == 0) return Double.NaN;
        if (!a.isContiguous()) {
            int minVal = Integer.MAX_VALUE;
            int maxVal = Integer.MIN_VALUE;
            NDIter iter = new NDIter(a.internalShapeUnsafe());
            while (iter.hasNext) {
                long byteOffset = ShapeUtil.getByteOffset(iter.coords, a.internalStridesUnsafe(), a.getDType());
                int val = a.getData().get(ValueLayout.JAVA_INT, byteOffset);
                if (val < minVal) minVal = val;
                if (val > maxVal) maxVal = val;
                iter.next();
            }
            return (double) maxVal - (double) minVal;
        }

        MemorySegment seg = a.getData();
        long loopbound = SPECIES_I32.loopBound(size);
        var vMin = IntVector.broadcast(SPECIES_I32, Integer.MAX_VALUE);
        var vMax = IntVector.broadcast(SPECIES_I32, Integer.MIN_VALUE);

        long i = 0;
        for (; i < loopbound; i += VL_I32) {
            var v = IntVector.fromMemorySegment(SPECIES_I32, seg, i * BYTES_I32, NATIVE_ORDER);
            vMin = vMin.min(v);
            vMax = vMax.max(v);
        }

        int minVal = vMin.reduceLanes(VectorOperators.MIN);
        int maxVal = vMax.reduceLanes(VectorOperators.MAX);

        for (; i < size; i++) {
            int val = seg.getAtIndex(ValueLayout.JAVA_INT, i);
            if (val < minVal) minVal = val;
            if (val > maxVal) maxVal = val;
        }

        return (double) maxVal - (double) minVal;
    }

    // =========================================================================
    // AXIS PTP
    // =========================================================================

    public static NDArray ptpFloatAxis(NDArray a, int axis, NDArray resArray) {
        long size = a.internalShapeUnsafe()[axis];
        long strideA = a.internalStridesUnsafe()[axis];
        long resSize = resArray.getSize();
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

            if (strideA == 1) {
                var vMin = FloatVector.broadcast(SPECIES_F32, Float.POSITIVE_INFINITY);
                var vMax = FloatVector.broadcast(SPECIES_F32, Float.NEGATIVE_INFINITY);
                long k = 0;
                long loopbound = SPECIES_F32.loopBound(size);
                for (; k < loopbound; k += VL_F32) {
                    var v = FloatVector.fromMemorySegment(SPECIES_F32, aSeg, (baseOffset + k) * BYTES_F32, NATIVE_ORDER);
                    vMin = vMin.min(v);
                    vMax = vMax.max(v);
                }
                float minVal = vMin.reduceLanes(VectorOperators.MIN);
                float maxVal = vMax.reduceLanes(VectorOperators.MAX);
                for (; k < size; k++) {
                    float val = aSeg.getAtIndex(ValueLayout.JAVA_FLOAT, baseOffset + k);
                    if (val < minVal) minVal = val;
                    if (val > maxVal) maxVal = val;
                }
                resSeg.setAtIndex(ValueLayout.JAVA_FLOAT, offsetRes, maxVal - minVal);
            } else {
                float minVal = Float.POSITIVE_INFINITY;
                float maxVal = Float.NEGATIVE_INFINITY;
                for (long k = 0; k < size; k++) {
                    float val = aSeg.getAtIndex(ValueLayout.JAVA_FLOAT, baseOffset + k * strideA);
                    if (val < minVal) minVal = val;
                    if (val > maxVal) maxVal = val;
                }
                resSeg.setAtIndex(ValueLayout.JAVA_FLOAT, offsetRes, maxVal - minVal);
            }
        }
        return resArray;
    }

    public static NDArray ptpDoubleAxis(NDArray a, int axis, NDArray resArray) {
        long size = a.internalShapeUnsafe()[axis];
        long strideA = a.internalStridesUnsafe()[axis];
        long resSize = resArray.getSize();
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

            if (strideA == 1) {
                var vMin = DoubleVector.broadcast(SPECIES_F64, Double.POSITIVE_INFINITY);
                var vMax = DoubleVector.broadcast(SPECIES_F64, Double.NEGATIVE_INFINITY);
                long k = 0;
                long loopbound = SPECIES_F64.loopBound(size);
                for (; k < loopbound; k += VL_F64) {
                    var v = DoubleVector.fromMemorySegment(SPECIES_F64, aSeg, (baseOffset + k) * BYTES_F64, NATIVE_ORDER);
                    vMin = vMin.min(v);
                    vMax = vMax.max(v);
                }
                double minVal = vMin.reduceLanes(VectorOperators.MIN);
                double maxVal = vMax.reduceLanes(VectorOperators.MAX);
                for (; k < size; k++) {
                    double val = aSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, baseOffset + k);
                    if (val < minVal) minVal = val;
                    if (val > maxVal) maxVal = val;
                }
                resSeg.setAtIndex(ValueLayout.JAVA_DOUBLE, offsetRes, maxVal - minVal);
            } else {
                double minVal = Double.POSITIVE_INFINITY;
                double maxVal = Double.NEGATIVE_INFINITY;
                for (long k = 0; k < size; k++) {
                    double val = aSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, baseOffset + k * strideA);
                    if (val < minVal) minVal = val;
                    if (val > maxVal) maxVal = val;
                }
                resSeg.setAtIndex(ValueLayout.JAVA_DOUBLE, offsetRes, maxVal - minVal);
            }
        }
        return resArray;
    }

    public static NDArray ptpIntAxis(NDArray a, int axis, NDArray resArray) {
        long size = a.internalShapeUnsafe()[axis];
        long strideA = a.internalStridesUnsafe()[axis];
        long resSize = resArray.getSize();
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

            int minVal = Integer.MAX_VALUE;
            int maxVal = Integer.MIN_VALUE;
            for (long k = 0; k < size; k++) {
                int val = aSeg.getAtIndex(ValueLayout.JAVA_INT, baseOffset + k * strideA);
                if (val < minVal) minVal = val;
                if (val > maxVal) maxVal = val;
            }
            resSeg.setAtIndex(ValueLayout.JAVA_INT, offsetRes, maxVal - minVal);
        }
        return resArray;
    }
}
