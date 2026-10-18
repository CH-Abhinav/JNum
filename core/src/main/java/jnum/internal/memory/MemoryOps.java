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
package jnum.internal.memory;

import jnum.DType;
import jnum.JNum;
import jnum.NDArray;
import jnum.internal.layout.LayoutOps;
import jnum.internal.layout.NDIter;
import jnum.internal.layout.ShapeUtil;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Arrays;

/**
 * Internal operations for tensor memory allocations, materialization, copying, and casting.
 */
public final class MemoryOps {

    private MemoryOps() {
        throw new AssertionError("MemoryOps utility class cannot be instantiated.");
    }

    /**
     * Reshapes a tensor to a new shape, materializing a contiguous copy if necessary.
     */
    public static NDArray reshape(NDArray a, long... newShape) {
        long newCalcSize = 1;
        for (long dim : newShape) newCalcSize *= dim;
        if (newCalcSize != a.getSize()) {
            throw new IllegalArgumentException("Cannot reshape array of size " + a.getSize() + " into shape " + Arrays.toString(newShape));
        }
        NDArray safeThis = a.isContiguous() ? a : a.contiguous();
        return NDArray.ofRaw(safeThis.getData(), newShape, ShapeUtil.calculateDefaultStrides(newShape), a.getDType());
    }

    /**
     * Materializes a contiguous copy of a potentially non-contiguous (strided/sliced) tensor.
     */
    public static NDArray contiguous(NDArray a, Arena arena) {
        if (a.isContiguous()) return a;
        long byteSize = a.getSize() * a.getDType().layout.byteSize();
        MemorySegment segment = arena.allocate(byteSize, 64);
        long[] shape = a.internalShapeUnsafe();
        long[] strides = a.internalStridesUnsafe();
        long[] newStrides = ShapeUtil.calculateDefaultStrides(shape);

        for (long i = 0; i < a.getSize(); i++) {
            long tempindex = i;
            long[] coord = new long[shape.length];
            for (int j = shape.length - 1; j >= 0; j--) {
                coord[j] = tempindex % shape[j];
                tempindex = tempindex / shape[j];
            }
            long oldFlatIndex = 0;
            for (int d = 0; d < shape.length; d++) {
                oldFlatIndex += coord[d] * strides[d];
            }
            switch (a.getDType()) {
                case f32 -> segment.setAtIndex(ValueLayout.JAVA_FLOAT, i, a.getData().getAtIndex(ValueLayout.JAVA_FLOAT, oldFlatIndex));
                case i32 -> segment.setAtIndex(ValueLayout.JAVA_INT, i, a.getData().getAtIndex(ValueLayout.JAVA_INT, oldFlatIndex));
                case f64 -> segment.setAtIndex(ValueLayout.JAVA_DOUBLE, i, a.getData().getAtIndex(ValueLayout.JAVA_DOUBLE, oldFlatIndex));
                case bool -> segment.setAtIndex(ValueLayout.JAVA_BYTE, i, a.getData().getAtIndex(ValueLayout.JAVA_BYTE, oldFlatIndex));
                default -> throw new UnsupportedOperationException("This dtype " + a.getDType() + " doesn't support this method");
            }
        }
        return NDArray.ofRaw(segment, shape, newStrides, a.getDType());
    }

    /**
     * Creates a deep copy of the tensor.
     */
    public static NDArray copy(NDArray a) {
        NDArray dups = JNum.zeros(a.getDType(), a.internalShapeUnsafe());
        if (a.isContiguous()) {
            long logicalBytes = a.getSize() * a.getDType().layout.byteSize();
            MemorySegment.copy(a.getData(), 0, dups.getData(), 0, logicalBytes);
            return dups;
        }
        NDIter srcIter = new NDIter(a.internalShapeUnsafe());
        long dstIndex = 0;
        while (srcIter.hasNext) {
            long byteOffset = ShapeUtil.getByteOffset(srcIter.coords, a.internalStridesUnsafe(), a.getDType());
            switch (a.getDType()) {
                case f32 -> dups.getData().setAtIndex(ValueLayout.JAVA_FLOAT, dstIndex, a.getData().get(ValueLayout.JAVA_FLOAT, byteOffset));
                case i32 -> dups.getData().setAtIndex(ValueLayout.JAVA_INT, dstIndex, a.getData().get(ValueLayout.JAVA_INT, byteOffset));
                case f64 -> dups.getData().setAtIndex(ValueLayout.JAVA_DOUBLE, dstIndex, a.getData().get(ValueLayout.JAVA_DOUBLE, byteOffset));
                case bool -> dups.getData().setAtIndex(ValueLayout.JAVA_BYTE, dstIndex, a.getData().get(ValueLayout.JAVA_BYTE, byteOffset));
                default -> throw new UnsupportedOperationException("This dtype " + a.getDType() + " doesn't support this method");
            }
            dstIndex++;
            srcIter.next();
        }
        return dups;
    }

    /**
     * Casts elements of the tensor to a target {@link DType}.
     */
    public static NDArray cast(NDArray a, DType target) {
        if (a.getDType() == target) {
            return a;
        }

        NDArray safeThis = a.isContiguous() ? a : a.contiguous();
        NDArray res = JNum.zeros(target, safeThis.internalShapeUnsafe());
        for (long i = 0; i < safeThis.getSize(); i++) {
            double val = switch (safeThis.getDType()) {
                case f32 -> safeThis.getData().getAtIndex(ValueLayout.JAVA_FLOAT, i);
                case f64 -> safeThis.getData().getAtIndex(ValueLayout.JAVA_DOUBLE, i);
                case i32 -> safeThis.getData().getAtIndex(ValueLayout.JAVA_INT, i);
                case bool -> safeThis.getData().getAtIndex(ValueLayout.JAVA_BYTE, i) != 0 ? 1.0 : 0.0;
                default -> throw new UnsupportedOperationException("This dtype " + safeThis.getDType() + " doesn't support this method");
            };
            switch (target) {
                case f32 -> res.getData().setAtIndex(ValueLayout.JAVA_FLOAT, i, (float) val);
                case f64 -> res.getData().setAtIndex(ValueLayout.JAVA_DOUBLE, i, val);
                case i32 -> res.getData().setAtIndex(ValueLayout.JAVA_INT, i, (int) val);
                case bool -> res.getData().setAtIndex(ValueLayout.JAVA_BYTE, i, (byte) (val != 0.0 ? 1 : 0));
                default -> throw new UnsupportedOperationException("This dtype " + target + " doesn't support this method");
            }
        }
        return res;
    }
}