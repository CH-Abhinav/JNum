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
package jnum.internal.kernel.linalg;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import jnum.NDArray;

/**
 * Matrix trace kernel (sum of main or offset diagonal elements).
 *
 * <p>Computes sum(A[i, i + offset]) across the specified diagonal offset.</p>
 */
public final class Trace {

    private Trace() {
        throw new AssertionError("Trace kernel cannot be instantiated.");
    }

    /** Computes trace along the diagonal defined by {@code offset}. */
    public static double compute(NDArray a, int offset) {
        if (a.dim() != 2) {
            throw new IllegalArgumentException("Trace requires a 2D matrix, got shape: " + a.shapeString());
        }

        long[] shape = a.internalShapeUnsafe();
        long rows = shape[0];
        long cols = shape[1];

        long startRow = offset < 0 ? -offset : 0;
        long startCol = offset > 0 ? offset : 0;

        if (startRow >= rows || startCol >= cols) {
            return 0.0;
        }

        long diagLen = Math.min(rows - startRow, cols - startCol);
        long[] strides = a.internalStridesUnsafe();
        long s0 = strides[0];
        long s1 = strides[1];
        MemorySegment seg = a.getData();

        double sum = 0.0;
        long baseOffset = startRow * s0 + startCol * s1;
        long step = s0 + s1;

        switch (a.getDType()) {
            case f32 -> {
                for (long i = 0; i < diagLen; i++) {
                    sum += seg.getAtIndex(ValueLayout.JAVA_FLOAT, baseOffset + i * step);
                }
            }
            case f64 -> {
                for (long i = 0; i < diagLen; i++) {
                    sum += seg.getAtIndex(ValueLayout.JAVA_DOUBLE, baseOffset + i * step);
                }
            }
            case i32 -> {
                for (long i = 0; i < diagLen; i++) {
                    sum += seg.getAtIndex(ValueLayout.JAVA_INT, baseOffset + i * step);
                }
            }
            case bool -> {
                // In boolean matrices, trace is the count of true values along the diagonal
                long count = 0;
                for (long i = 0; i < diagLen; i++) {
                    if (seg.getAtIndex(ValueLayout.JAVA_BYTE, baseOffset + i * step) != 0) {
                        count++;
                    }
                }
                return (double) count;
            }
        }
        return sum;
    }
}