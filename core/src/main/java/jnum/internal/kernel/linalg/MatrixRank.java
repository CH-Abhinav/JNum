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

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import jnum.DType;
import jnum.NDArray;

/**
 * Matrix rank computation kernel based on SVD singular value thresholding.
 *
 * <p>Counts singular values greater than the relative numerical tolerance:
 * {@code tol = max(M, N) * eps * max(S)}.</p>
 */
public final class MatrixRank {

    private MatrixRank() {
        throw new AssertionError("MatrixRank kernel cannot be instantiated.");
    }

    /** Computes numerical rank with automatic machine-precision cutoff tolerance. */
    public static int matrixRank(NDArray a, Arena arena) {
        return matrixRank(a, -1.0, arena);
    }

    public static int matrixRank(NDArray a, double tol, Arena arena) {
        if (a.dim() != 2) {
            throw new IllegalArgumentException("matrix_rank requires a 2D matrix, got shape: " + a.shapeString());
        }

        int m = (int) a.internalShapeUnsafe()[0];
        int n = (int) a.internalShapeUnsafe()[1];
        int k = Math.min(m, n);
        if (k == 0) return 0;

        var svd = SVD.svd(a, arena);
        MemorySegment sSeg = svd.s().getData();
        DType dtype = a.getDType();

        int rank = 0;
        if (dtype == DType.f32) {
            float maxSigma = sSeg.getAtIndex(ValueLayout.JAVA_FLOAT, 0);
            float threshold = tol >= 0.0 ? (float) tol : Math.max(m, n) * 1.1920929e-7f * maxSigma;

            for (int i = 0; i < k; i++) {
                if (sSeg.getAtIndex(ValueLayout.JAVA_FLOAT, i) > threshold) {
                    rank++;
                }
            }
        } else {
            double maxSigma = sSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, 0);
            double threshold = tol >= 0.0 ? tol : Math.max(m, n) * 2.220446049250313e-16 * maxSigma;

            for (int i = 0; i < k; i++) {
                if (sSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, i) > threshold) {
                    rank++;
                }
            }
        }
        return rank;
    }
}