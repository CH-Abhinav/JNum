package jnum.internal.kernel.linalg;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import jnum.DType;
import jnum.NDArray;

public final class MatrixRank {

    private MatrixRank() {
        throw new AssertionError("MatrixRank kernel cannot be instantiated.");
    }

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

        SVD.SVDResult svd = SVD.svd(a, arena);
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