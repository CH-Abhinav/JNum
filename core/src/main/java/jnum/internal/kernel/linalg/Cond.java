package jnum.internal.kernel.linalg;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import jnum.DType;
import jnum.NDArray;

public final class Cond {

    private Cond() {
        throw new AssertionError("Cond kernel cannot be instantiated.");
    }

    public static double cond(NDArray a, Arena arena) {
        if (a.dim() != 2) {
            throw new IllegalArgumentException("cond requires a 2D matrix, got shape: " + a.shapeString());
        }

        int m = (int) a.internalShapeUnsafe()[0];
        int n = (int) a.internalShapeUnsafe()[1];
        int k = Math.min(m, n);
        if (k == 0) return 0.0;

        SVD.SVDResult svd = SVD.svd(a, arena);
        MemorySegment sSeg = svd.s().getData();

        if (a.getDType() == DType.f32) {
            float maxSigma = sSeg.getAtIndex(ValueLayout.JAVA_FLOAT, 0);
            float minSigma = sSeg.getAtIndex(ValueLayout.JAVA_FLOAT, k - 1);
            if (minSigma <= 1e-12f || Float.isNaN(minSigma)) {
                return Double.POSITIVE_INFINITY;
            }
            return (double) maxSigma / minSigma;
        } else {
            double maxSigma = sSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, 0);
            double minSigma = sSeg.getAtIndex(ValueLayout.JAVA_DOUBLE, k - 1);
            if (minSigma <= 1e-15 || Double.isNaN(minSigma)) {
                return Double.POSITIVE_INFINITY;
            }
            return maxSigma / minSigma;
        }
    }
}