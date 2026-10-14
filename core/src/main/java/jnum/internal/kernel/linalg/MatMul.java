package jnum.internal.kernel.linalg;

import jnum.NDArray;
import jnum.internal.kernel.linalg.matmul.MatMulTier0Nano;
import jnum.internal.kernel.linalg.matmul.MatMulTier1DirectTiled;
import jnum.internal.kernel.linalg.matmul.MatMulTier2BlisSingle;
import jnum.internal.kernel.linalg.matmul.MatMulTier3BlisParallel;

/**
 * Top-level Matrix Multiplication Dispatcher.
 * Routes inputs based on maximum matrix dimension to one of 4 architecture-adaptive tiers:
 * - Tier 0: Nano kernel (maxDim <= 4)
 * - Tier 1: Direct register-blocked SIMD kernel (4 < maxDim <= 128)
 * - Tier 2: Single-threaded BLIS cache-blocked kernel (128 < maxDim <= 256)
 * - Tier 3: Parallel BLIS macro-kernel (maxDim > 256)
 */
public final class MatMul {

    private MatMul() {
        throw new AssertionError("Utility class");
    }

    public static NDArray matmulFloat(NDArray a, NDArray b, NDArray resArray) {
        int n = (int) a.internalShapeUnsafe()[0];
        int m = (int) a.internalShapeUnsafe()[1];
        int p = (int) b.internalShapeUnsafe()[1];

        int maxDim = Math.max(n, Math.max(m, p));
        if (maxDim <= 4) {
            MatMulTier0Nano.matmulFloat(a, b, resArray, n, m, p);
        } else if (maxDim <= 128) {
            MatMulTier1DirectTiled.matmulFloat(a, b, resArray, n, m, p);
        } else if (maxDim <= 256) {
            MatMulTier2BlisSingle.matmulFloat(a, b, resArray, n, m, p);
        } else {
            MatMulTier3BlisParallel.matmulFloat(a, b, resArray, n, m, p);
        }
        return resArray;
    }

    public static NDArray matmulDouble(NDArray a, NDArray b, NDArray resArray) {
        int n = (int) a.internalShapeUnsafe()[0];
        int m = (int) a.internalShapeUnsafe()[1];
        int p = (int) b.internalShapeUnsafe()[1];

        int maxDim = Math.max(n, Math.max(m, p));
        if (maxDim <= 4) {
            MatMulTier0Nano.matmulDouble(a, b, resArray, n, m, p);
        } else if (maxDim <= 128) {
            MatMulTier1DirectTiled.matmulDouble(a, b, resArray, n, m, p);
        } else if (maxDim <= 256) {
            MatMulTier2BlisSingle.matmulDouble(a, b, resArray, n, m, p);
        } else {
            MatMulTier3BlisParallel.matmulDouble(a, b, resArray, n, m, p);
        }
        return resArray;
    }

    public static NDArray matmulInt(NDArray a, NDArray b, NDArray resArray) {
        int n = (int) a.internalShapeUnsafe()[0];
        int m = (int) a.internalShapeUnsafe()[1];
        int p = (int) b.internalShapeUnsafe()[1];

        int maxDim = Math.max(n, Math.max(m, p));
        if (maxDim <= 4) {
            MatMulTier0Nano.matmulInt(a, b, resArray, n, m, p);
        } else if (maxDim <= 128) {
            MatMulTier1DirectTiled.matmulInt(a, b, resArray, n, m, p);
        } else if (maxDim <= 256) {
            MatMulTier2BlisSingle.matmulInt(a, b, resArray, n, m, p);
        } else {
            MatMulTier3BlisParallel.matmulInt(a, b, resArray, n, m, p);
        }
        return resArray;
    }
}
