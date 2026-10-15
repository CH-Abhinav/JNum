package jnum.internal.kernel.linalg;

import static jnum.internal.Constants.AVAILABLE_CORES;
import static jnum.internal.Constants.MATMUL_PARALLEL_FLOPS_THRESHOLD;
import static jnum.internal.Constants.MR;

import jnum.NDArray;
import jnum.internal.kernel.linalg.matmul.MatMulTier0Nano;
import jnum.internal.kernel.linalg.matmul.MatMulTier1DirectTiled;
import jnum.internal.kernel.linalg.matmul.MatMulTier2BlisSingle;
import jnum.internal.kernel.linalg.matmul.MatMulTier3BlisParallel;

/**
 * Top-level Matrix Multiplication Dispatcher.
 * Routes inputs using both matrix dimensions and total arithmetic work (FLOPs)
 * across 4 hardware-adaptive execution tiers:
 * - Tier 0: Nano scalar kernel (maxDim <= 4)
 * - Tier 1: Direct register-blocked SIMD kernel (4 < maxDim <= 128 and fits in L1/L2)
 * - Tier 2: Single-threaded BLIS cache-blocked kernel (128 < maxDim <= 256 or low-FLOP/single-core)
 * - Tier 3: Parallel BLIS macro-kernel (maxDim > 256 with parallel FLOP volume)
 */
public final class MatMul {

    private static final long TIER1_MAX_OPS = 128L * 128L * 128L;
    private static final long TIER2_MAX_OPS = 256L * 256L * 256L;

    private MatMul() {
        throw new AssertionError("Utility class");
    }

    private static int selectTier(int n, int m, int p) {
        int maxDim = Math.max(n, Math.max(m, p));
        if (maxDim <= 4) {
            return 0;
        }
        long ops = (long) n * m * p;
        if (maxDim <= 128 && ops <= TIER1_MAX_OPS) {
            return 1;
        }
        // Avoid parallel ForkJoin & shared B-panel overhead on single-core CPUs,
        // low-FLOP skinny matrices, or when M has fewer than 2 micro-panels.
        if (AVAILABLE_CORES <= 1 || n < MR || ops < MATMUL_PARALLEL_FLOPS_THRESHOLD
                || (maxDim <= 256 && ops <= TIER2_MAX_OPS)) {
            return 2;
        }
        return 3;
    }

    public static NDArray matmulFloat(NDArray a, NDArray b, NDArray resArray) {
        long[] aShape = a.internalShapeUnsafe();
        long[] bShape = b.internalShapeUnsafe();
        int n = (int) aShape[0];
        int m = (int) aShape[1];
        int p = (int) bShape[1];

        switch (selectTier(n, m, p)) {
            case 0 -> MatMulTier0Nano.matmulFloat(a, b, resArray, n, m, p);
            case 1 -> MatMulTier1DirectTiled.matmulFloat(a, b, resArray, n, m, p);
            case 2 -> MatMulTier2BlisSingle.matmulFloat(a, b, resArray, n, m, p);
            default -> MatMulTier3BlisParallel.matmulFloat(a, b, resArray, n, m, p);
        }
        return resArray;
    }

    public static NDArray matmulDouble(NDArray a, NDArray b, NDArray resArray) {
        long[] aShape = a.internalShapeUnsafe();
        long[] bShape = b.internalShapeUnsafe();
        int n = (int) aShape[0];
        int m = (int) aShape[1];
        int p = (int) bShape[1];

        switch (selectTier(n, m, p)) {
            case 0 -> MatMulTier0Nano.matmulDouble(a, b, resArray, n, m, p);
            case 1 -> MatMulTier1DirectTiled.matmulDouble(a, b, resArray, n, m, p);
            case 2 -> MatMulTier2BlisSingle.matmulDouble(a, b, resArray, n, m, p);
            default -> MatMulTier3BlisParallel.matmulDouble(a, b, resArray, n, m, p);
        }
        return resArray;
    }

    public static NDArray matmulInt(NDArray a, NDArray b, NDArray resArray) {
        long[] aShape = a.internalShapeUnsafe();
        long[] bShape = b.internalShapeUnsafe();
        int n = (int) aShape[0];
        int m = (int) aShape[1];
        int p = (int) bShape[1];

        switch (selectTier(n, m, p)) {
            case 0 -> MatMulTier0Nano.matmulInt(a, b, resArray, n, m, p);
            case 1 -> MatMulTier1DirectTiled.matmulInt(a, b, resArray, n, m, p);
            case 2 -> MatMulTier2BlisSingle.matmulInt(a, b, resArray, n, m, p);
            default -> MatMulTier3BlisParallel.matmulInt(a, b, resArray, n, m, p);
        }
        return resArray;
    }
}