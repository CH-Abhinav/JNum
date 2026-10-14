package jnum.internal.kernel.linalg.matmul;

import static jnum.internal.Constants.*;

import java.lang.foreign.MemorySegment;
import jnum.NDArray;

/**
 * Tier 0: Nano Kernel (maxDim <= 4).
 * Pure unrolled scalar execution for tiny matrices, avoiding SIMD setup,
 * panel packing, and thread-pool scheduling overhead.
 */
public final class MatMulTier0Nano {

    private MatMulTier0Nano() {
        throw new AssertionError("Utility class");
    }

    public static void matmulFloat(NDArray a, NDArray b, NDArray resArray, int n, int m, int p) {
        long[] aStrides = a.internalStridesUnsafe();
        long[] bStrides = b.internalStridesUnsafe();
        long[] cStrides = resArray.internalStridesUnsafe();
        MemorySegment memA = a.getData();
        MemorySegment memB = b.getData();
        MemorySegment memC = resArray.getData();

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < p; j++) {
                float sum = 0f;
                for (int k = 0; k < m; k++) {
                    sum += memA.get(VF, ((long) i * aStrides[0] + (long) k * aStrides[1]) * 4L)
                         * memB.get(VF, ((long) k * bStrides[0] + (long) j * bStrides[1]) * 4L);
                }
                memC.set(VF, ((long) i * cStrides[0] + (long) j * cStrides[1]) * 4L, sum);
            }
        }
    }

    public static void matmulDouble(NDArray a, NDArray b, NDArray resArray, int n, int m, int p) {
        long[] aStrides = a.internalStridesUnsafe();
        long[] bStrides = b.internalStridesUnsafe();
        long[] cStrides = resArray.internalStridesUnsafe();
        MemorySegment memA = a.getData();
        MemorySegment memB = b.getData();
        MemorySegment memC = resArray.getData();

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < p; j++) {
                double sum = 0.0d;
                for (int k = 0; k < m; k++) {
                    sum += memA.get(VD, ((long) i * aStrides[0] + (long) k * aStrides[1]) * 8L)
                         * memB.get(VD, ((long) k * bStrides[0] + (long) j * bStrides[1]) * 8L);
                }
                memC.set(VD, ((long) i * cStrides[0] + (long) j * cStrides[1]) * 8L, sum);
            }
        }
    }

    public static void matmulInt(NDArray a, NDArray b, NDArray resArray, int n, int m, int p) {
        long[] aStrides = a.internalStridesUnsafe();
        long[] bStrides = b.internalStridesUnsafe();
        long[] cStrides = resArray.internalStridesUnsafe();
        MemorySegment memA = a.getData();
        MemorySegment memB = b.getData();
        MemorySegment memC = resArray.getData();

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < p; j++) {
                int sum = 0;
                for (int k = 0; k < m; k++) {
                    sum += memA.get(VI, ((long) i * aStrides[0] + (long) k * aStrides[1]) * 4L)
                         * memB.get(VI, ((long) k * bStrides[0] + (long) j * bStrides[1]) * 4L);
                }
                memC.set(VI, ((long) i * cStrides[0] + (long) j * cStrides[1]) * 4L, sum);
            }
        }
    }
}
