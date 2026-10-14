package jnum.internal.kernel.linalg.matmul;

import static jnum.internal.Constants.*;

import java.lang.foreign.MemorySegment;
import jnum.NDArray;

/**
 * Tier 2: Single-Threaded BLIS Cache-Blocked Kernel (128 < maxDim <= 256).
 * Employs standard GotoBLIS 3-loop cache tiling (jc -> pc -> ic) to ensure
 * packed panels reside strictly in L2 and L1 caches without multi-threaded
 * ForkJoin scheduling overhead.
 */
public final class MatMulTier2BlisSingle {

    private MatMulTier2BlisSingle() {
        throw new AssertionError("Utility class");
    }

    // =========================================================================
    // FLOAT BLIS SINGLE-THREADED
    // =========================================================================

    public static void matmulFloat(NDArray a, NDArray b, NDArray resArray, int n, int m, int p) {
        long[] aStrides = a.internalStridesUnsafe();
        long[] bStrides = b.internalStridesUnsafe();
        matmulFloat(a.getData(), aStrides[0], aStrides[1],
                    b.getData(), bStrides[0], bStrides[1],
                    resArray.getData(), n, m, p);
    }

    public static void matmulFloat(MemorySegment A, long a_s0, long a_s1,
                                   MemorySegment B, long b_s0, long b_s1,
                                   MemorySegment C, int n, int m, int p) {
        if (IS_AARCH64) {
            MemorySegment pB = tlPackedB_Aarch_Float.get();
            for (int jc = 0; jc < p; jc += NC_AARCH) {
                int nc = Math.min(NC_AARCH, p - jc);
                for (int pc = 0; pc < m; pc += KC) {
                    int kc = Math.min(KC, m - pc);
                    boolean isFirstKBlock = (pc == 0);
                    MatMulPacking.packB_panel_Aarch_Float(B, pB, pc, jc, kc, nc, b_s0, b_s1);
                    for (int ic = 0; ic < n; ic += MC) {
                        int mc = Math.min(MC, n - ic);
                        MemorySegment pA = tlPackedA_Aarch_Float.get();
                        MatMulPacking.packA_panel_Aarch_Float(A, pA, ic, mc, pc, kc, a_s0, a_s1);
                        MatMulMicroKernels.gebpMacroKernel_Aarch_Float(pA, pB, C, ic, mc, jc, nc, kc, p, isFirstKBlock);
                    }
                }
            }
        } else {
            MemorySegment pB = tlPackedB_Arm_Float.get();
            for (int jc = 0; jc < p; jc += NC_ARM) {
                int nc = Math.min(NC_ARM, p - jc);
                for (int pc = 0; pc < m; pc += KC) {
                    int kc = Math.min(KC, m - pc);
                    boolean isFirstKBlock = (pc == 0);
                    MatMulPacking.packB_panel_Arm_Float(B, pB, pc, jc, kc, nc, b_s0, b_s1);
                    for (int ic = 0; ic < n; ic += MC) {
                        int mc = Math.min(MC, n - ic);
                        MemorySegment pA = tlPackedA_Arm_Float.get();
                        MatMulPacking.packA_panel_Arm_Float(A, pA, ic, mc, pc, kc, a_s0, a_s1);
                        MatMulMicroKernels.gebpMacroKernel_Arm_Float(pA, pB, C, ic, mc, jc, nc, kc, p, isFirstKBlock);
                    }
                }
            }
        }
    }

    // =========================================================================
    // DOUBLE BLIS SINGLE-THREADED
    // =========================================================================

    public static void matmulDouble(NDArray a, NDArray b, NDArray resArray, int n, int m, int p) {
        long[] aStrides = a.internalStridesUnsafe();
        long[] bStrides = b.internalStridesUnsafe();
        matmulDouble(a.getData(), aStrides[0], aStrides[1],
                     b.getData(), bStrides[0], bStrides[1],
                     resArray.getData(), n, m, p);
    }

    public static void matmulDouble(MemorySegment A, long a_s0, long a_s1,
                                    MemorySegment B, long b_s0, long b_s1,
                                    MemorySegment C, int n, int m, int p) {
        if (IS_AARCH64) {
            MemorySegment pB = tlPackedB_Aarch_Double.get();
            for (int jc = 0; jc < p; jc += NC_AARCH) {
                int nc = Math.min(NC_AARCH, p - jc);
                for (int pc = 0; pc < m; pc += KC) {
                    int kc = Math.min(KC, m - pc);
                    boolean isFirstKBlock = (pc == 0);
                    MatMulPacking.packB_panel_Aarch_Double(B, pB, pc, jc, kc, nc, b_s0, b_s1);
                    for (int ic = 0; ic < n; ic += MC) {
                        int mc = Math.min(MC, n - ic);
                        MemorySegment pA = tlPackedA_Aarch_Double.get();
                        MatMulPacking.packA_panel_Aarch_Double(A, pA, ic, mc, pc, kc, a_s0, a_s1);
                        MatMulMicroKernels.gebpMacroKernel_Aarch_Double(pA, pB, C, ic, mc, jc, nc, kc, p, isFirstKBlock);
                    }
                }
            }
        } else {
            MemorySegment pB = tlPackedB_Arm_Double.get();
            for (int jc = 0; jc < p; jc += NC_ARM) {
                int nc = Math.min(NC_ARM, p - jc);
                for (int pc = 0; pc < m; pc += KC) {
                    int kc = Math.min(KC, m - pc);
                    boolean isFirstKBlock = (pc == 0);
                    MatMulPacking.packB_panel_Arm_Double(B, pB, pc, jc, kc, nc, b_s0, b_s1);
                    for (int ic = 0; ic < n; ic += MC) {
                        int mc = Math.min(MC, n - ic);
                        MemorySegment pA = tlPackedA_Arm_Double.get();
                        MatMulPacking.packA_panel_Arm_Double(A, pA, ic, mc, pc, kc, a_s0, a_s1);
                        MatMulMicroKernels.gebpMacroKernel_Arm_Double(pA, pB, C, ic, mc, jc, nc, kc, p, isFirstKBlock);
                    }
                }
            }
        }
    }

    // =========================================================================
    // INT BLIS SINGLE-THREADED
    // =========================================================================

    public static void matmulInt(NDArray a, NDArray b, NDArray resArray, int n, int m, int p) {
        long[] aStrides = a.internalStridesUnsafe();
        long[] bStrides = b.internalStridesUnsafe();
        matmulInt(a.getData(), aStrides[0], aStrides[1],
                  b.getData(), bStrides[0], bStrides[1],
                  resArray.getData(), n, m, p);
    }

    public static void matmulInt(MemorySegment A, long a_s0, long a_s1,
                                 MemorySegment B, long b_s0, long b_s1,
                                 MemorySegment C, int n, int m, int p) {
        if (IS_AARCH64) {
            MemorySegment pB = tlPackedB_Aarch_Int.get();
            for (int jc = 0; jc < p; jc += NC_AARCH) {
                int nc = Math.min(NC_AARCH, p - jc);
                for (int pc = 0; pc < m; pc += KC) {
                    int kc = Math.min(KC, m - pc);
                    boolean isFirstKBlock = (pc == 0);
                    MatMulPacking.packB_panel_Aarch_Int(B, pB, pc, jc, kc, nc, b_s0, b_s1);
                    for (int ic = 0; ic < n; ic += MC) {
                        int mc = Math.min(MC, n - ic);
                        MemorySegment pA = tlPackedA_Aarch_Int.get();
                        MatMulPacking.packA_panel_Aarch_Int(A, pA, ic, mc, pc, kc, a_s0, a_s1);
                        MatMulMicroKernels.gebpMacroKernel_Aarch_Int(pA, pB, C, ic, mc, jc, nc, kc, p, isFirstKBlock);
                    }
                }
            }
        } else {
            MemorySegment pB = tlPackedB_Arm_Int.get();
            for (int jc = 0; jc < p; jc += NC_ARM) {
                int nc = Math.min(NC_ARM, p - jc);
                for (int pc = 0; pc < m; pc += KC) {
                    int kc = Math.min(KC, m - pc);
                    boolean isFirstKBlock = (pc == 0);
                    MatMulPacking.packB_panel_Arm_Int(B, pB, pc, jc, kc, nc, b_s0, b_s1);
                    for (int ic = 0; ic < n; ic += MC) {
                        int mc = Math.min(MC, n - ic);
                        MemorySegment pA = tlPackedA_Arm_Int.get();
                        MatMulPacking.packA_panel_Arm_Int(A, pA, ic, mc, pc, kc, a_s0, a_s1);
                        MatMulMicroKernels.gebpMacroKernel_Arm_Int(pA, pB, C, ic, mc, jc, nc, kc, p, isFirstKBlock);
                    }
                }
            }
        }
    }
}
