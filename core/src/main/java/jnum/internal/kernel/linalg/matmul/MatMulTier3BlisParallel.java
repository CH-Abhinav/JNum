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
package jnum.internal.kernel.linalg.matmul;

import static jnum.internal.Constants.*;
import static jnum.internal.kernel.linalg.matmul.MatMulPacking.*;
import static jnum.internal.kernel.linalg.matmul.MatMulMicroKernels.*;

import java.lang.foreign.MemorySegment;
import java.util.concurrent.RecursiveAction;
import jnum.NDArray;

/**
 * Tier 3: Platform-Adaptive Parallel BLIS Macro-Kernel (maxDim > 256).
 *
 * Key Platform-Adaptive Optimizations:
 * 1. Dynamic Panel Bisection: Splits M into balanced multiples of MR based on
 *    AVAILABLE_CORES and L2 cache capacity via computeDynamicLeafMc(), eliminating
 *    straggler thread load imbalance across any CPU core count or hybrid P/E topology.
 * 2. Safe Shared L3 B-Panel Reuse: Packs each NC x KC panel of B into a 64-byte aligned
 *    shared global buffer (SHARED_PACKED_B_*) before parallel GEBP execution, preventing
 *    ThreadLocal buffer corruption during work-stealing on large matrices.
 * 3. Vector-Width Agnostic Routing: Automatically uses the universal 6 x (2*VL) kernel
 *    across SSE, AVX2, AVX-512, 128-bit ARM NEON, SVE, and RVV.
 */
public final class MatMulTier3BlisParallel {

    private MatMulTier3BlisParallel() {
        throw new AssertionError("Utility class");
    }

    // =========================================================================
    // TOP-LEVEL DISPATCHERS
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
        if (USE_AARCH64_8X12_KERNEL) {
            blisAarchMacro_Float(A, a_s0, a_s1, B, b_s0, b_s1, C, n, m, p);
        } else {
            blisArmMacro_Float(A, a_s0, a_s1, B, b_s0, b_s1, C, n, m, p);
        }
    }

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
        if (USE_AARCH64_8X12_KERNEL) {
            blisAarchMacro_Double(A, a_s0, a_s1, B, b_s0, b_s1, C, n, m, p);
        } else {
            blisArmMacro_Double(A, a_s0, a_s1, B, b_s0, b_s1, C, n, m, p);
        }
    }

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
        if (USE_AARCH64_8X12_KERNEL) {
            blisAarchMacro_Int(A, a_s0, a_s1, B, b_s0, b_s1, C, n, m, p);
        } else {
            blisArmMacro_Int(A, a_s0, a_s1, B, b_s0, b_s1, C, n, m, p);
        }
    }

    // =========================================================================
    // 1. FLOAT (FP32) PARALLEL BLIS MACRO-KERNELS
    // =========================================================================

    public static void blisArmMacro_Float(MemorySegment A, long a_s0, long a_s1,
                                          MemorySegment B, long b_s0, long b_s1,
                                          MemorySegment C, int n, int m, int p) {
        final MemorySegment pB = SHARED_PACKED_B_FLOAT;
        final int leafMc = computeDynamicLeafMc(n, MR, MC);
        for (int jc = 0; jc < p; jc += NC_ARM) {
            int nc = Math.min(NC_ARM, p - jc);
            for (int pc = 0; pc < m; pc += KC) {
                int kc = Math.min(KC, m - pc);
                boolean isFirstKBlock = (pc == 0);
                packB_panel_Arm_Float(B, pB, pc, jc, kc, nc, b_s0, b_s1);
                POOL.invoke(new GEBPTask_Arm_Float(
                        A, a_s0, a_s1, pB, C, n, m, p, 0, n, pc, kc, jc, nc, isFirstKBlock, leafMc));
            }
        }
    }

    public static void blisAarchMacro_Float(MemorySegment A, long a_s0, long a_s1,
                                            MemorySegment B, long b_s0, long b_s1,
                                            MemorySegment C, int n, int m, int p) {
        final MemorySegment pB = SHARED_PACKED_B_FLOAT;
        final int leafMc = computeDynamicLeafMc(n, 8, MC);
        for (int jc = 0; jc < p; jc += NC_AARCH) {
            int nc = Math.min(NC_AARCH, p - jc);
            for (int pc = 0; pc < m; pc += KC) {
                int kc = Math.min(KC, m - pc);
                boolean isFirstKBlock = (pc == 0);
                packB_panel_Aarch_Float(B, pB, pc, jc, kc, nc, b_s0, b_s1);
                POOL.invoke(new GEBPTask_Aarch_Float(
                        A, a_s0, a_s1, pB, C, n, m, p, 0, n, pc, kc, jc, nc, isFirstKBlock, leafMc));
            }
        }
    }

    public static final class GEBPTask_Arm_Float extends RecursiveAction {
        final MemorySegment A, pB, C;
        final long a_s0, a_s1;
        final int n, m, p_cols, rowStart, rowEnd, pc, kc, jc, nc, leafMc;
        final boolean isFirstKBlock;

        GEBPTask_Arm_Float(MemorySegment A, long a_s0, long a_s1, MemorySegment pB, MemorySegment C,
                           int n, int m, int p_cols, int rowStart, int rowEnd,
                           int pc, int kc, int jc, int nc, boolean isFirstKBlock) {
            this(A, a_s0, a_s1, pB, C, n, m, p_cols, rowStart, rowEnd,
                    pc, kc, jc, nc, isFirstKBlock, computeDynamicLeafMc(n, MR, MC));
        }

        GEBPTask_Arm_Float(MemorySegment A, long a_s0, long a_s1, MemorySegment pB, MemorySegment C,
                           int n, int m, int p_cols, int rowStart, int rowEnd,
                           int pc, int kc, int jc, int nc, boolean isFirstKBlock, int leafMc) {
            this.A = A; this.a_s0 = a_s0; this.a_s1 = a_s1;
            this.pB = pB; this.C = C; this.n = n; this.m = m; this.p_cols = p_cols;
            this.rowStart = rowStart; this.rowEnd = rowEnd;
            this.pc = pc; this.kc = kc; this.jc = jc; this.nc = nc;
            this.isFirstKBlock = isFirstKBlock;
            this.leafMc = leafMc;
        }

        @Override
        protected void compute() {
            int mc = rowEnd - rowStart;
            if (mc <= leafMc) {
                MemorySegment pA = tlPackedA_Arm_Float.get();
                packA_panel_Arm_Float(A, pA, rowStart, mc, pc, kc, a_s0, a_s1);
                gebpMacroKernel_Arm_Float(pA, pB, C, rowStart, mc, jc, nc, kc, p_cols, isFirstKBlock);
            } else {
                int panels = (mc + MR - 1) / MR;
                int halfPanels = Math.max(1, panels >>> 1);
                int mid = rowStart + halfPanels * MR;
                invokeAll(
                        new GEBPTask_Arm_Float(A, a_s0, a_s1, pB, C, n, m, p_cols, rowStart, mid, pc, kc, jc, nc, isFirstKBlock, leafMc),
                        new GEBPTask_Arm_Float(A, a_s0, a_s1, pB, C, n, m, p_cols, mid, rowEnd, pc, kc, jc, nc, isFirstKBlock, leafMc)
                );
            }
        }
    }

    public static final class GEBPTask_Aarch_Float extends RecursiveAction {
        final MemorySegment A, pB, C;
        final long a_s0, a_s1;
        final int n, m, p_cols, rowStart, rowEnd, pc, kc, jc, nc, leafMc;
        final boolean isFirstKBlock;

        GEBPTask_Aarch_Float(MemorySegment A, long a_s0, long a_s1, MemorySegment pB, MemorySegment C,
                             int n, int m, int p_cols, int rowStart, int rowEnd,
                             int pc, int kc, int jc, int nc, boolean isFirstKBlock) {
            this(A, a_s0, a_s1, pB, C, n, m, p_cols, rowStart, rowEnd,
                    pc, kc, jc, nc, isFirstKBlock, computeDynamicLeafMc(n, 8, MC));
        }

        GEBPTask_Aarch_Float(MemorySegment A, long a_s0, long a_s1, MemorySegment pB, MemorySegment C,
                             int n, int m, int p_cols, int rowStart, int rowEnd,
                             int pc, int kc, int jc, int nc, boolean isFirstKBlock, int leafMc) {
            this.A = A; this.a_s0 = a_s0; this.a_s1 = a_s1;
            this.pB = pB; this.C = C; this.n = n; this.m = m; this.p_cols = p_cols;
            this.rowStart = rowStart; this.rowEnd = rowEnd;
            this.pc = pc; this.kc = kc; this.jc = jc; this.nc = nc;
            this.isFirstKBlock = isFirstKBlock;
            this.leafMc = leafMc;
        }

        @Override
        protected void compute() {
            int mc = rowEnd - rowStart;
            if (mc <= leafMc) {
                MemorySegment pA = tlPackedA_Aarch_Float.get();
                packA_panel_Aarch_Float(A, pA, rowStart, mc, pc, kc, a_s0, a_s1);
                gebpMacroKernel_Aarch_Float(pA, pB, C, rowStart, mc, jc, nc, kc, p_cols, isFirstKBlock);
            } else {
                int panels = (mc + 7) / 8;
                int halfPanels = Math.max(1, panels >>> 1);
                int mid = rowStart + halfPanels * 8;
                invokeAll(
                        new GEBPTask_Aarch_Float(A, a_s0, a_s1, pB, C, n, m, p_cols, rowStart, mid, pc, kc, jc, nc, isFirstKBlock, leafMc),
                        new GEBPTask_Aarch_Float(A, a_s0, a_s1, pB, C, n, m, p_cols, mid, rowEnd, pc, kc, jc, nc, isFirstKBlock, leafMc)
                );
            }
        }
    }

    public static final class JcTask_Arm_Float extends RecursiveAction {
        final MemorySegment A, B, C;
        final long a_s0, a_s1, b_s0, b_s1;
        final int n, m, p, colStart, colEnd;

        JcTask_Arm_Float(MemorySegment A, long a_s0, long a_s1,
                         MemorySegment B, long b_s0, long b_s1,
                         MemorySegment C, int n, int m, int p, int colStart, int colEnd) {
            this.A = A; this.a_s0 = a_s0; this.a_s1 = a_s1;
            this.B = B; this.b_s0 = b_s0; this.b_s1 = b_s1;
            this.C = C; this.n = n; this.m = m; this.p = p;
            this.colStart = colStart; this.colEnd = colEnd;
        }

        @Override
        protected void compute() {
            MemorySegment pB = SHARED_PACKED_B_FLOAT;
            for (int jc = colStart; jc < colEnd; jc += NC_ARM) {
                int nc = Math.min(NC_ARM, colEnd - jc);
                for (int pc = 0; pc < m; pc += KC) {
                    int kc = Math.min(KC, m - pc);
                    boolean isFirstKBlock = (pc == 0);
                    packB_panel_Arm_Float(B, pB, pc, jc, kc, nc, b_s0, b_s1);
                    new GEBPTask_Arm_Float(A, a_s0, a_s1, pB, C, n, m, p, 0, n, pc, kc, jc, nc, isFirstKBlock).invoke();
                }
            }
        }
    }

    public static final class JcTask_Aarch_Float extends RecursiveAction {
        final MemorySegment A, B, C;
        final long a_s0, a_s1, b_s0, b_s1;
        final int n, m, p, colStart, colEnd;

        JcTask_Aarch_Float(MemorySegment A, long a_s0, long a_s1,
                           MemorySegment B, long b_s0, long b_s1,
                           MemorySegment C, int n, int m, int p, int colStart, int colEnd) {
            this.A = A; this.a_s0 = a_s0; this.a_s1 = a_s1;
            this.B = B; this.b_s0 = b_s0; this.b_s1 = b_s1;
            this.C = C; this.n = n; this.m = m; this.p = p;
            this.colStart = colStart; this.colEnd = colEnd;
        }

        @Override
        protected void compute() {
            MemorySegment pB = SHARED_PACKED_B_FLOAT;
            for (int jc = colStart; jc < colEnd; jc += NC_AARCH) {
                int nc = Math.min(NC_AARCH, colEnd - jc);
                for (int pc = 0; pc < m; pc += KC) {
                    int kc = Math.min(KC, m - pc);
                    boolean isFirstKBlock = (pc == 0);
                    packB_panel_Aarch_Float(B, pB, pc, jc, kc, nc, b_s0, b_s1);
                    new GEBPTask_Aarch_Float(A, a_s0, a_s1, pB, C, n, m, p, 0, n, pc, kc, jc, nc, isFirstKBlock).invoke();
                }
            }
        }
    }

    // =========================================================================
    // 2. DOUBLE (FP64) PARALLEL BLIS MACRO-KERNELS
    // =========================================================================

    public static void blisArmMacro_Double(MemorySegment A, long a_s0, long a_s1,
                                           MemorySegment B, long b_s0, long b_s1,
                                           MemorySegment C, int n, int m, int p) {
        final MemorySegment pB = SHARED_PACKED_B_DOUBLE;
        final int leafMc = computeDynamicLeafMc(n, MR, MC);
        for (int jc = 0; jc < p; jc += NC_ARM) {
            int nc = Math.min(NC_ARM, p - jc);
            for (int pc = 0; pc < m; pc += KC) {
                int kc = Math.min(KC, m - pc);
                boolean isFirstKBlock = (pc == 0);
                packB_panel_Arm_Double(B, pB, pc, jc, kc, nc, b_s0, b_s1);
                POOL.invoke(new GEBPTask_Arm_Double(
                        A, a_s0, a_s1, pB, C, n, m, p, 0, n, pc, kc, jc, nc, isFirstKBlock, leafMc));
            }
        }
    }

    public static void blisAarchMacro_Double(MemorySegment A, long a_s0, long a_s1,
                                             MemorySegment B, long b_s0, long b_s1,
                                             MemorySegment C, int n, int m, int p) {
        final MemorySegment pB = SHARED_PACKED_B_DOUBLE;
        final int leafMc = computeDynamicLeafMc(n, 8, MC);
        for (int jc = 0; jc < p; jc += NC_AARCH) {
            int nc = Math.min(NC_AARCH, p - jc);
            for (int pc = 0; pc < m; pc += KC) {
                int kc = Math.min(KC, m - pc);
                boolean isFirstKBlock = (pc == 0);
                packB_panel_Aarch_Double(B, pB, pc, jc, kc, nc, b_s0, b_s1);
                POOL.invoke(new GEBPTask_Aarch_Double(
                        A, a_s0, a_s1, pB, C, n, m, p, 0, n, pc, kc, jc, nc, isFirstKBlock, leafMc));
            }
        }
    }

    public static final class GEBPTask_Arm_Double extends RecursiveAction {
        final MemorySegment A, pB, C;
        final long a_s0, a_s1;
        final int n, m, p_cols, rowStart, rowEnd, pc, kc, jc, nc, leafMc;
        final boolean isFirstKBlock;

        GEBPTask_Arm_Double(MemorySegment A, long a_s0, long a_s1, MemorySegment pB, MemorySegment C,
                            int n, int m, int p_cols, int rowStart, int rowEnd,
                            int pc, int kc, int jc, int nc, boolean isFirstKBlock) {
            this(A, a_s0, a_s1, pB, C, n, m, p_cols, rowStart, rowEnd,
                    pc, kc, jc, nc, isFirstKBlock, computeDynamicLeafMc(n, MR, MC));
        }

        GEBPTask_Arm_Double(MemorySegment A, long a_s0, long a_s1, MemorySegment pB, MemorySegment C,
                            int n, int m, int p_cols, int rowStart, int rowEnd,
                            int pc, int kc, int jc, int nc, boolean isFirstKBlock, int leafMc) {
            this.A = A; this.a_s0 = a_s0; this.a_s1 = a_s1;
            this.pB = pB; this.C = C; this.n = n; this.m = m; this.p_cols = p_cols;
            this.rowStart = rowStart; this.rowEnd = rowEnd;
            this.pc = pc; this.kc = kc; this.jc = jc; this.nc = nc;
            this.isFirstKBlock = isFirstKBlock;
            this.leafMc = leafMc;
        }

        @Override
        protected void compute() {
            int mc = rowEnd - rowStart;
            if (mc <= leafMc) {
                MemorySegment pA = tlPackedA_Arm_Double.get();
                packA_panel_Arm_Double(A, pA, rowStart, mc, pc, kc, a_s0, a_s1);
                gebpMacroKernel_Arm_Double(pA, pB, C, rowStart, mc, jc, nc, kc, p_cols, isFirstKBlock);
            } else {
                int panels = (mc + MR - 1) / MR;
                int halfPanels = Math.max(1, panels >>> 1);
                int mid = rowStart + halfPanels * MR;
                invokeAll(
                        new GEBPTask_Arm_Double(A, a_s0, a_s1, pB, C, n, m, p_cols, rowStart, mid, pc, kc, jc, nc, isFirstKBlock, leafMc),
                        new GEBPTask_Arm_Double(A, a_s0, a_s1, pB, C, n, m, p_cols, mid, rowEnd, pc, kc, jc, nc, isFirstKBlock, leafMc)
                );
            }
        }
    }

    public static final class GEBPTask_Aarch_Double extends RecursiveAction {
        final MemorySegment A, pB, C;
        final long a_s0, a_s1;
        final int n, m, p_cols, rowStart, rowEnd, pc, kc, jc, nc, leafMc;
        final boolean isFirstKBlock;

        GEBPTask_Aarch_Double(MemorySegment A, long a_s0, long a_s1, MemorySegment pB, MemorySegment C,
                              int n, int m, int p_cols, int rowStart, int rowEnd,
                              int pc, int kc, int jc, int nc, boolean isFirstKBlock) {
            this(A, a_s0, a_s1, pB, C, n, m, p_cols, rowStart, rowEnd,
                    pc, kc, jc, nc, isFirstKBlock, computeDynamicLeafMc(n, 8, MC));
        }

        GEBPTask_Aarch_Double(MemorySegment A, long a_s0, long a_s1, MemorySegment pB, MemorySegment C,
                              int n, int m, int p_cols, int rowStart, int rowEnd,
                              int pc, int kc, int jc, int nc, boolean isFirstKBlock, int leafMc) {
            this.A = A; this.a_s0 = a_s0; this.a_s1 = a_s1;
            this.pB = pB; this.C = C; this.n = n; this.m = m; this.p_cols = p_cols;
            this.rowStart = rowStart; this.rowEnd = rowEnd;
            this.pc = pc; this.kc = kc; this.jc = jc; this.nc = nc;
            this.isFirstKBlock = isFirstKBlock;
            this.leafMc = leafMc;
        }

        @Override
        protected void compute() {
            int mc = rowEnd - rowStart;
            if (mc <= leafMc) {
                MemorySegment pA = tlPackedA_Aarch_Double.get();
                packA_panel_Aarch_Double(A, pA, rowStart, mc, pc, kc, a_s0, a_s1);
                gebpMacroKernel_Aarch_Double(pA, pB, C, rowStart, mc, jc, nc, kc, p_cols, isFirstKBlock);
            } else {
                int panels = (mc + 7) / 8;
                int halfPanels = Math.max(1, panels >>> 1);
                int mid = rowStart + halfPanels * 8;
                invokeAll(
                        new GEBPTask_Aarch_Double(A, a_s0, a_s1, pB, C, n, m, p_cols, rowStart, mid, pc, kc, jc, nc, isFirstKBlock, leafMc),
                        new GEBPTask_Aarch_Double(A, a_s0, a_s1, pB, C, n, m, p_cols, mid, rowEnd, pc, kc, jc, nc, isFirstKBlock, leafMc)
                );
            }
        }
    }

    public static final class JcTask_Arm_Double extends RecursiveAction {
        final MemorySegment A, B, C;
        final long a_s0, a_s1, b_s0, b_s1;
        final int n, m, p, colStart, colEnd;

        JcTask_Arm_Double(MemorySegment A, long a_s0, long a_s1,
                          MemorySegment B, long b_s0, long b_s1,
                          MemorySegment C, int n, int m, int p, int colStart, int colEnd) {
            this.A = A; this.a_s0 = a_s0; this.a_s1 = a_s1;
            this.B = B; this.b_s0 = b_s0; this.b_s1 = b_s1;
            this.C = C; this.n = n; this.m = m; this.p = p;
            this.colStart = colStart; this.colEnd = colEnd;
        }

        @Override
        protected void compute() {
            MemorySegment pB = SHARED_PACKED_B_DOUBLE;
            for (int jc = colStart; jc < colEnd; jc += NC_ARM) {
                int nc = Math.min(NC_ARM, colEnd - jc);
                for (int pc = 0; pc < m; pc += KC) {
                    int kc = Math.min(KC, m - pc);
                    boolean isFirstKBlock = (pc == 0);
                    packB_panel_Arm_Double(B, pB, pc, jc, kc, nc, b_s0, b_s1);
                    new GEBPTask_Arm_Double(A, a_s0, a_s1, pB, C, n, m, p, 0, n, pc, kc, jc, nc, isFirstKBlock).invoke();
                }
            }
        }
    }

    public static final class JcTask_Aarch_Double extends RecursiveAction {
        final MemorySegment A, B, C;
        final long a_s0, a_s1, b_s0, b_s1;
        final int n, m, p, colStart, colEnd;

        JcTask_Aarch_Double(MemorySegment A, long a_s0, long a_s1,
                            MemorySegment B, long b_s0, long b_s1,
                            MemorySegment C, int n, int m, int p, int colStart, int colEnd) {
            this.A = A; this.a_s0 = a_s0; this.a_s1 = a_s1;
            this.B = B; this.b_s0 = b_s0; this.b_s1 = b_s1;
            this.C = C; this.n = n; this.m = m; this.p = p;
            this.colStart = colStart; this.colEnd = colEnd;
        }

        @Override
        protected void compute() {
            MemorySegment pB = SHARED_PACKED_B_DOUBLE;
            for (int jc = colStart; jc < colEnd; jc += NC_AARCH) {
                int nc = Math.min(NC_AARCH, colEnd - jc);
                for (int pc = 0; pc < m; pc += KC) {
                    int kc = Math.min(KC, m - pc);
                    boolean isFirstKBlock = (pc == 0);
                    packB_panel_Aarch_Double(B, pB, pc, jc, kc, nc, b_s0, b_s1);
                    new GEBPTask_Aarch_Double(A, a_s0, a_s1, pB, C, n, m, p, 0, n, pc, kc, jc, nc, isFirstKBlock).invoke();
                }
            }
        }
    }

    // =========================================================================
    // 3. INT (INT32) PARALLEL BLIS MACRO-KERNELS
    // =========================================================================

    public static void blisArmMacro_Int(MemorySegment A, long a_s0, long a_s1,
                                        MemorySegment B, long b_s0, long b_s1,
                                        MemorySegment C, int n, int m, int p) {
        final MemorySegment pB = SHARED_PACKED_B_INT;
        final int leafMc = computeDynamicLeafMc(n, MR, MC);
        for (int jc = 0; jc < p; jc += NC_ARM) {
            int nc = Math.min(NC_ARM, p - jc);
            for (int pc = 0; pc < m; pc += KC) {
                int kc = Math.min(KC, m - pc);
                boolean isFirstKBlock = (pc == 0);
                packB_panel_Arm_Int(B, pB, pc, jc, kc, nc, b_s0, b_s1);
                POOL.invoke(new GEBPTask_Arm_Int(
                        A, a_s0, a_s1, pB, C, n, m, p, 0, n, pc, kc, jc, nc, isFirstKBlock, leafMc));
            }
        }
    }

    public static void blisAarchMacro_Int(MemorySegment A, long a_s0, long a_s1,
                                          MemorySegment B, long b_s0, long b_s1,
                                          MemorySegment C, int n, int m, int p) {
        final MemorySegment pB = SHARED_PACKED_B_INT;
        final int leafMc = computeDynamicLeafMc(n, 8, MC);
        for (int jc = 0; jc < p; jc += NC_AARCH) {
            int nc = Math.min(NC_AARCH, p - jc);
            for (int pc = 0; pc < m; pc += KC) {
                int kc = Math.min(KC, m - pc);
                boolean isFirstKBlock = (pc == 0);
                packB_panel_Aarch_Int(B, pB, pc, jc, kc, nc, b_s0, b_s1);
                POOL.invoke(new GEBPTask_Aarch_Int(
                        A, a_s0, a_s1, pB, C, n, m, p, 0, n, pc, kc, jc, nc, isFirstKBlock, leafMc));
            }
        }
    }

    public static final class GEBPTask_Arm_Int extends RecursiveAction {
        final MemorySegment A, pB, C;
        final long a_s0, a_s1;
        final int n, m, p_cols, rowStart, rowEnd, pc, kc, jc, nc, leafMc;
        final boolean isFirstKBlock;

        GEBPTask_Arm_Int(MemorySegment A, long a_s0, long a_s1, MemorySegment pB, MemorySegment C,
                         int n, int m, int p_cols, int rowStart, int rowEnd,
                         int pc, int kc, int jc, int nc, boolean isFirstKBlock) {
            this(A, a_s0, a_s1, pB, C, n, m, p_cols, rowStart, rowEnd,
                    pc, kc, jc, nc, isFirstKBlock, computeDynamicLeafMc(n, MR, MC));
        }

        GEBPTask_Arm_Int(MemorySegment A, long a_s0, long a_s1, MemorySegment pB, MemorySegment C,
                         int n, int m, int p_cols, int rowStart, int rowEnd,
                         int pc, int kc, int jc, int nc, boolean isFirstKBlock, int leafMc) {
            this.A = A; this.a_s0 = a_s0; this.a_s1 = a_s1;
            this.pB = pB; this.C = C; this.n = n; this.m = m; this.p_cols = p_cols;
            this.rowStart = rowStart; this.rowEnd = rowEnd;
            this.pc = pc; this.kc = kc; this.jc = jc; this.nc = nc;
            this.isFirstKBlock = isFirstKBlock;
            this.leafMc = leafMc;
        }

        @Override
        protected void compute() {
            int mc = rowEnd - rowStart;
            if (mc <= leafMc) {
                MemorySegment pA = tlPackedA_Arm_Int.get();
                packA_panel_Arm_Int(A, pA, rowStart, mc, pc, kc, a_s0, a_s1);
                gebpMacroKernel_Arm_Int(pA, pB, C, rowStart, mc, jc, nc, kc, p_cols, isFirstKBlock);
            } else {
                int panels = (mc + MR - 1) / MR;
                int halfPanels = Math.max(1, panels >>> 1);
                int mid = rowStart + halfPanels * MR;
                invokeAll(
                        new GEBPTask_Arm_Int(A, a_s0, a_s1, pB, C, n, m, p_cols, rowStart, mid, pc, kc, jc, nc, isFirstKBlock, leafMc),
                        new GEBPTask_Arm_Int(A, a_s0, a_s1, pB, C, n, m, p_cols, mid, rowEnd, pc, kc, jc, nc, isFirstKBlock, leafMc)
                );
            }
        }
    }

    public static final class GEBPTask_Aarch_Int extends RecursiveAction {
        final MemorySegment A, pB, C;
        final long a_s0, a_s1;
        final int n, m, p_cols, rowStart, rowEnd, pc, kc, jc, nc, leafMc;
        final boolean isFirstKBlock;

        GEBPTask_Aarch_Int(MemorySegment A, long a_s0, long a_s1, MemorySegment pB, MemorySegment C,
                           int n, int m, int p_cols, int rowStart, int rowEnd,
                           int pc, int kc, int jc, int nc, boolean isFirstKBlock) {
            this(A, a_s0, a_s1, pB, C, n, m, p_cols, rowStart, rowEnd,
                    pc, kc, jc, nc, isFirstKBlock, computeDynamicLeafMc(n, 8, MC));
        }

        GEBPTask_Aarch_Int(MemorySegment A, long a_s0, long a_s1, MemorySegment pB, MemorySegment C,
                           int n, int m, int p_cols, int rowStart, int rowEnd,
                           int pc, int kc, int jc, int nc, boolean isFirstKBlock, int leafMc) {
            this.A = A; this.a_s0 = a_s0; this.a_s1 = a_s1;
            this.pB = pB; this.C = C; this.n = n; this.m = m; this.p_cols = p_cols;
            this.rowStart = rowStart; this.rowEnd = rowEnd;
            this.pc = pc; this.kc = kc; this.jc = jc; this.nc = nc;
            this.isFirstKBlock = isFirstKBlock;
            this.leafMc = leafMc;
        }

        @Override
        protected void compute() {
            int mc = rowEnd - rowStart;
            if (mc <= leafMc) {
                MemorySegment pA = tlPackedA_Aarch_Int.get();
                packA_panel_Aarch_Int(A, pA, rowStart, mc, pc, kc, a_s0, a_s1);
                gebpMacroKernel_Aarch_Int(pA, pB, C, rowStart, mc, jc, nc, kc, p_cols, isFirstKBlock);
            } else {
                int panels = (mc + 7) / 8;
                int halfPanels = Math.max(1, panels >>> 1);
                int mid = rowStart + halfPanels * 8;
                invokeAll(
                        new GEBPTask_Aarch_Int(A, a_s0, a_s1, pB, C, n, m, p_cols, rowStart, mid, pc, kc, jc, nc, isFirstKBlock, leafMc),
                        new GEBPTask_Aarch_Int(A, a_s0, a_s1, pB, C, n, m, p_cols, mid, rowEnd, pc, kc, jc, nc, isFirstKBlock, leafMc)
                );
            }
        }
    }

    public static final class JcTask_Arm_Int extends RecursiveAction {
        final MemorySegment A, B, C;
        final long a_s0, a_s1, b_s0, b_s1;
        final int n, m, p, colStart, colEnd;

        JcTask_Arm_Int(MemorySegment A, long a_s0, long a_s1,
                       MemorySegment B, long b_s0, long b_s1,
                       MemorySegment C, int n, int m, int p, int colStart, int colEnd) {
            this.A = A; this.a_s0 = a_s0; this.a_s1 = a_s1;
            this.B = B; this.b_s0 = b_s0; this.b_s1 = b_s1;
            this.C = C; this.n = n; this.m = m; this.p = p;
            this.colStart = colStart; this.colEnd = colEnd;
        }

        @Override
        protected void compute() {
            MemorySegment pB = SHARED_PACKED_B_INT;
            for (int jc = colStart; jc < colEnd; jc += NC_ARM) {
                int nc = Math.min(NC_ARM, colEnd - jc);
                for (int pc = 0; pc < m; pc += KC) {
                    int kc = Math.min(KC, m - pc);
                    boolean isFirstKBlock = (pc == 0);
                    packB_panel_Arm_Int(B, pB, pc, jc, kc, nc, b_s0, b_s1);
                    new GEBPTask_Arm_Int(A, a_s0, a_s1, pB, C, n, m, p, 0, n, pc, kc, jc, nc, isFirstKBlock).invoke();
                }
            }
        }
    }

    public static final class JcTask_Aarch_Int extends RecursiveAction {
        final MemorySegment A, B, C;
        final long a_s0, a_s1, b_s0, b_s1;
        final int n, m, p, colStart, colEnd;

        JcTask_Aarch_Int(MemorySegment A, long a_s0, long a_s1,
                         MemorySegment B, long b_s0, long b_s1,
                         MemorySegment C, int n, int m, int p, int colStart, int colEnd) {
            this.A = A; this.a_s0 = a_s0; this.a_s1 = a_s1;
            this.B = B; this.b_s0 = b_s0; this.b_s1 = b_s1;
            this.C = C; this.n = n; this.m = m; this.p = p;
            this.colStart = colStart; this.colEnd = colEnd;
        }

        @Override
        protected void compute() {
            MemorySegment pB = SHARED_PACKED_B_INT;
            for (int jc = colStart; jc < colEnd; jc += NC_AARCH) {
                int nc = Math.min(NC_AARCH, colEnd - jc);
                for (int pc = 0; pc < m; pc += KC) {
                    int kc = Math.min(KC, m - pc);
                    boolean isFirstKBlock = (pc == 0);
                    packB_panel_Aarch_Int(B, pB, pc, jc, kc, nc, b_s0, b_s1);
                    new GEBPTask_Aarch_Int(A, a_s0, a_s1, pB, C, n, m, p, 0, n, pc, kc, jc, nc, isFirstKBlock).invoke();
                }
            }
        }
    }
}