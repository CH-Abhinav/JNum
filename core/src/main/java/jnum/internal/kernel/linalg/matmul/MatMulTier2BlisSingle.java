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

import java.lang.foreign.MemorySegment;
import jnum.NDArray;

/**
 * Tier 2: Single-Threaded BLIS Cache-Blocked Kernel (128 < maxDim <= 256).
 *
 * Key Optimizations:
 * 1. Strictly Sequential B-Panel Packing: Calls packB_panel_sequential_* instead of
 *    packB_panel_Arm_*, eliminating unintended ForkJoinPool task spawning inside Tier 2.
 * 2. Hoisted ThreadLocal Lookups: Fetches thread-local packed A and B segments once
 *    per matmul call rather than inside the inner ic loop.
 * 3. Safe Vector-Length Routing: Uses USE_AARCH64_8X12_KERNEL so 128-bit ARM NEON
 *    routes to the universal 6 x (2*VL) register-blocked micro-kernel.
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
        if (USE_AARCH64_8X12_KERNEL) {
            final MemorySegment pB = tlPackedB_Aarch_Float.get();
            final MemorySegment pA = tlPackedA_Aarch_Float.get();
            for (int jc = 0; jc < p; jc += NC_AARCH) {
                int nc = Math.min(NC_AARCH, p - jc);
                int fullPanels = nc / 12;
                int tailCols = nc % 12;
                for (int pc = 0; pc < m; pc += KC) {
                    int kc = Math.min(KC, m - pc);
                    boolean isFirstKBlock = (pc == 0);
                    MatMulPacking.packB_panels_range_Aarch_Float(B, pB, pc, jc, kc, b_s0, b_s1, 0, fullPanels);
                    if (tailCols > 0) {
                        MatMulPacking.packB_tail_Aarch_Float(B, pB, pc, jc, kc, nc, b_s0, b_s1, fullPanels, tailCols);
                    }
                    for (int ic = 0; ic < n; ic += MC) {
                        int mc = Math.min(MC, n - ic);
                        MatMulPacking.packA_panel_Aarch_Float(A, pA, ic, mc, pc, kc, a_s0, a_s1);
                        MatMulMicroKernels.gebpMacroKernel_Aarch_Float(pA, pB, C, ic, mc, jc, nc, kc, p, isFirstKBlock);
                    }
                }
            }
        } else {
            final MemorySegment pB = tlPackedB_Arm_Float.get();
            final MemorySegment pA = tlPackedA_Arm_Float.get();
            for (int jc = 0; jc < p; jc += NC_ARM) {
                int nc = Math.min(NC_ARM, p - jc);
                for (int pc = 0; pc < m; pc += KC) {
                    int kc = Math.min(KC, m - pc);
                    boolean isFirstKBlock = (pc == 0);
                    MatMulPacking.packB_panel_sequential_Float(B, pB, pc, jc, kc, nc, b_s0, b_s1);
                    for (int ic = 0; ic < n; ic += MC) {
                        int mc = Math.min(MC, n - ic);
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
        if (USE_AARCH64_8X12_KERNEL) {
            final MemorySegment pB = tlPackedB_Aarch_Double.get();
            final MemorySegment pA = tlPackedA_Aarch_Double.get();
            for (int jc = 0; jc < p; jc += NC_AARCH) {
                int nc = Math.min(NC_AARCH, p - jc);
                int fullPanels = nc / 12;
                int tailCols = nc % 12;
                for (int pc = 0; pc < m; pc += KC) {
                    int kc = Math.min(KC, m - pc);
                    boolean isFirstKBlock = (pc == 0);
                    MatMulPacking.packB_panels_range_Aarch_Double(B, pB, pc, jc, kc, b_s0, b_s1, 0, fullPanels);
                    if (tailCols > 0) {
                        MatMulPacking.packB_tail_Aarch_Double(B, pB, pc, jc, kc, nc, b_s0, b_s1, fullPanels, tailCols);
                    }
                    for (int ic = 0; ic < n; ic += MC) {
                        int mc = Math.min(MC, n - ic);
                        MatMulPacking.packA_panel_Aarch_Double(A, pA, ic, mc, pc, kc, a_s0, a_s1);
                        MatMulMicroKernels.gebpMacroKernel_Aarch_Double(pA, pB, C, ic, mc, jc, nc, kc, p, isFirstKBlock);
                    }
                }
            }
        } else {
            final MemorySegment pB = tlPackedB_Arm_Double.get();
            final MemorySegment pA = tlPackedA_Arm_Double.get();
            for (int jc = 0; jc < p; jc += NC_ARM) {
                int nc = Math.min(NC_ARM, p - jc);
                for (int pc = 0; pc < m; pc += KC) {
                    int kc = Math.min(KC, m - pc);
                    boolean isFirstKBlock = (pc == 0);
                    MatMulPacking.packB_panel_sequential_Double(B, pB, pc, jc, kc, nc, b_s0, b_s1);
                    for (int ic = 0; ic < n; ic += MC) {
                        int mc = Math.min(MC, n - ic);
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
        if (USE_AARCH64_8X12_KERNEL) {
            final MemorySegment pB = tlPackedB_Aarch_Int.get();
            final MemorySegment pA = tlPackedA_Aarch_Int.get();
            for (int jc = 0; jc < p; jc += NC_AARCH) {
                int nc = Math.min(NC_AARCH, p - jc);
                int fullPanels = nc / 12;
                int tailCols = nc % 12;
                for (int pc = 0; pc < m; pc += KC) {
                    int kc = Math.min(KC, m - pc);
                    boolean isFirstKBlock = (pc == 0);
                    MatMulPacking.packB_panels_range_Aarch_Int(B, pB, pc, jc, kc, b_s0, b_s1, 0, fullPanels);
                    if (tailCols > 0) {
                        MatMulPacking.packB_tail_Aarch_Int(B, pB, pc, jc, kc, nc, b_s0, b_s1, fullPanels, tailCols);
                    }
                    for (int ic = 0; ic < n; ic += MC) {
                        int mc = Math.min(MC, n - ic);
                        MatMulPacking.packA_panel_Aarch_Int(A, pA, ic, mc, pc, kc, a_s0, a_s1);
                        MatMulMicroKernels.gebpMacroKernel_Aarch_Int(pA, pB, C, ic, mc, jc, nc, kc, p, isFirstKBlock);
                    }
                }
            }
        } else {
            final MemorySegment pB = tlPackedB_Arm_Int.get();
            final MemorySegment pA = tlPackedA_Arm_Int.get();
            for (int jc = 0; jc < p; jc += NC_ARM) {
                int nc = Math.min(NC_ARM, p - jc);
                for (int pc = 0; pc < m; pc += KC) {
                    int kc = Math.min(KC, m - pc);
                    boolean isFirstKBlock = (pc == 0);
                    MatMulPacking.packB_panel_sequential_Int(B, pB, pc, jc, kc, nc, b_s0, b_s1);
                    for (int ic = 0; ic < n; ic += MC) {
                        int mc = Math.min(MC, n - ic);
                        MatMulPacking.packA_panel_Arm_Int(A, pA, ic, mc, pc, kc, a_s0, a_s1);
                        MatMulMicroKernels.gebpMacroKernel_Arm_Int(pA, pB, C, ic, mc, jc, nc, kc, p, isFirstKBlock);
                    }
                }
            }
        }
    }
}