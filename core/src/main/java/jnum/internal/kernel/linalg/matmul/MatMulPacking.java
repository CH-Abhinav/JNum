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
import java.nio.ByteOrder;
import java.util.concurrent.RecursiveAction;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorSpecies;

/**
 * BLIS Panel Packing Routines for Matrices A and B.
 *
 * Key Platform & JDK 25-27 Optimizations:
 * 1. Pure 64-bit Byte-Pointer Induction Variables: Eliminates src.getAtIndex() scaling
 *    and int-to-long casts inside hot loops so HotSpot C2 hoists ScopedMemoryAccess checks.
 * 2. 4-Way Unrolled Contiguous Packing: Streams A and B micro-panels using sequential
 *    cache-line writes and 8x SIMD vector loads/stores per unrolled iteration.
 * 3. VectorMask Tail Packing: Replaces scalar tail loops in packB_tail_* with masked SIMD
 *    loads that automatically zero-pad inactive lanes in registers.
 * 4. Transposed-Source Fast Path: Detects unit row stride (a_s0 == 1L) when packing A^T.
 */
public final class MatMulPacking {

    private static final ByteOrder NATIVE = NATIVE_ORDER;
    private static final VectorSpecies<Float> SPECIES = SPECIES_F32;
    private static final VectorSpecies<Double> SPECIESDB = SPECIES_F64;
    private static final VectorSpecies<Integer> SPECIESINT = SPECIES_I32;

    private MatMulPacking() {
        throw new AssertionError("Utility class");
    }

    private static int computePackBLeafPanels(int fullPanels) {
        if (AVAILABLE_CORES <= 1) return fullPanels;
        int targetTasks = Math.max(AVAILABLE_CORES, AVAILABLE_CORES * MATMUL_TASKS_PER_CORE);
        return Math.max(2, (fullPanels + targetTasks - 1) / targetTasks);
    }

    // =========================================================================
    // 1. FLOAT (FP32) PANEL PACKING ROUTINES
    // =========================================================================

    public static void packA_panel_Arm_Float(MemorySegment src, MemorySegment dst,
                                             int rowStart, int mc, int colStart, int kc,
                                             long a_s0, long a_s1) {
        int fullPanels = mc / MR;
        int tailRows = mc % MR;
        final long rowStrideBytes = a_s0 * 4L;
        final long panelBytes = (long) kc * STEP_A_ROW_BYTES_F32;

        if (a_s1 == 1L) {
            for (int p = 0; p < fullPanels; p++) {
                long dOff = (long) p * panelBytes;
                long r0 = ((long) (rowStart + p * MR) * a_s0 + colStart) * 4L;
                long r1 = r0 + rowStrideBytes;
                long r2 = r1 + rowStrideBytes;
                long r3 = r2 + rowStrideBytes;
                long r4 = r3 + rowStrideBytes;
                long r5 = r4 + rowStrideBytes;

                long kByteOff = 0L;
                int k = 0;
                for (; k <= kc - 4; k += 4) {
                    dst.set(VF, dOff,       src.get(VF, r0 + kByteOff));
                    dst.set(VF, dOff + 4L,  src.get(VF, r1 + kByteOff));
                    dst.set(VF, dOff + 8L,  src.get(VF, r2 + kByteOff));
                    dst.set(VF, dOff + 12L, src.get(VF, r3 + kByteOff));
                    dst.set(VF, dOff + 16L, src.get(VF, r4 + kByteOff));
                    dst.set(VF, dOff + 20L, src.get(VF, r5 + kByteOff));

                    long k1 = kByteOff + 4L;
                    long d1 = dOff + STEP_A_ROW_BYTES_F32;
                    dst.set(VF, d1,       src.get(VF, r0 + k1));
                    dst.set(VF, d1 + 4L,  src.get(VF, r1 + k1));
                    dst.set(VF, d1 + 8L,  src.get(VF, r2 + k1));
                    dst.set(VF, d1 + 12L, src.get(VF, r3 + k1));
                    dst.set(VF, d1 + 16L, src.get(VF, r4 + k1));
                    dst.set(VF, d1 + 20L, src.get(VF, r5 + k1));

                    long k2 = kByteOff + 8L;
                    long d2 = d1 + STEP_A_ROW_BYTES_F32;
                    dst.set(VF, d2,       src.get(VF, r0 + k2));
                    dst.set(VF, d2 + 4L,  src.get(VF, r1 + k2));
                    dst.set(VF, d2 + 8L,  src.get(VF, r2 + k2));
                    dst.set(VF, d2 + 12L, src.get(VF, r3 + k2));
                    dst.set(VF, d2 + 16L, src.get(VF, r4 + k2));
                    dst.set(VF, d2 + 20L, src.get(VF, r5 + k2));

                    long k3 = kByteOff + 12L;
                    long d3 = d2 + STEP_A_ROW_BYTES_F32;
                    dst.set(VF, d3,       src.get(VF, r0 + k3));
                    dst.set(VF, d3 + 4L,  src.get(VF, r1 + k3));
                    dst.set(VF, d3 + 8L,  src.get(VF, r2 + k3));
                    dst.set(VF, d3 + 12L, src.get(VF, r3 + k3));
                    dst.set(VF, d3 + 16L, src.get(VF, r4 + k3));
                    dst.set(VF, d3 + 20L, src.get(VF, r5 + k3));

                    kByteOff += 16L;
                    dOff += STEP_A_K4_BYTES_F32;
                }
                for (; k < kc; k++) {
                    dst.set(VF, dOff,       src.get(VF, r0 + kByteOff));
                    dst.set(VF, dOff + 4L,  src.get(VF, r1 + kByteOff));
                    dst.set(VF, dOff + 8L,  src.get(VF, r2 + kByteOff));
                    dst.set(VF, dOff + 12L, src.get(VF, r3 + kByteOff));
                    dst.set(VF, dOff + 16L, src.get(VF, r4 + kByteOff));
                    dst.set(VF, dOff + 20L, src.get(VF, r5 + kByteOff));
                    kByteOff += 4L;
                    dOff += STEP_A_ROW_BYTES_F32;
                }
            }
            if (tailRows > 0) {
                packA_tail_Arm_Float(src, dst, rowStart + fullPanels * MR, tailRows, colStart, kc, a_s0, 1L, (long) fullPanels * panelBytes);
            }
        } else {
            final long colStrideBytes = a_s1 * 4L;
            for (int p = 0; p < fullPanels; p++) {
                long dOff = (long) p * panelBytes;
                long r0 = ((long) (rowStart + p * MR) * a_s0 + (long) colStart * a_s1) * 4L;
                long r1 = r0 + rowStrideBytes;
                long r2 = r1 + rowStrideBytes;
                long r3 = r2 + rowStrideBytes;
                long r4 = r3 + rowStrideBytes;
                long r5 = r4 + rowStrideBytes;

                long kByteOff = 0L;
                for (int k = 0; k < kc; k++) {
                    dst.set(VF, dOff,       src.get(VF, r0 + kByteOff));
                    dst.set(VF, dOff + 4L,  src.get(VF, r1 + kByteOff));
                    dst.set(VF, dOff + 8L,  src.get(VF, r2 + kByteOff));
                    dst.set(VF, dOff + 12L, src.get(VF, r3 + kByteOff));
                    dst.set(VF, dOff + 16L, src.get(VF, r4 + kByteOff));
                    dst.set(VF, dOff + 20L, src.get(VF, r5 + kByteOff));
                    kByteOff += colStrideBytes;
                    dOff += STEP_A_ROW_BYTES_F32;
                }
            }
            if (tailRows > 0) {
                packA_tail_Arm_Float(src, dst, rowStart + fullPanels * MR, tailRows, colStart, kc, a_s0, a_s1, (long) fullPanels * panelBytes);
            }
        }
    }

    private static void packA_tail_Arm_Float(MemorySegment src, MemorySegment dst,
                                             int tailRowStart, int tailRows, int colStart, int kc,
                                             long a_s0, long a_s1, long dstBase) {
        final long colStrideBytes = a_s1 * 4L;
        final long rowStrideBytes = a_s0 * 4L;
        final long base0 = ((long) tailRowStart * a_s0 + (long) colStart * a_s1) * 4L;

        long dOff = dstBase;
        long kByteOff = 0L;
        for (int k = 0; k < kc; k++) {
            long sRow = base0 + kByteOff;
            int r = 0;
            for (; r < tailRows; r++) {
                dst.set(VF, dOff + (long) r * 4L, src.get(VF, sRow));
                sRow += rowStrideBytes;
            }
            for (; r < MR; r++) {
                dst.set(VF, dOff + (long) r * 4L, 0.0f);
            }
            kByteOff += colStrideBytes;
            dOff += STEP_A_ROW_BYTES_F32;
        }
    }

    public static void packB_panel_Arm_Float(MemorySegment src, MemorySegment dst,
                                             int rowStart, int colStart, int kc, int nc,
                                             long b_s0, long b_s1) {
        int fullPanels = nc / NR;
        int tailCols = nc % NR;
        if (fullPanels >= 4 && AVAILABLE_CORES >= 2) {
            int leafPanels = computePackBLeafPanels(fullPanels);
            POOL.invoke(new PackBTask_Arm_Float(src, dst, rowStart, colStart, kc, nc, b_s0, b_s1, 0, fullPanels, leafPanels));
        } else {
            packB_panels_range_Arm_Float(src, dst, rowStart, colStart, kc, b_s0, b_s1, 0, fullPanels);
        }
        if (tailCols > 0) {
            packB_tail_Arm_Float(src, dst, rowStart, colStart, kc, nc, b_s0, b_s1, fullPanels, tailCols);
        }
    }

    public static void packB_panels_range_Arm_Float(MemorySegment src, MemorySegment dst,
                                                    int rowStart, int colStart, int kc,
                                                    long b_s0, long b_s1, int pStart, int pEnd) {
        final long panelBytes = (long) kc * STEP_B_ROW_BYTES_F32;
        final long srcRowStrideBytes = b_s0 * 4L;
        final long srcRowStride2 = srcRowStrideBytes * 2L;
        final long srcRowStride3 = srcRowStrideBytes * 3L;
        final long srcRowStride4 = srcRowStrideBytes * 4L;

        if (b_s1 == 1L) {
            for (int p = pStart; p < pEnd; p++) {
                long srcOff = ((long) rowStart * b_s0 + colStart + (long) p * NR) * 4L;
                long dstOff = (long) p * panelBytes;
                int k = 0;
                for (; k <= kc - 4; k += 4) {
                    // k + 0
                    FloatVector.fromMemorySegment(SPECIES, src, srcOff, NATIVE).intoMemorySegment(dst, dstOff, NATIVE);
                    FloatVector.fromMemorySegment(SPECIES, src, srcOff + STRIDE_VEC_BYTES_F32, NATIVE).intoMemorySegment(dst, dstOff + STRIDE_VEC_BYTES_F32, NATIVE);

                    // k + 1
                    long s1 = srcOff + srcRowStrideBytes;
                    long d1 = dstOff + STEP_B_ROW_BYTES_F32;
                    FloatVector.fromMemorySegment(SPECIES, src, s1, NATIVE).intoMemorySegment(dst, d1, NATIVE);
                    FloatVector.fromMemorySegment(SPECIES, src, s1 + STRIDE_VEC_BYTES_F32, NATIVE).intoMemorySegment(dst, d1 + STRIDE_VEC_BYTES_F32, NATIVE);

                    // k + 2
                    long s2 = srcOff + srcRowStride2;
                    long d2 = d1 + STEP_B_ROW_BYTES_F32;
                    FloatVector.fromMemorySegment(SPECIES, src, s2, NATIVE).intoMemorySegment(dst, d2, NATIVE);
                    FloatVector.fromMemorySegment(SPECIES, src, s2 + STRIDE_VEC_BYTES_F32, NATIVE).intoMemorySegment(dst, d2 + STRIDE_VEC_BYTES_F32, NATIVE);

                    // k + 3
                    long s3 = srcOff + srcRowStride3;
                    long d3 = d2 + STEP_B_ROW_BYTES_F32;
                    FloatVector.fromMemorySegment(SPECIES, src, s3, NATIVE).intoMemorySegment(dst, d3, NATIVE);
                    FloatVector.fromMemorySegment(SPECIES, src, s3 + STRIDE_VEC_BYTES_F32, NATIVE).intoMemorySegment(dst, d3 + STRIDE_VEC_BYTES_F32, NATIVE);

                    srcOff += srcRowStride4;
                    dstOff += STEP_B_K4_BYTES_F32;
                }
                for (; k < kc; k++) {
                    FloatVector.fromMemorySegment(SPECIES, src, srcOff, NATIVE).intoMemorySegment(dst, dstOff, NATIVE);
                    FloatVector.fromMemorySegment(SPECIES, src, srcOff + STRIDE_VEC_BYTES_F32, NATIVE).intoMemorySegment(dst, dstOff + STRIDE_VEC_BYTES_F32, NATIVE);
                    srcOff += srcRowStrideBytes;
                    dstOff += STEP_B_ROW_BYTES_F32;
                }
            }
        } else {
            final long colStrideBytes = b_s1 * 4L;
            for (int p = pStart; p < pEnd; p++) {
                long dstOff = (long) p * panelBytes;
                long rowBase = ((long) rowStart * b_s0 + (long) (colStart + p * NR) * b_s1) * 4L;
                for (int k = 0; k < kc; k++) {
                    long sAddr = rowBase;
                    for (int c = 0; c < NR; c++) {
                        dst.set(VF, dstOff + (long) c * 4L, src.get(VF, sAddr));
                        sAddr += colStrideBytes;
                    }
                    rowBase += srcRowStrideBytes;
                    dstOff += STEP_B_ROW_BYTES_F32;
                }
            }
        }
    }

    public static void packB_tail_Arm_Float(MemorySegment src, MemorySegment dst,
                                            int rowStart, int colStart, int kc, int nc,
                                            long b_s0, long b_s1, int fullPanels, int tailCols) {
        long dstOff = (long) fullPanels * STEP_B_ROW_BYTES_F32 * kc;
        final long srcRowStrideBytes = b_s0 * 4L;

        if (b_s1 == 1L) {
            final int lanes0 = Math.min(VL_F32, tailCols);
            final int lanes1 = Math.max(0, tailCols - VL_F32);
            final VectorMask<Float> mask0 = SPECIES.indexInRange(0, lanes0);
            final VectorMask<Float> mask1 = SPECIES.indexInRange(0, lanes1);
            final FloatVector zero = FloatVector.zero(SPECIES);

            long srcOff = ((long) rowStart * b_s0 + colStart + (long) fullPanels * NR) * 4L;
            for (int k = 0; k < kc; k++) {
                FloatVector v0 = FloatVector.fromMemorySegment(SPECIES, src, srcOff, NATIVE, mask0);
                v0.intoMemorySegment(dst, dstOff, NATIVE);
                FloatVector v1 = (lanes1 > 0)
                        ? FloatVector.fromMemorySegment(SPECIES, src, srcOff + STRIDE_VEC_BYTES_F32, NATIVE, mask1)
                        : zero;
                v1.intoMemorySegment(dst, dstOff + STRIDE_VEC_BYTES_F32, NATIVE);
                srcOff += srcRowStrideBytes;
                dstOff += STEP_B_ROW_BYTES_F32;
            }
        } else {
            final long colStrideBytes = b_s1 * 4L;
            long rowBase = ((long) rowStart * b_s0 + (long) (colStart + fullPanels * NR) * b_s1) * 4L;
            for (int k = 0; k < kc; k++) {
                long sAddr = rowBase;
                int c = 0;
                for (; c < tailCols; c++) {
                    dst.set(VF, dstOff + (long) c * 4L, src.get(VF, sAddr));
                    sAddr += colStrideBytes;
                }
                for (; c < NR; c++) {
                    dst.set(VF, dstOff + (long) c * 4L, 0.0f);
                }
                rowBase += srcRowStrideBytes;
                dstOff += STEP_B_ROW_BYTES_F32;
            }
        }
    }

    public static final class PackBTask_Arm_Float extends RecursiveAction {
        final MemorySegment src, dst;
        final int rowStart, colStart, kc, nc;
        final long b_s0, b_s1;
        final int pStart, pEnd, leafPanels;

        PackBTask_Arm_Float(MemorySegment src, MemorySegment dst, int rowStart, int colStart,
                            int kc, int nc, long b_s0, long b_s1, int pStart, int pEnd) {
            this(src, dst, rowStart, colStart, kc, nc, b_s0, b_s1, pStart, pEnd, computePackBLeafPanels(pEnd - pStart));
        }

        PackBTask_Arm_Float(MemorySegment src, MemorySegment dst, int rowStart, int colStart,
                            int kc, int nc, long b_s0, long b_s1, int pStart, int pEnd, int leafPanels) {
            this.src = src; this.dst = dst; this.rowStart = rowStart; this.colStart = colStart;
            this.kc = kc; this.nc = nc; this.b_s0 = b_s0; this.b_s1 = b_s1;
            this.pStart = pStart; this.pEnd = pEnd; this.leafPanels = leafPanels;
        }

        @Override
        protected void compute() {
            int panels = pEnd - pStart;
            if (panels <= leafPanels) {
                packB_panels_range_Arm_Float(src, dst, rowStart, colStart, kc, b_s0, b_s1, pStart, pEnd);
            } else {
                int mid = pStart + (panels >>> 1);
                invokeAll(
                        new PackBTask_Arm_Float(src, dst, rowStart, colStart, kc, nc, b_s0, b_s1, pStart, mid, leafPanels),
                        new PackBTask_Arm_Float(src, dst, rowStart, colStart, kc, nc, b_s0, b_s1, mid, pEnd, leafPanels)
                );
            }
        }
    }

    public static void packA_panel_Aarch_Float(MemorySegment src, MemorySegment dst,
                                               int rowStart, int mc, int colStart, int kc,
                                               long a_s0, long a_s1) {
        int fullPanels = mc / 8;
        int tailRows = mc % 8;
        final long rowStrideBytes = a_s0 * 4L;
        final long colStrideBytes = a_s1 * 4L;
        final long panelBytes = (long) kc * 32L;

        for (int p = 0; p < fullPanels; p++) {
            long dOff = (long) p * panelBytes;
            long r0 = ((long) (rowStart + p * 8) * a_s0 + (long) colStart * a_s1) * 4L;
            long r1 = r0 + rowStrideBytes;
            long r2 = r1 + rowStrideBytes;
            long r3 = r2 + rowStrideBytes;
            long r4 = r3 + rowStrideBytes;
            long r5 = r4 + rowStrideBytes;
            long r6 = r5 + rowStrideBytes;
            long r7 = r6 + rowStrideBytes;

            long kByteOff = 0L;
            for (int k = 0; k < kc; k++) {
                dst.set(VF, dOff,       src.get(VF, r0 + kByteOff));
                dst.set(VF, dOff + 4L,  src.get(VF, r1 + kByteOff));
                dst.set(VF, dOff + 8L,  src.get(VF, r2 + kByteOff));
                dst.set(VF, dOff + 12L, src.get(VF, r3 + kByteOff));
                dst.set(VF, dOff + 16L, src.get(VF, r4 + kByteOff));
                dst.set(VF, dOff + 20L, src.get(VF, r5 + kByteOff));
                dst.set(VF, dOff + 24L, src.get(VF, r6 + kByteOff));
                dst.set(VF, dOff + 28L, src.get(VF, r7 + kByteOff));
                kByteOff += colStrideBytes;
                dOff += 32L;
            }
        }
        if (tailRows > 0) {
            long dOff = (long) fullPanels * panelBytes;
            long base0 = ((long) (rowStart + fullPanels * 8) * a_s0 + (long) colStart * a_s1) * 4L;
            long kByteOff = 0L;
            for (int k = 0; k < kc; k++) {
                long sRow = base0 + kByteOff;
                int r = 0;
                for (; r < tailRows; r++) {
                    dst.set(VF, dOff + (long) r * 4L, src.get(VF, sRow));
                    sRow += rowStrideBytes;
                }
                for (; r < 8; r++) {
                    dst.set(VF, dOff + (long) r * 4L, 0.0f);
                }
                kByteOff += colStrideBytes;
                dOff += 32L;
            }
        }
    }

    public static void packB_panel_Aarch_Float(MemorySegment src, MemorySegment dst,
                                               int rowStart, int colStart, int kc, int nc,
                                               long b_s0, long b_s1) {
        int fullPanels = nc / 12;
        int tailCols = nc % 12;
        if (fullPanels >= 4 && AVAILABLE_CORES >= 2) {
            int leafPanels = computePackBLeafPanels(fullPanels);
            POOL.invoke(new PackBTask_Aarch_Float(src, dst, rowStart, colStart, kc, nc, b_s0, b_s1, 0, fullPanels, leafPanels));
        } else {
            packB_panels_range_Aarch_Float(src, dst, rowStart, colStart, kc, b_s0, b_s1, 0, fullPanels);
        }
        if (tailCols > 0) {
            packB_tail_Aarch_Float(src, dst, rowStart, colStart, kc, nc, b_s0, b_s1, fullPanels, tailCols);
        }
    }

    public static void packB_panels_range_Aarch_Float(MemorySegment src, MemorySegment dst,
                                                      int rowStart, int colStart, int kc,
                                                      long b_s0, long b_s1, int pStart, int pEnd) {
        final long srcRowStrideBytes = b_s0 * 4L;
        if (b_s1 == 1L) {
            for (int p = pStart; p < pEnd; p++) {
                long srcOff = ((long) rowStart * b_s0 + colStart + (long) p * 12) * 4L;
                long dstOff = (long) p * 16 * kc * 4L;
                for (int k = 0; k < kc; k++) {
                    MemorySegment.copy(src, srcOff, dst, dstOff, 48L);
                    dst.set(VF, dstOff + 48L, 0.0f);
                    dst.set(VF, dstOff + 52L, 0.0f);
                    dst.set(VF, dstOff + 56L, 0.0f);
                    dst.set(VF, dstOff + 60L, 0.0f);
                    srcOff += srcRowStrideBytes;
                    dstOff += 64L;
                }
            }
        } else {
            final long colStrideBytes = b_s1 * 4L;
            for (int p = pStart; p < pEnd; p++) {
                long dstOff = (long) p * 16 * kc * 4L;
                long rowBase = ((long) rowStart * b_s0 + (long) (colStart + p * 12) * b_s1) * 4L;
                for (int k = 0; k < kc; k++) {
                    long sAddr = rowBase;
                    for (int c = 0; c < 12; c++) {
                        dst.set(VF, dstOff + (long) c * 4L, src.get(VF, sAddr));
                        sAddr += colStrideBytes;
                    }
                    dst.set(VF, dstOff + 48L, 0.0f);
                    dst.set(VF, dstOff + 52L, 0.0f);
                    dst.set(VF, dstOff + 56L, 0.0f);
                    dst.set(VF, dstOff + 60L, 0.0f);
                    rowBase += srcRowStrideBytes;
                    dstOff += 64L;
                }
            }
        }
    }

    public static void packB_tail_Aarch_Float(MemorySegment src, MemorySegment dst,
                                              int rowStart, int colStart, int kc, int nc,
                                              long b_s0, long b_s1, int fullPanels, int tailCols) {
        long dstOff = (long) fullPanels * 16 * kc * 4L;
        final long srcRowStrideBytes = b_s0 * 4L;
        if (b_s1 == 1L) {
            long srcOff = ((long) rowStart * b_s0 + colStart + (long) fullPanels * 12) * 4L;
            long copyBytes = (long) tailCols * 4L;
            for (int k = 0; k < kc; k++) {
                MemorySegment.copy(src, srcOff, dst, dstOff, copyBytes);
                for (int c = tailCols; c < 16; c++) {
                    dst.set(VF, dstOff + (long) c * 4L, 0.0f);
                }
                srcOff += srcRowStrideBytes;
                dstOff += 64L;
            }
        } else {
            final long colStrideBytes = b_s1 * 4L;
            long rowBase = ((long) rowStart * b_s0 + (long) (colStart + fullPanels * 12) * b_s1) * 4L;
            for (int k = 0; k < kc; k++) {
                long sAddr = rowBase;
                int c = 0;
                for (; c < tailCols; c++) {
                    dst.set(VF, dstOff + (long) c * 4L, src.get(VF, sAddr));
                    sAddr += colStrideBytes;
                }
                for (; c < 16; c++) {
                    dst.set(VF, dstOff + (long) c * 4L, 0.0f);
                }
                rowBase += srcRowStrideBytes;
                dstOff += 64L;
            }
        }
    }

    public static final class PackBTask_Aarch_Float extends RecursiveAction {
        final MemorySegment src, dst;
        final int rowStart, colStart, kc, nc;
        final long b_s0, b_s1;
        final int pStart, pEnd, leafPanels;

        PackBTask_Aarch_Float(MemorySegment src, MemorySegment dst, int rowStart, int colStart,
                              int kc, int nc, long b_s0, long b_s1, int pStart, int pEnd) {
            this(src, dst, rowStart, colStart, kc, nc, b_s0, b_s1, pStart, pEnd, computePackBLeafPanels(pEnd - pStart));
        }

        PackBTask_Aarch_Float(MemorySegment src, MemorySegment dst, int rowStart, int colStart,
                              int kc, int nc, long b_s0, long b_s1, int pStart, int pEnd, int leafPanels) {
            this.src = src; this.dst = dst; this.rowStart = rowStart; this.colStart = colStart;
            this.kc = kc; this.nc = nc; this.b_s0 = b_s0; this.b_s1 = b_s1;
            this.pStart = pStart; this.pEnd = pEnd; this.leafPanels = leafPanels;
        }

        @Override
        protected void compute() {
            int panels = pEnd - pStart;
            if (panels <= leafPanels) {
                packB_panels_range_Aarch_Float(src, dst, rowStart, colStart, kc, b_s0, b_s1, pStart, pEnd);
            } else {
                int mid = pStart + (panels >>> 1);
                invokeAll(
                        new PackBTask_Aarch_Float(src, dst, rowStart, colStart, kc, nc, b_s0, b_s1, pStart, mid, leafPanels),
                        new PackBTask_Aarch_Float(src, dst, rowStart, colStart, kc, nc, b_s0, b_s1, mid, pEnd, leafPanels)
                );
            }
        }
    }

    // =========================================================================
    // 2. DOUBLE (FP64) PANEL PACKING ROUTINES
    // =========================================================================

    public static void packA_panel_Arm_Double(MemorySegment src, MemorySegment dst,
                                              int rowStart, int mc, int colStart, int kc,
                                              long a_s0, long a_s1) {
        int fullPanels = mc / MR;
        int tailRows = mc % MR;
        final long rowStrideBytes = a_s0 * 8L;
        final long panelBytes = (long) kc * STEP_A_ROW_BYTES_F64;

        if (a_s1 == 1L) {
            for (int p = 0; p < fullPanels; p++) {
                long dOff = (long) p * panelBytes;
                long r0 = ((long) (rowStart + p * MR) * a_s0 + colStart) * 8L;
                long r1 = r0 + rowStrideBytes;
                long r2 = r1 + rowStrideBytes;
                long r3 = r2 + rowStrideBytes;
                long r4 = r3 + rowStrideBytes;
                long r5 = r4 + rowStrideBytes;

                long kByteOff = 0L;
                int k = 0;
                for (; k <= kc - 4; k += 4) {
                    dst.set(VD, dOff,       src.get(VD, r0 + kByteOff));
                    dst.set(VD, dOff + 8L,  src.get(VD, r1 + kByteOff));
                    dst.set(VD, dOff + 16L, src.get(VD, r2 + kByteOff));
                    dst.set(VD, dOff + 24L, src.get(VD, r3 + kByteOff));
                    dst.set(VD, dOff + 32L, src.get(VD, r4 + kByteOff));
                    dst.set(VD, dOff + 40L, src.get(VD, r5 + kByteOff));

                    long k1 = kByteOff + 8L;
                    long d1 = dOff + STEP_A_ROW_BYTES_F64;
                    dst.set(VD, d1,       src.get(VD, r0 + k1));
                    dst.set(VD, d1 + 8L,  src.get(VD, r1 + k1));
                    dst.set(VD, d1 + 16L, src.get(VD, r2 + k1));
                    dst.set(VD, d1 + 24L, src.get(VD, r3 + k1));
                    dst.set(VD, d1 + 32L, src.get(VD, r4 + k1));
                    dst.set(VD, d1 + 40L, src.get(VD, r5 + k1));

                    long k2 = kByteOff + 16L;
                    long d2 = d1 + STEP_A_ROW_BYTES_F64;
                    dst.set(VD, d2,       src.get(VD, r0 + k2));
                    dst.set(VD, d2 + 8L,  src.get(VD, r1 + k2));
                    dst.set(VD, d2 + 16L, src.get(VD, r2 + k2));
                    dst.set(VD, d2 + 24L, src.get(VD, r3 + k2));
                    dst.set(VD, d2 + 32L, src.get(VD, r4 + k2));
                    dst.set(VD, d2 + 40L, src.get(VD, r5 + k2));

                    long k3 = kByteOff + 24L;
                    long d3 = d2 + STEP_A_ROW_BYTES_F64;
                    dst.set(VD, d3,       src.get(VD, r0 + k3));
                    dst.set(VD, d3 + 8L,  src.get(VD, r1 + k3));
                    dst.set(VD, d3 + 16L, src.get(VD, r2 + k3));
                    dst.set(VD, d3 + 24L, src.get(VD, r3 + k3));
                    dst.set(VD, d3 + 32L, src.get(VD, r4 + k3));
                    dst.set(VD, d3 + 40L, src.get(VD, r5 + k3));

                    kByteOff += 32L;
                    dOff += STEP_A_K4_BYTES_F64;
                }
                for (; k < kc; k++) {
                    dst.set(VD, dOff,       src.get(VD, r0 + kByteOff));
                    dst.set(VD, dOff + 8L,  src.get(VD, r1 + kByteOff));
                    dst.set(VD, dOff + 16L, src.get(VD, r2 + kByteOff));
                    dst.set(VD, dOff + 24L, src.get(VD, r3 + kByteOff));
                    dst.set(VD, dOff + 32L, src.get(VD, r4 + kByteOff));
                    dst.set(VD, dOff + 40L, src.get(VD, r5 + kByteOff));
                    kByteOff += 8L;
                    dOff += STEP_A_ROW_BYTES_F64;
                }
            }
            if (tailRows > 0) {
                packA_tail_Arm_Double(src, dst, rowStart + fullPanels * MR, tailRows, colStart, kc, a_s0, 1L, (long) fullPanels * panelBytes);
            }
        } else {
            final long colStrideBytes = a_s1 * 8L;
            for (int p = 0; p < fullPanels; p++) {
                long dOff = (long) p * panelBytes;
                long r0 = ((long) (rowStart + p * MR) * a_s0 + (long) colStart * a_s1) * 8L;
                long r1 = r0 + rowStrideBytes;
                long r2 = r1 + rowStrideBytes;
                long r3 = r2 + rowStrideBytes;
                long r4 = r3 + rowStrideBytes;
                long r5 = r4 + rowStrideBytes;

                long kByteOff = 0L;
                for (int k = 0; k < kc; k++) {
                    dst.set(VD, dOff,       src.get(VD, r0 + kByteOff));
                    dst.set(VD, dOff + 8L,  src.get(VD, r1 + kByteOff));
                    dst.set(VD, dOff + 16L, src.get(VD, r2 + kByteOff));
                    dst.set(VD, dOff + 24L, src.get(VD, r3 + kByteOff));
                    dst.set(VD, dOff + 32L, src.get(VD, r4 + kByteOff));
                    dst.set(VD, dOff + 40L, src.get(VD, r5 + kByteOff));
                    kByteOff += colStrideBytes;
                    dOff += STEP_A_ROW_BYTES_F64;
                }
            }
            if (tailRows > 0) {
                packA_tail_Arm_Double(src, dst, rowStart + fullPanels * MR, tailRows, colStart, kc, a_s0, a_s1, (long) fullPanels * panelBytes);
            }
        }
    }

    private static void packA_tail_Arm_Double(MemorySegment src, MemorySegment dst,
                                              int tailRowStart, int tailRows, int colStart, int kc,
                                              long a_s0, long a_s1, long dstBase) {
        final long colStrideBytes = a_s1 * 8L;
        final long rowStrideBytes = a_s0 * 8L;
        final long base0 = ((long) tailRowStart * a_s0 + (long) colStart * a_s1) * 8L;

        long dOff = dstBase;
        long kByteOff = 0L;
        for (int k = 0; k < kc; k++) {
            long sRow = base0 + kByteOff;
            int r = 0;
            for (; r < tailRows; r++) {
                dst.set(VD, dOff + (long) r * 8L, src.get(VD, sRow));
                sRow += rowStrideBytes;
            }
            for (; r < MR; r++) {
                dst.set(VD, dOff + (long) r * 8L, 0.0d);
            }
            kByteOff += colStrideBytes;
            dOff += STEP_A_ROW_BYTES_F64;
        }
    }

    public static void packB_panel_Arm_Double(MemorySegment src, MemorySegment dst,
                                              int rowStart, int colStart, int kc, int nc,
                                              long b_s0, long b_s1) {
        int fullPanels = nc / NR_DB;
        int tailCols = nc % NR_DB;
        if (fullPanels >= 4 && AVAILABLE_CORES >= 2) {
            int leafPanels = computePackBLeafPanels(fullPanels);
            POOL.invoke(new PackBTask_Arm_Double(src, dst, rowStart, colStart, kc, nc, b_s0, b_s1, 0, fullPanels, leafPanels));
        } else {
            packB_panels_range_Arm_Double(src, dst, rowStart, colStart, kc, b_s0, b_s1, 0, fullPanels);
        }
        if (tailCols > 0) {
            packB_tail_Arm_Double(src, dst, rowStart, colStart, kc, nc, b_s0, b_s1, fullPanels, tailCols);
        }
    }

    public static void packB_panels_range_Arm_Double(MemorySegment src, MemorySegment dst,
                                                     int rowStart, int colStart, int kc,
                                                     long b_s0, long b_s1, int pStart, int pEnd) {
        final long panelBytes = (long) kc * STEP_B_ROW_BYTES_F64;
        final long srcRowStrideBytes = b_s0 * 8L;
        final long srcRowStride2 = srcRowStrideBytes * 2L;
        final long srcRowStride3 = srcRowStrideBytes * 3L;
        final long srcRowStride4 = srcRowStrideBytes * 4L;

        if (b_s1 == 1L) {
            for (int p = pStart; p < pEnd; p++) {
                long srcOff = ((long) rowStart * b_s0 + colStart + (long) p * NR_DB) * 8L;
                long dstOff = (long) p * panelBytes;
                int k = 0;
                for (; k <= kc - 4; k += 4) {
                    DoubleVector.fromMemorySegment(SPECIESDB, src, srcOff, NATIVE).intoMemorySegment(dst, dstOff, NATIVE);
                    DoubleVector.fromMemorySegment(SPECIESDB, src, srcOff + STRIDE_VEC_BYTES_F64, NATIVE).intoMemorySegment(dst, dstOff + STRIDE_VEC_BYTES_F64, NATIVE);

                    long s1 = srcOff + srcRowStrideBytes;
                    long d1 = dstOff + STEP_B_ROW_BYTES_F64;
                    DoubleVector.fromMemorySegment(SPECIESDB, src, s1, NATIVE).intoMemorySegment(dst, d1, NATIVE);
                    DoubleVector.fromMemorySegment(SPECIESDB, src, s1 + STRIDE_VEC_BYTES_F64, NATIVE).intoMemorySegment(dst, d1 + STRIDE_VEC_BYTES_F64, NATIVE);

                    long s2 = srcOff + srcRowStride2;
                    long d2 = d1 + STEP_B_ROW_BYTES_F64;
                    DoubleVector.fromMemorySegment(SPECIESDB, src, s2, NATIVE).intoMemorySegment(dst, d2, NATIVE);
                    DoubleVector.fromMemorySegment(SPECIESDB, src, s2 + STRIDE_VEC_BYTES_F64, NATIVE).intoMemorySegment(dst, d2 + STRIDE_VEC_BYTES_F64, NATIVE);

                    long s3 = srcOff + srcRowStride3;
                    long d3 = d2 + STEP_B_ROW_BYTES_F64;
                    DoubleVector.fromMemorySegment(SPECIESDB, src, s3, NATIVE).intoMemorySegment(dst, d3, NATIVE);
                    DoubleVector.fromMemorySegment(SPECIESDB, src, s3 + STRIDE_VEC_BYTES_F64, NATIVE).intoMemorySegment(dst, d3 + STRIDE_VEC_BYTES_F64, NATIVE);

                    srcOff += srcRowStride4;
                    dstOff += STEP_B_K4_BYTES_F64;
                }
                for (; k < kc; k++) {
                    DoubleVector.fromMemorySegment(SPECIESDB, src, srcOff, NATIVE).intoMemorySegment(dst, dstOff, NATIVE);
                    DoubleVector.fromMemorySegment(SPECIESDB, src, srcOff + STRIDE_VEC_BYTES_F64, NATIVE).intoMemorySegment(dst, dstOff + STRIDE_VEC_BYTES_F64, NATIVE);
                    srcOff += srcRowStrideBytes;
                    dstOff += STEP_B_ROW_BYTES_F64;
                }
            }
        } else {
            final long colStrideBytes = b_s1 * 8L;
            for (int p = pStart; p < pEnd; p++) {
                long dstOff = (long) p * panelBytes;
                long rowBase = ((long) rowStart * b_s0 + (long) (colStart + p * NR_DB) * b_s1) * 8L;
                for (int k = 0; k < kc; k++) {
                    long sAddr = rowBase;
                    for (int c = 0; c < NR_DB; c++) {
                        dst.set(VD, dstOff + (long) c * 8L, src.get(VD, sAddr));
                        sAddr += colStrideBytes;
                    }
                    rowBase += srcRowStrideBytes;
                    dstOff += STEP_B_ROW_BYTES_F64;
                }
            }
        }
    }

    public static void packB_tail_Arm_Double(MemorySegment src, MemorySegment dst,
                                             int rowStart, int colStart, int kc, int nc,
                                             long b_s0, long b_s1, int fullPanels, int tailCols) {
        long dstOff = (long) fullPanels * STEP_B_ROW_BYTES_F64 * kc;
        final long srcRowStrideBytes = b_s0 * 8L;

        if (b_s1 == 1L) {
            final int lanes0 = Math.min(VL_F64, tailCols);
            final int lanes1 = Math.max(0, tailCols - VL_F64);
            final VectorMask<Double> mask0 = SPECIESDB.indexInRange(0, lanes0);
            final VectorMask<Double> mask1 = SPECIESDB.indexInRange(0, lanes1);
            final DoubleVector zero = DoubleVector.zero(SPECIESDB);

            long srcOff = ((long) rowStart * b_s0 + colStart + (long) fullPanels * NR_DB) * 8L;
            for (int k = 0; k < kc; k++) {
                DoubleVector v0 = DoubleVector.fromMemorySegment(SPECIESDB, src, srcOff, NATIVE, mask0);
                v0.intoMemorySegment(dst, dstOff, NATIVE);
                DoubleVector v1 = (lanes1 > 0)
                        ? DoubleVector.fromMemorySegment(SPECIESDB, src, srcOff + STRIDE_VEC_BYTES_F64, NATIVE, mask1)
                        : zero;
                v1.intoMemorySegment(dst, dstOff + STRIDE_VEC_BYTES_F64, NATIVE);
                srcOff += srcRowStrideBytes;
                dstOff += STEP_B_ROW_BYTES_F64;
            }
        } else {
            final long colStrideBytes = b_s1 * 8L;
            long rowBase = ((long) rowStart * b_s0 + (long) (colStart + fullPanels * NR_DB) * b_s1) * 8L;
            for (int k = 0; k < kc; k++) {
                long sAddr = rowBase;
                int c = 0;
                for (; c < tailCols; c++) {
                    dst.set(VD, dstOff + (long) c * 8L, src.get(VD, sAddr));
                    sAddr += colStrideBytes;
                }
                for (; c < NR_DB; c++) {
                    dst.set(VD, dstOff + (long) c * 8L, 0.0d);
                }
                rowBase += srcRowStrideBytes;
                dstOff += STEP_B_ROW_BYTES_F64;
            }
        }
    }

    public static final class PackBTask_Arm_Double extends RecursiveAction {
        final MemorySegment src, dst;
        final int rowStart, colStart, kc, nc;
        final long b_s0, b_s1;
        final int pStart, pEnd, leafPanels;

        PackBTask_Arm_Double(MemorySegment src, MemorySegment dst, int rowStart, int colStart,
                             int kc, int nc, long b_s0, long b_s1, int pStart, int pEnd) {
            this(src, dst, rowStart, colStart, kc, nc, b_s0, b_s1, pStart, pEnd, computePackBLeafPanels(pEnd - pStart));
        }

        PackBTask_Arm_Double(MemorySegment src, MemorySegment dst, int rowStart, int colStart,
                             int kc, int nc, long b_s0, long b_s1, int pStart, int pEnd, int leafPanels) {
            this.src = src; this.dst = dst; this.rowStart = rowStart; this.colStart = colStart;
            this.kc = kc; this.nc = nc; this.b_s0 = b_s0; this.b_s1 = b_s1;
            this.pStart = pStart; this.pEnd = pEnd; this.leafPanels = leafPanels;
        }

        @Override
        protected void compute() {
            int panels = pEnd - pStart;
            if (panels <= leafPanels) {
                packB_panels_range_Arm_Double(src, dst, rowStart, colStart, kc, b_s0, b_s1, pStart, pEnd);
            } else {
                int mid = pStart + (panels >>> 1);
                invokeAll(
                        new PackBTask_Arm_Double(src, dst, rowStart, colStart, kc, nc, b_s0, b_s1, pStart, mid, leafPanels),
                        new PackBTask_Arm_Double(src, dst, rowStart, colStart, kc, nc, b_s0, b_s1, mid, pEnd, leafPanels)
                );
            }
        }
    }

    public static void packA_panel_Aarch_Double(MemorySegment src, MemorySegment dst,
                                                int rowStart, int mc, int colStart, int kc,
                                                long a_s0, long a_s1) {
        int fullPanels = mc / 8;
        int tailRows = mc % 8;
        final long rowStrideBytes = a_s0 * 8L;
        final long colStrideBytes = a_s1 * 8L;
        final long panelBytes = (long) kc * 64L;

        for (int p = 0; p < fullPanels; p++) {
            long dOff = (long) p * panelBytes;
            long r0 = ((long) (rowStart + p * 8) * a_s0 + (long) colStart * a_s1) * 8L;
            long r1 = r0 + rowStrideBytes;
            long r2 = r1 + rowStrideBytes;
            long r3 = r2 + rowStrideBytes;
            long r4 = r3 + rowStrideBytes;
            long r5 = r4 + rowStrideBytes;
            long r6 = r5 + rowStrideBytes;
            long r7 = r6 + rowStrideBytes;

            long kByteOff = 0L;
            for (int k = 0; k < kc; k++) {
                dst.set(VD, dOff,       src.get(VD, r0 + kByteOff));
                dst.set(VD, dOff + 8L,  src.get(VD, r1 + kByteOff));
                dst.set(VD, dOff + 16L, src.get(VD, r2 + kByteOff));
                dst.set(VD, dOff + 24L, src.get(VD, r3 + kByteOff));
                dst.set(VD, dOff + 32L, src.get(VD, r4 + kByteOff));
                dst.set(VD, dOff + 40L, src.get(VD, r5 + kByteOff));
                dst.set(VD, dOff + 48L, src.get(VD, r6 + kByteOff));
                dst.set(VD, dOff + 56L, src.get(VD, r7 + kByteOff));
                kByteOff += colStrideBytes;
                dOff += 64L;
            }
        }
        if (tailRows > 0) {
            long dOff = (long) fullPanels * panelBytes;
            long base0 = ((long) (rowStart + fullPanels * 8) * a_s0 + (long) colStart * a_s1) * 8L;
            long kByteOff = 0L;
            for (int k = 0; k < kc; k++) {
                long sRow = base0 + kByteOff;
                int r = 0;
                for (; r < tailRows; r++) {
                    dst.set(VD, dOff + (long) r * 8L, src.get(VD, sRow));
                    sRow += rowStrideBytes;
                }
                for (; r < 8; r++) {
                    dst.set(VD, dOff + (long) r * 8L, 0.0d);
                }
                kByteOff += colStrideBytes;
                dOff += 64L;
            }
        }
    }

    public static void packB_panel_Aarch_Double(MemorySegment src, MemorySegment dst,
                                                int rowStart, int colStart, int kc, int nc,
                                                long b_s0, long b_s1) {
        int fullPanels = nc / 12;
        int tailCols = nc % 12;
        if (fullPanels >= 4 && AVAILABLE_CORES >= 2) {
            int leafPanels = computePackBLeafPanels(fullPanels);
            POOL.invoke(new PackBTask_Aarch_Double(src, dst, rowStart, colStart, kc, nc, b_s0, b_s1, 0, fullPanels, leafPanels));
        } else {
            packB_panels_range_Aarch_Double(src, dst, rowStart, colStart, kc, b_s0, b_s1, 0, fullPanels);
        }
        if (tailCols > 0) {
            packB_tail_Aarch_Double(src, dst, rowStart, colStart, kc, nc, b_s0, b_s1, fullPanels, tailCols);
        }
    }

    public static void packB_panels_range_Aarch_Double(MemorySegment src, MemorySegment dst,
                                                       int rowStart, int colStart, int kc,
                                                       long b_s0, long b_s1, int pStart, int pEnd) {
        final long srcRowStrideBytes = b_s0 * 8L;
        if (b_s1 == 1L) {
            for (int p = pStart; p < pEnd; p++) {
                long srcOff = ((long) rowStart * b_s0 + colStart + (long) p * 12) * 8L;
                long dstOff = (long) p * 16 * kc * 8L;
                for (int k = 0; k < kc; k++) {
                    MemorySegment.copy(src, srcOff, dst, dstOff, 96L);
                    dst.set(VD, dstOff + 96L,  0.0d);
                    dst.set(VD, dstOff + 104L, 0.0d);
                    dst.set(VD, dstOff + 112L, 0.0d);
                    dst.set(VD, dstOff + 120L, 0.0d);
                    srcOff += srcRowStrideBytes;
                    dstOff += 128L;
                }
            }
        } else {
            final long colStrideBytes = b_s1 * 8L;
            for (int p = pStart; p < pEnd; p++) {
                long dstOff = (long) p * 16 * kc * 8L;
                long rowBase = ((long) rowStart * b_s0 + (long) (colStart + p * 12) * b_s1) * 8L;
                for (int k = 0; k < kc; k++) {
                    long sAddr = rowBase;
                    for (int c = 0; c < 12; c++) {
                        dst.set(VD, dstOff + (long) c * 8L, src.get(VD, sAddr));
                        sAddr += colStrideBytes;
                    }
                    dst.set(VD, dstOff + 96L,  0.0d);
                    dst.set(VD, dstOff + 104L, 0.0d);
                    dst.set(VD, dstOff + 112L, 0.0d);
                    dst.set(VD, dstOff + 120L, 0.0d);
                    rowBase += srcRowStrideBytes;
                    dstOff += 128L;
                }
            }
        }
    }

    public static void packB_tail_Aarch_Double(MemorySegment src, MemorySegment dst,
                                               int rowStart, int colStart, int kc, int nc,
                                               long b_s0, long b_s1, int fullPanels, int tailCols) {
        long dstOff = (long) fullPanels * 16 * kc * 8L;
        final long srcRowStrideBytes = b_s0 * 8L;
        if (b_s1 == 1L) {
            long srcOff = ((long) rowStart * b_s0 + colStart + (long) fullPanels * 12) * 8L;
            long copyBytes = (long) tailCols * 8L;
            for (int k = 0; k < kc; k++) {
                MemorySegment.copy(src, srcOff, dst, dstOff, copyBytes);
                for (int c = tailCols; c < 16; c++) {
                    dst.set(VD, dstOff + (long) c * 8L, 0.0d);
                }
                srcOff += srcRowStrideBytes;
                dstOff += 128L;
            }
        } else {
            final long colStrideBytes = b_s1 * 8L;
            long rowBase = ((long) rowStart * b_s0 + (long) (colStart + fullPanels * 12) * b_s1) * 8L;
            for (int k = 0; k < kc; k++) {
                long sAddr = rowBase;
                int c = 0;
                for (; c < tailCols; c++) {
                    dst.set(VD, dstOff + (long) c * 8L, src.get(VD, sAddr));
                    sAddr += colStrideBytes;
                }
                for (; c < 16; c++) {
                    dst.set(VD, dstOff + (long) c * 8L, 0.0d);
                }
                rowBase += srcRowStrideBytes;
                dstOff += 128L;
            }
        }
    }

    public static final class PackBTask_Aarch_Double extends RecursiveAction {
        final MemorySegment src, dst;
        final int rowStart, colStart, kc, nc;
        final long b_s0, b_s1;
        final int pStart, pEnd, leafPanels;

        PackBTask_Aarch_Double(MemorySegment src, MemorySegment dst, int rowStart, int colStart,
                               int kc, int nc, long b_s0, long b_s1, int pStart, int pEnd) {
            this(src, dst, rowStart, colStart, kc, nc, b_s0, b_s1, pStart, pEnd, computePackBLeafPanels(pEnd - pStart));
        }

        PackBTask_Aarch_Double(MemorySegment src, MemorySegment dst, int rowStart, int colStart,
                               int kc, int nc, long b_s0, long b_s1, int pStart, int pEnd, int leafPanels) {
            this.src = src; this.dst = dst; this.rowStart = rowStart; this.colStart = colStart;
            this.kc = kc; this.nc = nc; this.b_s0 = b_s0; this.b_s1 = b_s1;
            this.pStart = pStart; this.pEnd = pEnd; this.leafPanels = leafPanels;
        }

        @Override
        protected void compute() {
            int panels = pEnd - pStart;
            if (panels <= leafPanels) {
                packB_panels_range_Aarch_Double(src, dst, rowStart, colStart, kc, b_s0, b_s1, pStart, pEnd);
            } else {
                int mid = pStart + (panels >>> 1);
                invokeAll(
                        new PackBTask_Aarch_Double(src, dst, rowStart, colStart, kc, nc, b_s0, b_s1, pStart, mid, leafPanels),
                        new PackBTask_Aarch_Double(src, dst, rowStart, colStart, kc, nc, b_s0, b_s1, mid, pEnd, leafPanels)
                );
            }
        }
    }

    // =========================================================================
    // 3. INT (INT32) PANEL PACKING ROUTINES
    // =========================================================================

    public static void packA_panel_Arm_Int(MemorySegment src, MemorySegment dst,
                                           int rowStart, int mc, int colStart, int kc,
                                           long a_s0, long a_s1) {
        int fullPanels = mc / MR;
        int tailRows = mc % MR;
        final long rowStrideBytes = a_s0 * 4L;
        final long panelBytes = (long) kc * STEP_A_ROW_BYTES_I32;

        if (a_s1 == 1L) {
            for (int p = 0; p < fullPanels; p++) {
                long dOff = (long) p * panelBytes;
                long r0 = ((long) (rowStart + p * MR) * a_s0 + colStart) * 4L;
                long r1 = r0 + rowStrideBytes;
                long r2 = r1 + rowStrideBytes;
                long r3 = r2 + rowStrideBytes;
                long r4 = r3 + rowStrideBytes;
                long r5 = r4 + rowStrideBytes;

                long kByteOff = 0L;
                int k = 0;
                for (; k <= kc - 4; k += 4) {
                    dst.set(VI, dOff,       src.get(VI, r0 + kByteOff));
                    dst.set(VI, dOff + 4L,  src.get(VI, r1 + kByteOff));
                    dst.set(VI, dOff + 8L,  src.get(VI, r2 + kByteOff));
                    dst.set(VI, dOff + 12L, src.get(VI, r3 + kByteOff));
                    dst.set(VI, dOff + 16L, src.get(VI, r4 + kByteOff));
                    dst.set(VI, dOff + 20L, src.get(VI, r5 + kByteOff));

                    long k1 = kByteOff + 4L;
                    long d1 = dOff + STEP_A_ROW_BYTES_I32;
                    dst.set(VI, d1,       src.get(VI, r0 + k1));
                    dst.set(VI, d1 + 4L,  src.get(VI, r1 + k1));
                    dst.set(VI, d1 + 8L,  src.get(VI, r2 + k1));
                    dst.set(VI, d1 + 12L, src.get(VI, r3 + k1));
                    dst.set(VI, d1 + 16L, src.get(VI, r4 + k1));
                    dst.set(VI, d1 + 20L, src.get(VI, r5 + k1));

                    long k2 = kByteOff + 8L;
                    long d2 = d1 + STEP_A_ROW_BYTES_I32;
                    dst.set(VI, d2,       src.get(VI, r0 + k2));
                    dst.set(VI, d2 + 4L,  src.get(VI, r1 + k2));
                    dst.set(VI, d2 + 8L,  src.get(VI, r2 + k2));
                    dst.set(VI, d2 + 12L, src.get(VI, r3 + k2));
                    dst.set(VI, d2 + 16L, src.get(VI, r4 + k2));
                    dst.set(VI, d2 + 20L, src.get(VI, r5 + k2));

                    long k3 = kByteOff + 12L;
                    long d3 = d2 + STEP_A_ROW_BYTES_I32;
                    dst.set(VI, d3,       src.get(VI, r0 + k3));
                    dst.set(VI, d3 + 4L,  src.get(VI, r1 + k3));
                    dst.set(VI, d3 + 8L,  src.get(VI, r2 + k3));
                    dst.set(VI, d3 + 12L, src.get(VI, r3 + k3));
                    dst.set(VI, d3 + 16L, src.get(VI, r4 + k3));
                    dst.set(VI, d3 + 20L, src.get(VI, r5 + k3));

                    kByteOff += 16L;
                    dOff += STEP_A_K4_BYTES_I32;
                }
                for (; k < kc; k++) {
                    dst.set(VI, dOff,       src.get(VI, r0 + kByteOff));
                    dst.set(VI, dOff + 4L,  src.get(VI, r1 + kByteOff));
                    dst.set(VI, dOff + 8L,  src.get(VI, r2 + kByteOff));
                    dst.set(VI, dOff + 12L, src.get(VI, r3 + kByteOff));
                    dst.set(VI, dOff + 16L, src.get(VI, r4 + kByteOff));
                    dst.set(VI, dOff + 20L, src.get(VI, r5 + kByteOff));
                    kByteOff += 4L;
                    dOff += STEP_A_ROW_BYTES_I32;
                }
            }
            if (tailRows > 0) {
                packA_tail_Arm_Int(src, dst, rowStart + fullPanels * MR, tailRows, colStart, kc, a_s0, 1L, (long) fullPanels * panelBytes);
            }
        } else {
            final long colStrideBytes = a_s1 * 4L;
            for (int p = 0; p < fullPanels; p++) {
                long dOff = (long) p * panelBytes;
                long r0 = ((long) (rowStart + p * MR) * a_s0 + (long) colStart * a_s1) * 4L;
                long r1 = r0 + rowStrideBytes;
                long r2 = r1 + rowStrideBytes;
                long r3 = r2 + rowStrideBytes;
                long r4 = r3 + rowStrideBytes;
                long r5 = r4 + rowStrideBytes;

                long kByteOff = 0L;
                for (int k = 0; k < kc; k++) {
                    dst.set(VI, dOff,       src.get(VI, r0 + kByteOff));
                    dst.set(VI, dOff + 4L,  src.get(VI, r1 + kByteOff));
                    dst.set(VI, dOff + 8L,  src.get(VI, r2 + kByteOff));
                    dst.set(VI, dOff + 12L, src.get(VI, r3 + kByteOff));
                    dst.set(VI, dOff + 16L, src.get(VI, r4 + kByteOff));
                    dst.set(VI, dOff + 20L, src.get(VI, r5 + kByteOff));
                    kByteOff += colStrideBytes;
                    dOff += STEP_A_ROW_BYTES_I32;
                }
            }
            if (tailRows > 0) {
                packA_tail_Arm_Int(src, dst, rowStart + fullPanels * MR, tailRows, colStart, kc, a_s0, a_s1, (long) fullPanels * panelBytes);
            }
        }
    }

    private static void packA_tail_Arm_Int(MemorySegment src, MemorySegment dst,
                                           int tailRowStart, int tailRows, int colStart, int kc,
                                           long a_s0, long a_s1, long dstBase) {
        final long colStrideBytes = a_s1 * 4L;
        final long rowStrideBytes = a_s0 * 4L;
        final long base0 = ((long) tailRowStart * a_s0 + (long) colStart * a_s1) * 4L;

        long dOff = dstBase;
        long kByteOff = 0L;
        for (int k = 0; k < kc; k++) {
            long sRow = base0 + kByteOff;
            int r = 0;
            for (; r < tailRows; r++) {
                dst.set(VI, dOff + (long) r * 4L, src.get(VI, sRow));
                sRow += rowStrideBytes;
            }
            for (; r < MR; r++) {
                dst.set(VI, dOff + (long) r * 4L, 0);
            }
            kByteOff += colStrideBytes;
            dOff += STEP_A_ROW_BYTES_I32;
        }
    }

    public static void packB_panel_Arm_Int(MemorySegment src, MemorySegment dst,
                                           int rowStart, int colStart, int kc, int nc,
                                           long b_s0, long b_s1) {
        int fullPanels = nc / NR_INT;
        int tailCols = nc % NR_INT;
        if (fullPanels >= 4 && AVAILABLE_CORES >= 2) {
            int leafPanels = computePackBLeafPanels(fullPanels);
            POOL.invoke(new PackBTask_Arm_Int(src, dst, rowStart, colStart, kc, nc, b_s0, b_s1, 0, fullPanels, leafPanels));
        } else {
            packB_panels_range_Arm_Int(src, dst, rowStart, colStart, kc, b_s0, b_s1, 0, fullPanels);
        }
        if (tailCols > 0) {
            packB_tail_Arm_Int(src, dst, rowStart, colStart, kc, nc, b_s0, b_s1, fullPanels, tailCols);
        }
    }

    public static void packB_panels_range_Arm_Int(MemorySegment src, MemorySegment dst,
                                                  int rowStart, int colStart, int kc,
                                                  long b_s0, long b_s1, int pStart, int pEnd) {
        final long panelBytes = (long) kc * STEP_B_ROW_BYTES_I32;
        final long srcRowStrideBytes = b_s0 * 4L;
        final long srcRowStride2 = srcRowStrideBytes * 2L;
        final long srcRowStride3 = srcRowStrideBytes * 3L;
        final long srcRowStride4 = srcRowStrideBytes * 4L;

        if (b_s1 == 1L) {
            for (int p = pStart; p < pEnd; p++) {
                long srcOff = ((long) rowStart * b_s0 + colStart + (long) p * NR_INT) * 4L;
                long dstOff = (long) p * panelBytes;
                int k = 0;
                for (; k <= kc - 4; k += 4) {
                    IntVector.fromMemorySegment(SPECIESINT, src, srcOff, NATIVE).intoMemorySegment(dst, dstOff, NATIVE);
                    IntVector.fromMemorySegment(SPECIESINT, src, srcOff + STRIDE_VEC_BYTES_I32, NATIVE).intoMemorySegment(dst, dstOff + STRIDE_VEC_BYTES_I32, NATIVE);

                    long s1 = srcOff + srcRowStrideBytes;
                    long d1 = dstOff + STEP_B_ROW_BYTES_I32;
                    IntVector.fromMemorySegment(SPECIESINT, src, s1, NATIVE).intoMemorySegment(dst, d1, NATIVE);
                    IntVector.fromMemorySegment(SPECIESINT, src, s1 + STRIDE_VEC_BYTES_I32, NATIVE).intoMemorySegment(dst, d1 + STRIDE_VEC_BYTES_I32, NATIVE);

                    long s2 = srcOff + srcRowStride2;
                    long d2 = d1 + STEP_B_ROW_BYTES_I32;
                    IntVector.fromMemorySegment(SPECIESINT, src, s2, NATIVE).intoMemorySegment(dst, d2, NATIVE);
                    IntVector.fromMemorySegment(SPECIESINT, src, s2 + STRIDE_VEC_BYTES_I32, NATIVE).intoMemorySegment(dst, d2 + STRIDE_VEC_BYTES_I32, NATIVE);

                    long s3 = srcOff + srcRowStride3;
                    long d3 = d2 + STEP_B_ROW_BYTES_I32;
                    IntVector.fromMemorySegment(SPECIESINT, src, s3, NATIVE).intoMemorySegment(dst, d3, NATIVE);
                    IntVector.fromMemorySegment(SPECIESINT, src, s3 + STRIDE_VEC_BYTES_I32, NATIVE).intoMemorySegment(dst, d3 + STRIDE_VEC_BYTES_I32, NATIVE);

                    srcOff += srcRowStride4;
                    dstOff += STEP_B_K4_BYTES_I32;
                }
                for (; k < kc; k++) {
                    IntVector.fromMemorySegment(SPECIESINT, src, srcOff, NATIVE).intoMemorySegment(dst, dstOff, NATIVE);
                    IntVector.fromMemorySegment(SPECIESINT, src, srcOff + STRIDE_VEC_BYTES_I32, NATIVE).intoMemorySegment(dst, dstOff + STRIDE_VEC_BYTES_I32, NATIVE);
                    srcOff += srcRowStrideBytes;
                    dstOff += STEP_B_ROW_BYTES_I32;
                }
            }
        } else {
            final long colStrideBytes = b_s1 * 4L;
            for (int p = pStart; p < pEnd; p++) {
                long dstOff = (long) p * panelBytes;
                long rowBase = ((long) rowStart * b_s0 + (long) (colStart + p * NR_INT) * b_s1) * 4L;
                for (int k = 0; k < kc; k++) {
                    long sAddr = rowBase;
                    for (int c = 0; c < NR_INT; c++) {
                        dst.set(VI, dstOff + (long) c * 4L, src.get(VI, sAddr));
                        sAddr += colStrideBytes;
                    }
                    rowBase += srcRowStrideBytes;
                    dstOff += STEP_B_ROW_BYTES_I32;
                }
            }
        }
    }

    public static void packB_tail_Arm_Int(MemorySegment src, MemorySegment dst,
                                          int rowStart, int colStart, int kc, int nc,
                                          long b_s0, long b_s1, int fullPanels, int tailCols) {
        long dstOff = (long) fullPanels * STEP_B_ROW_BYTES_I32 * kc;
        final long srcRowStrideBytes = b_s0 * 4L;

        if (b_s1 == 1L) {
            final int lanes0 = Math.min(VL_I32, tailCols);
            final int lanes1 = Math.max(0, tailCols - VL_I32);
            final VectorMask<Integer> mask0 = SPECIESINT.indexInRange(0, lanes0);
            final VectorMask<Integer> mask1 = SPECIESINT.indexInRange(0, lanes1);
            final IntVector zero = IntVector.zero(SPECIESINT);

            long srcOff = ((long) rowStart * b_s0 + colStart + (long) fullPanels * NR_INT) * 4L;
            for (int k = 0; k < kc; k++) {
                IntVector v0 = IntVector.fromMemorySegment(SPECIESINT, src, srcOff, NATIVE, mask0);
                v0.intoMemorySegment(dst, dstOff, NATIVE);
                IntVector v1 = (lanes1 > 0)
                        ? IntVector.fromMemorySegment(SPECIESINT, src, srcOff + STRIDE_VEC_BYTES_I32, NATIVE, mask1)
                        : zero;
                v1.intoMemorySegment(dst, dstOff + STRIDE_VEC_BYTES_I32, NATIVE);
                srcOff += srcRowStrideBytes;
                dstOff += STEP_B_ROW_BYTES_I32;
            }
        } else {
            final long colStrideBytes = b_s1 * 4L;
            long rowBase = ((long) rowStart * b_s0 + (long) (colStart + fullPanels * NR_INT) * b_s1) * 4L;
            for (int k = 0; k < kc; k++) {
                long sAddr = rowBase;
                int c = 0;
                for (; c < tailCols; c++) {
                    dst.set(VI, dstOff + (long) c * 4L, src.get(VI, sAddr));
                    sAddr += colStrideBytes;
                }
                for (; c < NR_INT; c++) {
                    dst.set(VI, dstOff + (long) c * 4L, 0);
                }
                rowBase += srcRowStrideBytes;
                dstOff += STEP_B_ROW_BYTES_I32;
            }
        }
    }

    public static final class PackBTask_Arm_Int extends RecursiveAction {
        final MemorySegment src, dst;
        final int rowStart, colStart, kc, nc;
        final long b_s0, b_s1;
        final int pStart, pEnd, leafPanels;

        PackBTask_Arm_Int(MemorySegment src, MemorySegment dst, int rowStart, int colStart,
                          int kc, int nc, long b_s0, long b_s1, int pStart, int pEnd) {
            this(src, dst, rowStart, colStart, kc, nc, b_s0, b_s1, pStart, pEnd, computePackBLeafPanels(pEnd - pStart));
        }

        PackBTask_Arm_Int(MemorySegment src, MemorySegment dst, int rowStart, int colStart,
                          int kc, int nc, long b_s0, long b_s1, int pStart, int pEnd, int leafPanels) {
            this.src = src; this.dst = dst; this.rowStart = rowStart; this.colStart = colStart;
            this.kc = kc; this.nc = nc; this.b_s0 = b_s0; this.b_s1 = b_s1;
            this.pStart = pStart; this.pEnd = pEnd; this.leafPanels = leafPanels;
        }

        @Override
        protected void compute() {
            int panels = pEnd - pStart;
            if (panels <= leafPanels) {
                packB_panels_range_Arm_Int(src, dst, rowStart, colStart, kc, b_s0, b_s1, pStart, pEnd);
            } else {
                int mid = pStart + (panels >>> 1);
                invokeAll(
                        new PackBTask_Arm_Int(src, dst, rowStart, colStart, kc, nc, b_s0, b_s1, pStart, mid, leafPanels),
                        new PackBTask_Arm_Int(src, dst, rowStart, colStart, kc, nc, b_s0, b_s1, mid, pEnd, leafPanels)
                );
            }
        }
    }

    public static void packA_panel_Aarch_Int(MemorySegment src, MemorySegment dst,
                                             int rowStart, int mc, int colStart, int kc,
                                             long a_s0, long a_s1) {
        int fullPanels = mc / 8;
        int tailRows = mc % 8;
        final long rowStrideBytes = a_s0 * 4L;
        final long colStrideBytes = a_s1 * 4L;
        final long panelBytes = (long) kc * 32L;

        for (int p = 0; p < fullPanels; p++) {
            long dOff = (long) p * panelBytes;
            long r0 = ((long) (rowStart + p * 8) * a_s0 + (long) colStart * a_s1) * 4L;
            long r1 = r0 + rowStrideBytes;
            long r2 = r1 + rowStrideBytes;
            long r3 = r2 + rowStrideBytes;
            long r4 = r3 + rowStrideBytes;
            long r5 = r4 + rowStrideBytes;
            long r6 = r5 + rowStrideBytes;
            long r7 = r6 + rowStrideBytes;

            long kByteOff = 0L;
            for (int k = 0; k < kc; k++) {
                dst.set(VI, dOff,       src.get(VI, r0 + kByteOff));
                dst.set(VI, dOff + 4L,  src.get(VI, r1 + kByteOff));
                dst.set(VI, dOff + 8L,  src.get(VI, r2 + kByteOff));
                dst.set(VI, dOff + 12L, src.get(VI, r3 + kByteOff));
                dst.set(VI, dOff + 16L, src.get(VI, r4 + kByteOff));
                dst.set(VI, dOff + 20L, src.get(VI, r5 + kByteOff));
                dst.set(VI, dOff + 24L, src.get(VI, r6 + kByteOff));
                dst.set(VI, dOff + 28L, src.get(VI, r7 + kByteOff));
                kByteOff += colStrideBytes;
                dOff += 32L;
            }
        }
        if (tailRows > 0) {
            long dOff = (long) fullPanels * panelBytes;
            long base0 = ((long) (rowStart + fullPanels * 8) * a_s0 + (long) colStart * a_s1) * 4L;
            long kByteOff = 0L;
            for (int k = 0; k < kc; k++) {
                long sRow = base0 + kByteOff;
                int r = 0;
                for (; r < tailRows; r++) {
                    dst.set(VI, dOff + (long) r * 4L, src.get(VI, sRow));
                    sRow += rowStrideBytes;
                }
                for (; r < 8; r++) {
                    dst.set(VI, dOff + (long) r * 4L, 0);
                }
                kByteOff += colStrideBytes;
                dOff += 32L;
            }
        }
    }

    public static void packB_panel_Aarch_Int(MemorySegment src, MemorySegment dst,
                                             int rowStart, int colStart, int kc, int nc,
                                             long b_s0, long b_s1) {
        int fullPanels = nc / 12;
        int tailCols = nc % 12;
        if (fullPanels >= 4 && AVAILABLE_CORES >= 2) {
            int leafPanels = computePackBLeafPanels(fullPanels);
            POOL.invoke(new PackBTask_Aarch_Int(src, dst, rowStart, colStart, kc, nc, b_s0, b_s1, 0, fullPanels, leafPanels));
        } else {
            packB_panels_range_Aarch_Int(src, dst, rowStart, colStart, kc, b_s0, b_s1, 0, fullPanels);
        }
        if (tailCols > 0) {
            packB_tail_Aarch_Int(src, dst, rowStart, colStart, kc, nc, b_s0, b_s1, fullPanels, tailCols);
        }
    }

    public static void packB_panels_range_Aarch_Int(MemorySegment src, MemorySegment dst,
                                                    int rowStart, int colStart, int kc,
                                                    long b_s0, long b_s1, int pStart, int pEnd) {
        final long srcRowStrideBytes = b_s0 * 4L;
        if (b_s1 == 1L) {
            for (int p = pStart; p < pEnd; p++) {
                long srcOff = ((long) rowStart * b_s0 + colStart + (long) p * 12) * 4L;
                long dstOff = (long) p * 16 * kc * 4L;
                for (int k = 0; k < kc; k++) {
                    MemorySegment.copy(src, srcOff, dst, dstOff, 48L);
                    dst.set(VI, dstOff + 48L, 0);
                    dst.set(VI, dstOff + 52L, 0);
                    dst.set(VI, dstOff + 56L, 0);
                    dst.set(VI, dstOff + 60L, 0);
                    srcOff += srcRowStrideBytes;
                    dstOff += 64L;
                }
            }
        } else {
            final long colStrideBytes = b_s1 * 4L;
            for (int p = pStart; p < pEnd; p++) {
                long dstOff = (long) p * 16 * kc * 4L;
                long rowBase = ((long) rowStart * b_s0 + (long) (colStart + p * 12) * b_s1) * 4L;
                for (int k = 0; k < kc; k++) {
                    long sAddr = rowBase;
                    for (int c = 0; c < 12; c++) {
                        dst.set(VI, dstOff + (long) c * 4L, src.get(VI, sAddr));
                        sAddr += colStrideBytes;
                    }
                    dst.set(VI, dstOff + 48L, 0);
                    dst.set(VI, dstOff + 52L, 0);
                    dst.set(VI, dstOff + 56L, 0);
                    dst.set(VI, dstOff + 60L, 0);
                    rowBase += srcRowStrideBytes;
                    dstOff += 64L;
                }
            }
        }
    }

    public static void packB_tail_Aarch_Int(MemorySegment src, MemorySegment dst,
                                            int rowStart, int colStart, int kc, int nc,
                                            long b_s0, long b_s1, int fullPanels, int tailCols) {
        long dstOff = (long) fullPanels * 16 * kc * 4L;
        final long srcRowStrideBytes = b_s0 * 4L;
        if (b_s1 == 1L) {
            long srcOff = ((long) rowStart * b_s0 + colStart + (long) fullPanels * 12) * 4L;
            long copyBytes = (long) tailCols * 4L;
            for (int k = 0; k < kc; k++) {
                MemorySegment.copy(src, srcOff, dst, dstOff, copyBytes);
                for (int c = tailCols; c < 16; c++) {
                    dst.set(VI, dstOff + (long) c * 4L, 0);
                }
                srcOff += srcRowStrideBytes;
                dstOff += 64L;
            }
        } else {
            final long colStrideBytes = b_s1 * 4L;
            long rowBase = ((long) rowStart * b_s0 + (long) (colStart + fullPanels * 12) * b_s1) * 4L;
            for (int k = 0; k < kc; k++) {
                long sAddr = rowBase;
                int c = 0;
                for (; c < tailCols; c++) {
                    dst.set(VI, dstOff + (long) c * 4L, src.get(VI, sAddr));
                    sAddr += colStrideBytes;
                }
                for (; c < 16; c++) {
                    dst.set(VI, dstOff + (long) c * 4L, 0);
                }
                rowBase += srcRowStrideBytes;
                dstOff += 64L;
            }
        }
    }

    public static final class PackBTask_Aarch_Int extends RecursiveAction {
        final MemorySegment src, dst;
        final int rowStart, colStart, kc, nc;
        final long b_s0, b_s1;
        final int pStart, pEnd, leafPanels;

        PackBTask_Aarch_Int(MemorySegment src, MemorySegment dst, int rowStart, int colStart,
                            int kc, int nc, long b_s0, long b_s1, int pStart, int pEnd) {
            this(src, dst, rowStart, colStart, kc, nc, b_s0, b_s1, pStart, pEnd, computePackBLeafPanels(pEnd - pStart));
        }

        PackBTask_Aarch_Int(MemorySegment src, MemorySegment dst, int rowStart, int colStart,
                            int kc, int nc, long b_s0, long b_s1, int pStart, int pEnd, int leafPanels) {
            this.src = src; this.dst = dst; this.rowStart = rowStart; this.colStart = colStart;
            this.kc = kc; this.nc = nc; this.b_s0 = b_s0; this.b_s1 = b_s1;
            this.pStart = pStart; this.pEnd = pEnd; this.leafPanels = leafPanels;
        }

        @Override
        protected void compute() {
            int panels = pEnd - pStart;
            if (panels <= leafPanels) {
                packB_panels_range_Aarch_Int(src, dst, rowStart, colStart, kc, b_s0, b_s1, pStart, pEnd);
            } else {
                int mid = pStart + (panels >>> 1);
                invokeAll(
                        new PackBTask_Aarch_Int(src, dst, rowStart, colStart, kc, nc, b_s0, b_s1, pStart, mid, leafPanels),
                        new PackBTask_Aarch_Int(src, dst, rowStart, colStart, kc, nc, b_s0, b_s1, mid, pEnd, leafPanels)
                );
            }
        }
    }

    // =========================================================================
    // BACKWARD-COMPATIBILITY ALIASES FOR TESTS AND EXTERNAL CALLERS
    // =========================================================================

    public static void packA_panel_Float(MemorySegment src, MemorySegment dst, int rowStart, int mc, int colStart, int kc, long a_s0, long a_s1) {
        packA_panel_Arm_Float(src, dst, rowStart, mc, colStart, kc, a_s0, a_s1);
    }

    public static void packB_panel_Float(MemorySegment src, MemorySegment dst, int rowStart, int colStart, int kc, int nc, long b_s0, long b_s1) {
        packB_panel_Arm_Float(src, dst, rowStart, colStart, kc, nc, b_s0, b_s1);
    }

    public static void packB_panel_sequential_Float(MemorySegment src, MemorySegment dst, int rowStart, int colStart, int kc, int nc, long b_s0, long b_s1) {
        int fullPanels = nc / NR;
        int tailCols = nc % NR;
        packB_panels_range_Arm_Float(src, dst, rowStart, colStart, kc, b_s0, b_s1, 0, fullPanels);
        if (tailCols > 0) {
            packB_tail_Arm_Float(src, dst, rowStart, colStart, kc, nc, b_s0, b_s1, fullPanels, tailCols);
        }
    }

    public static void packA_panel_Double(MemorySegment src, MemorySegment dst, int rowStart, int mc, int colStart, int kc, long a_s0, long a_s1) {
        packA_panel_Arm_Double(src, dst, rowStart, mc, colStart, kc, a_s0, a_s1);
    }

    public static void packB_panel_Double(MemorySegment src, MemorySegment dst, int rowStart, int colStart, int kc, int nc, long b_s0, long b_s1) {
        packB_panel_Arm_Double(src, dst, rowStart, colStart, kc, nc, b_s0, b_s1);
    }

    public static void packB_panel_sequential_Double(MemorySegment src, MemorySegment dst, int rowStart, int colStart, int kc, int nc, long b_s0, long b_s1) {
        int fullPanels = nc / NR_DB;
        int tailCols = nc % NR_DB;
        packB_panels_range_Arm_Double(src, dst, rowStart, colStart, kc, b_s0, b_s1, 0, fullPanels);
        if (tailCols > 0) {
            packB_tail_Arm_Double(src, dst, rowStart, colStart, kc, nc, b_s0, b_s1, fullPanels, tailCols);
        }
    }

    public static void packA_panel_Int(MemorySegment src, MemorySegment dst, int rowStart, int mc, int colStart, int kc, long a_s0, long a_s1) {
        packA_panel_Arm_Int(src, dst, rowStart, mc, colStart, kc, a_s0, a_s1);
    }

    public static void packB_panel_Int(MemorySegment src, MemorySegment dst, int rowStart, int colStart, int kc, int nc, long b_s0, long b_s1) {
        packB_panel_Arm_Int(src, dst, rowStart, colStart, kc, nc, b_s0, b_s1);
    }

    public static void packB_panel_sequential_Int(MemorySegment src, MemorySegment dst, int rowStart, int colStart, int kc, int nc, long b_s0, long b_s1) {
        int fullPanels = nc / NR_INT;
        int tailCols = nc % NR_INT;
        packB_panels_range_Arm_Int(src, dst, rowStart, colStart, kc, b_s0, b_s1, 0, fullPanels);
        if (tailCols > 0) {
            packB_tail_Arm_Int(src, dst, rowStart, colStart, kc, nc, b_s0, b_s1, fullPanels, tailCols);
        }
    }
}