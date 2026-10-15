package jnum.internal.kernel.linalg.matmul;

import static jnum.internal.Constants.*;

import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorSpecies;

/**
 * BLIS Compute Microkernels and GEBP Macro-kernels.
 * Engineered for HotSpot C2 (JDK 25 - 27):
 * - 2-way k-loop unrolling with ping-pong register reuse (16 live vector locals = 16 AVX2 YMM registers).
 * - Pre-loop C accumulator loading to eliminate post-loop read-modify-write bloat.
 * - Pure 64-bit long pointer induction variables so C2 hoists ScopedMemoryAccess checks.
 * - Zero-allocation SIMD VectorMask edge handling for tail columns.
 */
public final class MatMulMicroKernels {

    private static final ByteOrder NATIVE = NATIVE_ORDER;
    private static final VectorSpecies<Float> SPECIES = SPECIES_F32;
    private static final VectorSpecies<Double> SPECIESDB = SPECIES_F64;
    private static final VectorSpecies<Integer> SPECIESINT = SPECIES_I32;

    private MatMulMicroKernels() {
        throw new AssertionError("Utility class");
    }

    // =========================================================================
    // 1. FLOAT (FP32) MACRO & MICRO KERNELS
    // =========================================================================

    public static void gebpMacroKernel_Arm_Float(MemorySegment pA, MemorySegment pB, MemorySegment C,
                                                 int rowStart, int mc, int jc, int nc, int kc, int p,
                                                 boolean isFirstKBlock) {
        int nrPanels = (nc + NR - 1) / NR;
        int fullIPanels = mc / MR;
        int tailRows = mc % MR;

        for (int jp = 0; jp < nrPanels; jp++) {
            int jr = jp * NR;
            int actualNR = Math.min(NR, nc - jr);
            long bBase = (long) jp * NR * kc * 4L;
            boolean fullNR = (actualNR == NR);

            for (int ip = 0; ip < fullIPanels; ip++) {
                long aBase = (long) ip * MR * kc * 4L;
                int ci = rowStart + ip * MR;
                int cj = jc + jr;
                if (fullNR) {
                    microKernel6x16_Float(pA, aBase, pB, bBase, C, ci, cj, kc, p, isFirstKBlock);
                } else {
                    microKernel6x16Masked_Float(pA, aBase, pB, bBase, C, ci, cj, kc, p, actualNR, isFirstKBlock);
                }
            }

            if (tailRows > 0) {
                long aBase = (long) fullIPanels * MR * kc * 4L;
                int ci = rowStart + fullIPanels * MR;
                int cj = jc + jr;
                int rOff = 0;
                while (rOff + 2 <= tailRows) {
                    if (fullNR) {
                        microKernel2x16_Float(pA, aBase, rOff, pB, bBase, C, ci + rOff, cj, kc, p, isFirstKBlock);
                    } else {
                        microKernel2x16Masked_Float(pA, aBase, rOff, pB, bBase, C, ci + rOff, cj, kc, p, actualNR, isFirstKBlock);
                    }
                    rOff += 2;
                }
                if (rOff < tailRows) {
                    if (fullNR) {
                        microKernel1x16_Float(pA, aBase, rOff, pB, bBase, C, ci + rOff, cj, kc, p, isFirstKBlock);
                    } else {
                        microKernel1x16Masked_Float(pA, aBase, rOff, pB, bBase, C, ci + rOff, cj, kc, p, actualNR, isFirstKBlock);
                    }
                }
            }
        }
    }

    public static void gebpMacroKernel_Aarch_Float(MemorySegment pA, MemorySegment pB, MemorySegment C,
                                                   int rowStart, int mc, int jc, int nc, int kc, int p,
                                                   boolean isFirstKBlock) {
        int nrPanels = (nc + 11) / 12;
        int fullIPanels = mc / 8;
        int tailRows = mc % 8;

        for (int jp = 0; jp < nrPanels; jp++) {
            int jr = jp * 12;
            int actualNR = Math.min(12, nc - jr);
            long bBase = (long) jp * 16 * kc * 4L;
            if (actualNR == 12) {
                for (int ip = 0; ip < fullIPanels; ip++) {
                    microKernel8x12_Float(pA, (long) ip * 8 * kc * 4L, pB, bBase, C, rowStart + ip * 8, jc + jr, kc, p, isFirstKBlock);
                }
                if (tailRows > 0) {
                    microKernelScalar_Float(pA, (long) fullIPanels * 8 * kc * 4L, 0, pB, bBase, C, rowStart + fullIPanels * 8, jc + jr, kc, p, tailRows, 12, 8, 16, isFirstKBlock);
                }
            } else {
                for (int ip = 0; ip < fullIPanels; ip++) {
                    microKernelScalar_Float(pA, (long) ip * 8 * kc * 4L, 0, pB, bBase, C, rowStart + ip * 8, jc + jr, kc, p, 8, actualNR, 8, 16, isFirstKBlock);
                }
                if (tailRows > 0) {
                    microKernelScalar_Float(pA, (long) fullIPanels * 8 * kc * 4L, 0, pB, bBase, C, rowStart + fullIPanels * 8, jc + jr, kc, p, tailRows, actualNR, 8, 16, isFirstKBlock);
                }
            }
        }
    }

    /**
     * Microkernel 6x(2*V) Float - 2-way unrolled with 16 live vector locals.
     */
    public static void microKernel6x16_Float(MemorySegment pA, long aBase, MemorySegment pB, long bBase,
                                             MemorySegment C, int ci, int cj, int kc, int N, boolean isFirstKBlock) {
        final long cStride = (long) N * 4L;
        final long row0 = ((long) ci * N + cj) * 4L;
        final long row1 = row0 + cStride;
        final long row2 = row1 + cStride;
        final long row3 = row2 + cStride;
        final long row4 = row3 + cStride;
        final long row5 = row4 + cStride;

        FloatVector c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51;
        if (isFirstKBlock) {
            var z = FloatVector.zero(SPECIES);
            c00 = z; c01 = z;
            c10 = z; c11 = z;
            c20 = z; c21 = z;
            c30 = z; c31 = z;
            c40 = z; c41 = z;
            c50 = z; c51 = z;
        } else {
            c00 = FloatVector.fromMemorySegment(SPECIES, C, row0, NATIVE);
            c01 = FloatVector.fromMemorySegment(SPECIES, C, row0 + STRIDE_VEC_BYTES_F32, NATIVE);
            c10 = FloatVector.fromMemorySegment(SPECIES, C, row1, NATIVE);
            c11 = FloatVector.fromMemorySegment(SPECIES, C, row1 + STRIDE_VEC_BYTES_F32, NATIVE);
            c20 = FloatVector.fromMemorySegment(SPECIES, C, row2, NATIVE);
            c21 = FloatVector.fromMemorySegment(SPECIES, C, row2 + STRIDE_VEC_BYTES_F32, NATIVE);
            c30 = FloatVector.fromMemorySegment(SPECIES, C, row3, NATIVE);
            c31 = FloatVector.fromMemorySegment(SPECIES, C, row3 + STRIDE_VEC_BYTES_F32, NATIVE);
            c40 = FloatVector.fromMemorySegment(SPECIES, C, row4, NATIVE);
            c41 = FloatVector.fromMemorySegment(SPECIES, C, row4 + STRIDE_VEC_BYTES_F32, NATIVE);
            c50 = FloatVector.fromMemorySegment(SPECIES, C, row5, NATIVE);
            c51 = FloatVector.fromMemorySegment(SPECIES, C, row5 + STRIDE_VEC_BYTES_F32, NATIVE);
        }

        long aOff = aBase;
        long bOff = bBase;
        int k = 0;
        final int kUnroll = kc - 1;

        for (; k < kUnroll; k += 2) {
            // Iteration 0
            var b0 = FloatVector.fromMemorySegment(SPECIES, pB, bOff, NATIVE);
            var b1 = FloatVector.fromMemorySegment(SPECIES, pB, bOff + STRIDE_VEC_BYTES_F32, NATIVE);

            var a0 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff));
            var a1 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff + 4L));
            c00 = a0.fma(b0, c00); c01 = a0.fma(b1, c01);
            c10 = a1.fma(b0, c10); c11 = a1.fma(b1, c11);

            a0 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff + 8L));
            a1 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff + 12L));
            c20 = a0.fma(b0, c20); c21 = a0.fma(b1, c21);
            c30 = a1.fma(b0, c30); c31 = a1.fma(b1, c31);

            a0 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff + 16L));
            a1 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff + 20L));
            c40 = a0.fma(b0, c40); c41 = a0.fma(b1, c41);
            c50 = a1.fma(b0, c50); c51 = a1.fma(b1, c51);

            // Iteration 1
            long aOff1 = aOff + STEP_A_ROW_BYTES_F32;
            long bOff1 = bOff + STEP_B_ROW_BYTES_F32;
            b0 = FloatVector.fromMemorySegment(SPECIES, pB, bOff1, NATIVE);
            b1 = FloatVector.fromMemorySegment(SPECIES, pB, bOff1 + STRIDE_VEC_BYTES_F32, NATIVE);

            a0 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff1));
            a1 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff1 + 4L));
            c00 = a0.fma(b0, c00); c01 = a0.fma(b1, c01);
            c10 = a1.fma(b0, c10); c11 = a1.fma(b1, c11);

            a0 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff1 + 8L));
            a1 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff1 + 12L));
            c20 = a0.fma(b0, c20); c21 = a0.fma(b1, c21);
            c30 = a1.fma(b0, c30); c31 = a1.fma(b1, c31);

            a0 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff1 + 16L));
            a1 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff1 + 20L));
            c40 = a0.fma(b0, c40); c41 = a0.fma(b1, c41);
            c50 = a1.fma(b0, c50); c51 = a1.fma(b1, c51);

            aOff = aOff1 + STEP_A_ROW_BYTES_F32;
            bOff = bOff1 + STEP_B_ROW_BYTES_F32;
        }

        for (; k < kc; k++) {
            var b0 = FloatVector.fromMemorySegment(SPECIES, pB, bOff, NATIVE);
            var b1 = FloatVector.fromMemorySegment(SPECIES, pB, bOff + STRIDE_VEC_BYTES_F32, NATIVE);

            var a0 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff));
            var a1 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff + 4L));
            c00 = a0.fma(b0, c00); c01 = a0.fma(b1, c01);
            c10 = a1.fma(b0, c10); c11 = a1.fma(b1, c11);

            a0 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff + 8L));
            a1 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff + 12L));
            c20 = a0.fma(b0, c20); c21 = a0.fma(b1, c21);
            c30 = a1.fma(b0, c30); c31 = a1.fma(b1, c31);

            a0 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff + 16L));
            a1 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff + 20L));
            c40 = a0.fma(b0, c40); c41 = a0.fma(b1, c41);
            c50 = a1.fma(b0, c50); c51 = a1.fma(b1, c51);

            aOff += STEP_A_ROW_BYTES_F32;
            bOff += STEP_B_ROW_BYTES_F32;
        }

        c00.intoMemorySegment(C, row0, NATIVE); c01.intoMemorySegment(C, row0 + STRIDE_VEC_BYTES_F32, NATIVE);
        c10.intoMemorySegment(C, row1, NATIVE); c11.intoMemorySegment(C, row1 + STRIDE_VEC_BYTES_F32, NATIVE);
        c20.intoMemorySegment(C, row2, NATIVE); c21.intoMemorySegment(C, row2 + STRIDE_VEC_BYTES_F32, NATIVE);
        c30.intoMemorySegment(C, row3, NATIVE); c31.intoMemorySegment(C, row3 + STRIDE_VEC_BYTES_F32, NATIVE);
        c40.intoMemorySegment(C, row4, NATIVE); c41.intoMemorySegment(C, row4 + STRIDE_VEC_BYTES_F32, NATIVE);
        c50.intoMemorySegment(C, row5, NATIVE); c51.intoMemorySegment(C, row5 + STRIDE_VEC_BYTES_F32, NATIVE);
    }

    private static void microKernel6x16Masked_Float(MemorySegment pA, long aBase, MemorySegment pB, long bBase,
                                                    MemorySegment C, int ci, int cj, int kc, int N, int actualNR,
                                                    boolean isFirstKBlock) {
        final int vLen = VL_F32;
        final int lanes0 = Math.min(vLen, actualNR);
        final int lanes1 = Math.max(0, actualNR - vLen);
        final VectorMask<Float> mask0 = SPECIES.indexInRange(0, lanes0);
        final VectorMask<Float> mask1 = SPECIES.indexInRange(0, lanes1);

        var z = FloatVector.zero(SPECIES);
        FloatVector c00 = z, c01 = z, c10 = z, c11 = z, c20 = z, c21 = z;
        FloatVector c30 = z, c31 = z, c40 = z, c41 = z, c50 = z, c51 = z;

        long aOff = aBase;
        long bOff = bBase;
        for (int k = 0; k < kc; k++) {
            var b0 = FloatVector.fromMemorySegment(SPECIES, pB, bOff, NATIVE);
            var b1 = FloatVector.fromMemorySegment(SPECIES, pB, bOff + STRIDE_VEC_BYTES_F32, NATIVE);

            var a0 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff));
            var a1 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff + 4L));
            c00 = a0.fma(b0, c00); c01 = a0.fma(b1, c01);
            c10 = a1.fma(b0, c10); c11 = a1.fma(b1, c11);

            a0 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff + 8L));
            a1 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff + 12L));
            c20 = a0.fma(b0, c20); c21 = a0.fma(b1, c21);
            c30 = a1.fma(b0, c30); c31 = a1.fma(b1, c31);

            a0 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff + 16L));
            a1 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff + 20L));
            c40 = a0.fma(b0, c40); c41 = a0.fma(b1, c41);
            c50 = a1.fma(b0, c50); c51 = a1.fma(b1, c51);

            aOff += STEP_A_ROW_BYTES_F32;
            bOff += STEP_B_ROW_BYTES_F32;
        }

        final long cStride = (long) N * 4L;
        long row0 = ((long) ci * N + cj) * 4L;
        storeMaskedRow_Float(C, row0, c00, c01, mask0, mask1, lanes1 > 0, isFirstKBlock);
        row0 += cStride;
        storeMaskedRow_Float(C, row0, c10, c11, mask0, mask1, lanes1 > 0, isFirstKBlock);
        row0 += cStride;
        storeMaskedRow_Float(C, row0, c20, c21, mask0, mask1, lanes1 > 0, isFirstKBlock);
        row0 += cStride;
        storeMaskedRow_Float(C, row0, c30, c31, mask0, mask1, lanes1 > 0, isFirstKBlock);
        row0 += cStride;
        storeMaskedRow_Float(C, row0, c40, c41, mask0, mask1, lanes1 > 0, isFirstKBlock);
        row0 += cStride;
        storeMaskedRow_Float(C, row0, c50, c51, mask0, mask1, lanes1 > 0, isFirstKBlock);
    }

    private static void storeMaskedRow_Float(MemorySegment C, long rowOff, FloatVector v0, FloatVector v1,
                                             VectorMask<Float> mask0, VectorMask<Float> mask1,
                                             boolean hasSecondVec, boolean isFirstKBlock) {
        if (!isFirstKBlock) {
            v0 = v0.add(FloatVector.fromMemorySegment(SPECIES, C, rowOff, NATIVE, mask0));
        }
        v0.intoMemorySegment(C, rowOff, NATIVE, mask0);
        if (hasSecondVec) {
            long off1 = rowOff + STRIDE_VEC_BYTES_F32;
            if (!isFirstKBlock) {
                v1 = v1.add(FloatVector.fromMemorySegment(SPECIES, C, off1, NATIVE, mask1));
            }
            v1.intoMemorySegment(C, off1, NATIVE, mask1);
        }
    }

    /**
     * Microkernel 2x(2*V) Float - 2-way unrolled.
     */
    public static void microKernel2x16_Float(MemorySegment pA, long aBase, int rOff,
                                             MemorySegment pB, long bBase,
                                             MemorySegment C, int ci, int cj,
                                             int kc, int N, boolean isFirstKBlock) {
        final long row0 = ((long) ci * N + cj) * 4L;
        final long row1 = row0 + (long) N * 4L;

        FloatVector c00, c01, c10, c11;
        if (isFirstKBlock) {
            var z = FloatVector.zero(SPECIES);
            c00 = z; c01 = z; c10 = z; c11 = z;
        } else {
            c00 = FloatVector.fromMemorySegment(SPECIES, C, row0, NATIVE);
            c01 = FloatVector.fromMemorySegment(SPECIES, C, row0 + STRIDE_VEC_BYTES_F32, NATIVE);
            c10 = FloatVector.fromMemorySegment(SPECIES, C, row1, NATIVE);
            c11 = FloatVector.fromMemorySegment(SPECIES, C, row1 + STRIDE_VEC_BYTES_F32, NATIVE);
        }

        long aOff = aBase + (long) rOff * 4L;
        long bOff = bBase;
        int k = 0;
        final int kUnroll = kc - 1;

        for (; k < kUnroll; k += 2) {
            var b0 = FloatVector.fromMemorySegment(SPECIES, pB, bOff, NATIVE);
            var b1 = FloatVector.fromMemorySegment(SPECIES, pB, bOff + STRIDE_VEC_BYTES_F32, NATIVE);
            var a0 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff));
            var a1 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff + 4L));
            c00 = a0.fma(b0, c00); c01 = a0.fma(b1, c01);
            c10 = a1.fma(b0, c10); c11 = a1.fma(b1, c11);

            long aOff1 = aOff + STEP_A_ROW_BYTES_F32;
            long bOff1 = bOff + STEP_B_ROW_BYTES_F32;
            b0 = FloatVector.fromMemorySegment(SPECIES, pB, bOff1, NATIVE);
            b1 = FloatVector.fromMemorySegment(SPECIES, pB, bOff1 + STRIDE_VEC_BYTES_F32, NATIVE);
            a0 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff1));
            a1 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff1 + 4L));
            c00 = a0.fma(b0, c00); c01 = a0.fma(b1, c01);
            c10 = a1.fma(b0, c10); c11 = a1.fma(b1, c11);

            aOff = aOff1 + STEP_A_ROW_BYTES_F32;
            bOff = bOff1 + STEP_B_ROW_BYTES_F32;
        }

        for (; k < kc; k++) {
            var b0 = FloatVector.fromMemorySegment(SPECIES, pB, bOff, NATIVE);
            var b1 = FloatVector.fromMemorySegment(SPECIES, pB, bOff + STRIDE_VEC_BYTES_F32, NATIVE);
            var a0 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff));
            var a1 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff + 4L));
            c00 = a0.fma(b0, c00); c01 = a0.fma(b1, c01);
            c10 = a1.fma(b0, c10); c11 = a1.fma(b1, c11);

            aOff += STEP_A_ROW_BYTES_F32;
            bOff += STEP_B_ROW_BYTES_F32;
        }

        c00.intoMemorySegment(C, row0, NATIVE); c01.intoMemorySegment(C, row0 + STRIDE_VEC_BYTES_F32, NATIVE);
        c10.intoMemorySegment(C, row1, NATIVE); c11.intoMemorySegment(C, row1 + STRIDE_VEC_BYTES_F32, NATIVE);
    }

    private static void microKernel2x16Masked_Float(MemorySegment pA, long aBase, int rOff,
                                                    MemorySegment pB, long bBase,
                                                    MemorySegment C, int ci, int cj,
                                                    int kc, int N, int actualNR, boolean isFirstKBlock) {
        final int vLen = VL_F32;
        final int lanes0 = Math.min(vLen, actualNR);
        final int lanes1 = Math.max(0, actualNR - vLen);
        final VectorMask<Float> mask0 = SPECIES.indexInRange(0, lanes0);
        final VectorMask<Float> mask1 = SPECIES.indexInRange(0, lanes1);

        var z = FloatVector.zero(SPECIES);
        FloatVector c00 = z, c01 = z, c10 = z, c11 = z;

        long aOff = aBase + (long) rOff * 4L;
        long bOff = bBase;
        for (int k = 0; k < kc; k++) {
            var b0 = FloatVector.fromMemorySegment(SPECIES, pB, bOff, NATIVE);
            var b1 = FloatVector.fromMemorySegment(SPECIES, pB, bOff + STRIDE_VEC_BYTES_F32, NATIVE);
            var a0 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff));
            var a1 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff + 4L));
            c00 = a0.fma(b0, c00); c01 = a0.fma(b1, c01);
            c10 = a1.fma(b0, c10); c11 = a1.fma(b1, c11);

            aOff += STEP_A_ROW_BYTES_F32;
            bOff += STEP_B_ROW_BYTES_F32;
        }

        long row0 = ((long) ci * N + cj) * 4L;
        storeMaskedRow_Float(C, row0, c00, c01, mask0, mask1, lanes1 > 0, isFirstKBlock);
        storeMaskedRow_Float(C, row0 + (long) N * 4L, c10, c11, mask0, mask1, lanes1 > 0, isFirstKBlock);
    }

    /**
     * Microkernel 1x(2*V) Float - 2-way unrolled.
     */
    public static void microKernel1x16_Float(MemorySegment pA, long aBase, int rOff,
                                             MemorySegment pB, long bBase,
                                             MemorySegment C, int ci, int cj,
                                             int kc, int N, boolean isFirstKBlock) {
        final long row = ((long) ci * N + cj) * 4L;

        FloatVector c00, c01;
        if (isFirstKBlock) {
            var z = FloatVector.zero(SPECIES);
            c00 = z; c01 = z;
        } else {
            c00 = FloatVector.fromMemorySegment(SPECIES, C, row, NATIVE);
            c01 = FloatVector.fromMemorySegment(SPECIES, C, row + STRIDE_VEC_BYTES_F32, NATIVE);
        }

        long aOff = aBase + (long) rOff * 4L;
        long bOff = bBase;
        int k = 0;
        final int kUnroll = kc - 1;

        for (; k < kUnroll; k += 2) {
            var b0 = FloatVector.fromMemorySegment(SPECIES, pB, bOff, NATIVE);
            var b1 = FloatVector.fromMemorySegment(SPECIES, pB, bOff + STRIDE_VEC_BYTES_F32, NATIVE);
            var a0 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff));
            c00 = a0.fma(b0, c00); c01 = a0.fma(b1, c01);

            long aOff1 = aOff + STEP_A_ROW_BYTES_F32;
            long bOff1 = bOff + STEP_B_ROW_BYTES_F32;
            b0 = FloatVector.fromMemorySegment(SPECIES, pB, bOff1, NATIVE);
            b1 = FloatVector.fromMemorySegment(SPECIES, pB, bOff1 + STRIDE_VEC_BYTES_F32, NATIVE);
            a0 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff1));
            c00 = a0.fma(b0, c00); c01 = a0.fma(b1, c01);

            aOff = aOff1 + STEP_A_ROW_BYTES_F32;
            bOff = bOff1 + STEP_B_ROW_BYTES_F32;
        }

        for (; k < kc; k++) {
            var b0 = FloatVector.fromMemorySegment(SPECIES, pB, bOff, NATIVE);
            var b1 = FloatVector.fromMemorySegment(SPECIES, pB, bOff + STRIDE_VEC_BYTES_F32, NATIVE);
            var a0 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff));
            c00 = a0.fma(b0, c00); c01 = a0.fma(b1, c01);

            aOff += STEP_A_ROW_BYTES_F32;
            bOff += STEP_B_ROW_BYTES_F32;
        }

        c00.intoMemorySegment(C, row, NATIVE);
        c01.intoMemorySegment(C, row + STRIDE_VEC_BYTES_F32, NATIVE);
    }

    private static void microKernel1x16Masked_Float(MemorySegment pA, long aBase, int rOff,
                                                    MemorySegment pB, long bBase,
                                                    MemorySegment C, int ci, int cj,
                                                    int kc, int N, int actualNR, boolean isFirstKBlock) {
        final int vLen = VL_F32;
        final int lanes0 = Math.min(vLen, actualNR);
        final int lanes1 = Math.max(0, actualNR - vLen);
        final VectorMask<Float> mask0 = SPECIES.indexInRange(0, lanes0);
        final VectorMask<Float> mask1 = SPECIES.indexInRange(0, lanes1);

        var z = FloatVector.zero(SPECIES);
        FloatVector c00 = z, c01 = z;

        long aOff = aBase + (long) rOff * 4L;
        long bOff = bBase;
        for (int k = 0; k < kc; k++) {
            var b0 = FloatVector.fromMemorySegment(SPECIES, pB, bOff, NATIVE);
            var b1 = FloatVector.fromMemorySegment(SPECIES, pB, bOff + STRIDE_VEC_BYTES_F32, NATIVE);
            var a0 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff));
            c00 = a0.fma(b0, c00); c01 = a0.fma(b1, c01);

            aOff += STEP_A_ROW_BYTES_F32;
            bOff += STEP_B_ROW_BYTES_F32;
        }

        long row = ((long) ci * N + cj) * 4L;
        storeMaskedRow_Float(C, row, c00, c01, mask0, mask1, lanes1 > 0, isFirstKBlock);
    }

    /**
     * Microkernel 8x12 Float - 2-way unrolled with ping-pong register reuse and VectorMask tail store.
     */
    public static void microKernel8x12_Float(MemorySegment pA, long aBase, MemorySegment pB, long bBase,
                                             MemorySegment C, int ci, int cj, int kc, int N, boolean isFirstKBlock) {
        var z = FloatVector.zero(SPECIES);
        FloatVector c00 = z, c01 = z, c10 = z, c11 = z, c20 = z, c21 = z, c30 = z, c31 = z;
        FloatVector c40 = z, c41 = z, c50 = z, c51 = z, c60 = z, c61 = z, c70 = z, c71 = z;

        final long stride = STRIDE_VEC_BYTES_F32;
        long aOff = aBase;
        long bOff = bBase;
        int k = 0;
        final int kUnroll = kc - 1;

        for (; k < kUnroll; k += 2) {
            var b0 = FloatVector.fromMemorySegment(SPECIES, pB, bOff, NATIVE);
            var b1 = FloatVector.fromMemorySegment(SPECIES, pB, bOff + stride, NATIVE);

            var a0 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff));
            var a1 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff + 4L));
            c00 = a0.fma(b0, c00); c01 = a0.fma(b1, c01);
            c10 = a1.fma(b0, c10); c11 = a1.fma(b1, c11);

            a0 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff + 8L));
            a1 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff + 12L));
            c20 = a0.fma(b0, c20); c21 = a0.fma(b1, c21);
            c30 = a1.fma(b0, c30); c31 = a1.fma(b1, c31);

            a0 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff + 16L));
            a1 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff + 20L));
            c40 = a0.fma(b0, c40); c41 = a0.fma(b1, c41);
            c50 = a1.fma(b0, c50); c51 = a1.fma(b1, c51);

            a0 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff + 24L));
            a1 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff + 28L));
            c60 = a0.fma(b0, c60); c61 = a0.fma(b1, c61);
            c70 = a1.fma(b0, c70); c71 = a1.fma(b1, c71);

            long aOff1 = aOff + 32L;
            long bOff1 = bOff + 64L;
            b0 = FloatVector.fromMemorySegment(SPECIES, pB, bOff1, NATIVE);
            b1 = FloatVector.fromMemorySegment(SPECIES, pB, bOff1 + stride, NATIVE);

            a0 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff1));
            a1 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff1 + 4L));
            c00 = a0.fma(b0, c00); c01 = a0.fma(b1, c01);
            c10 = a1.fma(b0, c10); c11 = a1.fma(b1, c11);

            a0 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff1 + 8L));
            a1 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff1 + 12L));
            c20 = a0.fma(b0, c20); c21 = a0.fma(b1, c21);
            c30 = a1.fma(b0, c30); c31 = a1.fma(b1, c31);

            a0 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff1 + 16L));
            a1 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff1 + 20L));
            c40 = a0.fma(b0, c40); c41 = a0.fma(b1, c41);
            c50 = a1.fma(b0, c50); c51 = a1.fma(b1, c51);

            a0 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff1 + 24L));
            a1 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff1 + 28L));
            c60 = a0.fma(b0, c60); c61 = a0.fma(b1, c61);
            c70 = a1.fma(b0, c70); c71 = a1.fma(b1, c71);

            aOff = aOff1 + 32L;
            bOff = bOff1 + 64L;
        }

        for (; k < kc; k++) {
            var b0 = FloatVector.fromMemorySegment(SPECIES, pB, bOff, NATIVE);
            var b1 = FloatVector.fromMemorySegment(SPECIES, pB, bOff + stride, NATIVE);

            var a0 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff));
            var a1 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff + 4L));
            c00 = a0.fma(b0, c00); c01 = a0.fma(b1, c01);
            c10 = a1.fma(b0, c10); c11 = a1.fma(b1, c11);

            a0 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff + 8L));
            a1 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff + 12L));
            c20 = a0.fma(b0, c20); c21 = a0.fma(b1, c21);
            c30 = a1.fma(b0, c30); c31 = a1.fma(b1, c31);

            a0 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff + 16L));
            a1 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff + 20L));
            c40 = a0.fma(b0, c40); c41 = a0.fma(b1, c41);
            c50 = a1.fma(b0, c50); c51 = a1.fma(b1, c51);

            a0 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff + 24L));
            a1 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff + 28L));
            c60 = a0.fma(b0, c60); c61 = a0.fma(b1, c61);
            c70 = a1.fma(b0, c70); c71 = a1.fma(b1, c71);

            aOff += 32L;
            bOff += 64L;
        }

        final VectorMask<Float> mask0 = SPECIES.indexInRange(0, Math.min(VL_F32, 12));
        final int rem = Math.max(0, 12 - VL_F32);
        final VectorMask<Float> mask1 = SPECIES.indexInRange(0, rem);
        final boolean hasRem = rem > 0;
        final long cStride = (long) N * 4L;
        long row = ((long) ci * N + cj) * 4L;

        storeMaskedRow_Float(C, row, c00, c01, mask0, mask1, hasRem, isFirstKBlock); row += cStride;
        storeMaskedRow_Float(C, row, c10, c11, mask0, mask1, hasRem, isFirstKBlock); row += cStride;
        storeMaskedRow_Float(C, row, c20, c21, mask0, mask1, hasRem, isFirstKBlock); row += cStride;
        storeMaskedRow_Float(C, row, c30, c31, mask0, mask1, hasRem, isFirstKBlock); row += cStride;
        storeMaskedRow_Float(C, row, c40, c41, mask0, mask1, hasRem, isFirstKBlock); row += cStride;
        storeMaskedRow_Float(C, row, c50, c51, mask0, mask1, hasRem, isFirstKBlock); row += cStride;
        storeMaskedRow_Float(C, row, c60, c61, mask0, mask1, hasRem, isFirstKBlock); row += cStride;
        storeMaskedRow_Float(C, row, c70, c71, mask0, mask1, hasRem, isFirstKBlock);
    }

    /**
     * Zero-allocation scalar fallback micro-kernel (eliminates new float[mr * nr] heap allocation).
     */
    public static void microKernelScalar_Float(MemorySegment pA, long aBase, int rOff,
                                               MemorySegment pB, long bBase, MemorySegment C,
                                               int ci, int cj, int kc, int N, int mr, int nr,
                                               int MR_dim, int NR_dim, boolean isFirstKBlock) {
        final long aStepBytes = (long) MR_dim * 4L;
        final long bStepBytes = (long) NR_dim * 4L;

        for (int r = 0; r < mr; r++) {
            long aRowBase = aBase + (long) (rOff + r) * 4L;
            long cRowBase = ((long) (ci + r) * N + cj) * 4L;
            for (int c = 0; c < nr; c++) {
                long cAddr = cRowBase + (long) c * 4L;
                float sum = isFirstKBlock ? 0.0f : C.get(VF, cAddr);
                long aOff = aRowBase;
                long bOff = bBase + (long) c * 4L;
                for (int k = 0; k < kc; k++) {
                    sum = Math.fma(pA.get(VF, aOff), pB.get(VF, bOff), sum);
                    aOff += aStepBytes;
                    bOff += bStepBytes;
                }
                C.set(VF, cAddr, sum);
            }
        }
    }

    // =========================================================================
    // 2. DOUBLE (FP64) MACRO & MICRO KERNELS
    // =========================================================================

    public static void gebpMacroKernel_Arm_Double(MemorySegment pA, MemorySegment pB, MemorySegment C,
                                                  int rowStart, int mc, int jc, int nc, int kc, int p,
                                                  boolean isFirstKBlock) {
        int nrPanels = (nc + NR_DB - 1) / NR_DB;
        int fullIPanels = mc / MR;
        int tailRows = mc % MR;

        for (int jp = 0; jp < nrPanels; jp++) {
            int jr = jp * NR_DB;
            int actualNR = Math.min(NR_DB, nc - jr);
            long bBase = (long) jp * NR_DB * kc * 8L;
            boolean fullNR = (actualNR == NR_DB);

            for (int ip = 0; ip < fullIPanels; ip++) {
                long aBase = (long) ip * MR * kc * 8L;
                int ci = rowStart + ip * MR;
                int cj = jc + jr;
                if (fullNR) {
                    microKernel6x16_Double(pA, aBase, pB, bBase, C, ci, cj, kc, p, isFirstKBlock);
                } else {
                    microKernel6x16Masked_Double(pA, aBase, pB, bBase, C, ci, cj, kc, p, actualNR, isFirstKBlock);
                }
            }

            if (tailRows > 0) {
                long aBase = (long) fullIPanels * MR * kc * 8L;
                int ci = rowStart + fullIPanels * MR;
                int cj = jc + jr;
                int rOff = 0;
                while (rOff + 2 <= tailRows) {
                    if (fullNR) {
                        microKernel2x16_Double(pA, aBase, rOff, pB, bBase, C, ci + rOff, cj, kc, p, isFirstKBlock);
                    } else {
                        microKernel2x16Masked_Double(pA, aBase, rOff, pB, bBase, C, ci + rOff, cj, kc, p, actualNR, isFirstKBlock);
                    }
                    rOff += 2;
                }
                if (rOff < tailRows) {
                    if (fullNR) {
                        microKernel1x16_Double(pA, aBase, rOff, pB, bBase, C, ci + rOff, cj, kc, p, isFirstKBlock);
                    } else {
                        microKernel1x16Masked_Double(pA, aBase, rOff, pB, bBase, C, ci + rOff, cj, kc, p, actualNR, isFirstKBlock);
                    }
                }
            }
        }
    }

    public static void gebpMacroKernel_Aarch_Double(MemorySegment pA, MemorySegment pB, MemorySegment C,
                                                    int rowStart, int mc, int jc, int nc, int kc, int p,
                                                    boolean isFirstKBlock) {
        int nrPanels = (nc + 11) / 12;
        int fullIPanels = mc / 8;
        int tailRows = mc % 8;

        for (int jp = 0; jp < nrPanels; jp++) {
            int jr = jp * 12;
            int actualNR = Math.min(12, nc - jr);
            long bBase = (long) jp * 16 * kc * 8L;
            if (actualNR == 12) {
                for (int ip = 0; ip < fullIPanels; ip++) {
                    microKernel8x12_Double(pA, (long) ip * 8 * kc * 8L, pB, bBase, C, rowStart + ip * 8, jc + jr, kc, p, isFirstKBlock);
                }
                if (tailRows > 0) {
                    microKernelScalar_Double(pA, (long) fullIPanels * 8 * kc * 8L, 0, pB, bBase, C, rowStart + fullIPanels * 8, jc + jr, kc, p, tailRows, 12, 8, 16, isFirstKBlock);
                }
            } else {
                for (int ip = 0; ip < fullIPanels; ip++) {
                    microKernelScalar_Double(pA, (long) ip * 8 * kc * 8L, 0, pB, bBase, C, rowStart + ip * 8, jc + jr, kc, p, 8, actualNR, 8, 16, isFirstKBlock);
                }
                if (tailRows > 0) {
                    microKernelScalar_Double(pA, (long) fullIPanels * 8 * kc * 8L, 0, pB, bBase, C, rowStart + fullIPanels * 8, jc + jr, kc, p, tailRows, actualNR, 8, 16, isFirstKBlock);
                }
            }
        }
    }

    /**
     * Microkernel 6x(2*V) Double - 2-way unrolled with 16 live vector locals.
     */
    public static void microKernel6x16_Double(MemorySegment pA, long aBase, MemorySegment pB, long bBase,
                                              MemorySegment C, int ci, int cj, int kc, int N, boolean isFirstKBlock) {
        final long cStride = (long) N * 8L;
        final long row0 = ((long) ci * N + cj) * 8L;
        final long row1 = row0 + cStride;
        final long row2 = row1 + cStride;
        final long row3 = row2 + cStride;
        final long row4 = row3 + cStride;
        final long row5 = row4 + cStride;

        DoubleVector c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51;
        if (isFirstKBlock) {
            var z = DoubleVector.zero(SPECIESDB);
            c00 = z; c01 = z;
            c10 = z; c11 = z;
            c20 = z; c21 = z;
            c30 = z; c31 = z;
            c40 = z; c41 = z;
            c50 = z; c51 = z;
        } else {
            c00 = DoubleVector.fromMemorySegment(SPECIESDB, C, row0, NATIVE);
            c01 = DoubleVector.fromMemorySegment(SPECIESDB, C, row0 + STRIDE_VEC_BYTES_F64, NATIVE);
            c10 = DoubleVector.fromMemorySegment(SPECIESDB, C, row1, NATIVE);
            c11 = DoubleVector.fromMemorySegment(SPECIESDB, C, row1 + STRIDE_VEC_BYTES_F64, NATIVE);
            c20 = DoubleVector.fromMemorySegment(SPECIESDB, C, row2, NATIVE);
            c21 = DoubleVector.fromMemorySegment(SPECIESDB, C, row2 + STRIDE_VEC_BYTES_F64, NATIVE);
            c30 = DoubleVector.fromMemorySegment(SPECIESDB, C, row3, NATIVE);
            c31 = DoubleVector.fromMemorySegment(SPECIESDB, C, row3 + STRIDE_VEC_BYTES_F64, NATIVE);
            c40 = DoubleVector.fromMemorySegment(SPECIESDB, C, row4, NATIVE);
            c41 = DoubleVector.fromMemorySegment(SPECIESDB, C, row4 + STRIDE_VEC_BYTES_F64, NATIVE);
            c50 = DoubleVector.fromMemorySegment(SPECIESDB, C, row5, NATIVE);
            c51 = DoubleVector.fromMemorySegment(SPECIESDB, C, row5 + STRIDE_VEC_BYTES_F64, NATIVE);
        }

        long aOff = aBase;
        long bOff = bBase;
        int k = 0;
        final int kUnroll = kc - 1;

        for (; k < kUnroll; k += 2) {
            // Iteration 0
            var b0 = DoubleVector.fromMemorySegment(SPECIESDB, pB, bOff, NATIVE);
            var b1 = DoubleVector.fromMemorySegment(SPECIESDB, pB, bOff + STRIDE_VEC_BYTES_F64, NATIVE);

            var a0 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff));
            var a1 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff + 8L));
            c00 = a0.fma(b0, c00); c01 = a0.fma(b1, c01);
            c10 = a1.fma(b0, c10); c11 = a1.fma(b1, c11);

            a0 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff + 16L));
            a1 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff + 24L));
            c20 = a0.fma(b0, c20); c21 = a0.fma(b1, c21);
            c30 = a1.fma(b0, c30); c31 = a1.fma(b1, c31);

            a0 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff + 32L));
            a1 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff + 40L));
            c40 = a0.fma(b0, c40); c41 = a0.fma(b1, c41);
            c50 = a1.fma(b0, c50); c51 = a1.fma(b1, c51);

            // Iteration 1
            long aOff1 = aOff + STEP_A_ROW_BYTES_F64;
            long bOff1 = bOff + STEP_B_ROW_BYTES_F64;
            b0 = DoubleVector.fromMemorySegment(SPECIESDB, pB, bOff1, NATIVE);
            b1 = DoubleVector.fromMemorySegment(SPECIESDB, pB, bOff1 + STRIDE_VEC_BYTES_F64, NATIVE);

            a0 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff1));
            a1 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff1 + 8L));
            c00 = a0.fma(b0, c00); c01 = a0.fma(b1, c01);
            c10 = a1.fma(b0, c10); c11 = a1.fma(b1, c11);

            a0 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff1 + 16L));
            a1 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff1 + 24L));
            c20 = a0.fma(b0, c20); c21 = a0.fma(b1, c21);
            c30 = a1.fma(b0, c30); c31 = a1.fma(b1, c31);

            a0 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff1 + 32L));
            a1 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff1 + 40L));
            c40 = a0.fma(b0, c40); c41 = a0.fma(b1, c41);
            c50 = a1.fma(b0, c50); c51 = a1.fma(b1, c51);

            aOff = aOff1 + STEP_A_ROW_BYTES_F64;
            bOff = bOff1 + STEP_B_ROW_BYTES_F64;
        }

        for (; k < kc; k++) {
            var b0 = DoubleVector.fromMemorySegment(SPECIESDB, pB, bOff, NATIVE);
            var b1 = DoubleVector.fromMemorySegment(SPECIESDB, pB, bOff + STRIDE_VEC_BYTES_F64, NATIVE);

            var a0 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff));
            var a1 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff + 8L));
            c00 = a0.fma(b0, c00); c01 = a0.fma(b1, c01);
            c10 = a1.fma(b0, c10); c11 = a1.fma(b1, c11);

            a0 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff + 16L));
            a1 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff + 24L));
            c20 = a0.fma(b0, c20); c21 = a0.fma(b1, c21);
            c30 = a1.fma(b0, c30); c31 = a1.fma(b1, c31);

            a0 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff + 32L));
            a1 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff + 40L));
            c40 = a0.fma(b0, c40); c41 = a0.fma(b1, c41);
            c50 = a1.fma(b0, c50); c51 = a1.fma(b1, c51);

            aOff += STEP_A_ROW_BYTES_F64;
            bOff += STEP_B_ROW_BYTES_F64;
        }

        c00.intoMemorySegment(C, row0, NATIVE); c01.intoMemorySegment(C, row0 + STRIDE_VEC_BYTES_F64, NATIVE);
        c10.intoMemorySegment(C, row1, NATIVE); c11.intoMemorySegment(C, row1 + STRIDE_VEC_BYTES_F64, NATIVE);
        c20.intoMemorySegment(C, row2, NATIVE); c21.intoMemorySegment(C, row2 + STRIDE_VEC_BYTES_F64, NATIVE);
        c30.intoMemorySegment(C, row3, NATIVE); c31.intoMemorySegment(C, row3 + STRIDE_VEC_BYTES_F64, NATIVE);
        c40.intoMemorySegment(C, row4, NATIVE); c41.intoMemorySegment(C, row4 + STRIDE_VEC_BYTES_F64, NATIVE);
        c50.intoMemorySegment(C, row5, NATIVE); c51.intoMemorySegment(C, row5 + STRIDE_VEC_BYTES_F64, NATIVE);
    }

    private static void microKernel6x16Masked_Double(MemorySegment pA, long aBase, MemorySegment pB, long bBase,
                                                     MemorySegment C, int ci, int cj, int kc, int N, int actualNR,
                                                     boolean isFirstKBlock) {
        final int vLen = VL_F64;
        final int lanes0 = Math.min(vLen, actualNR);
        final int lanes1 = Math.max(0, actualNR - vLen);
        final VectorMask<Double> mask0 = SPECIESDB.indexInRange(0, lanes0);
        final VectorMask<Double> mask1 = SPECIESDB.indexInRange(0, lanes1);

        var z = DoubleVector.zero(SPECIESDB);
        DoubleVector c00 = z, c01 = z, c10 = z, c11 = z, c20 = z, c21 = z;
        DoubleVector c30 = z, c31 = z, c40 = z, c41 = z, c50 = z, c51 = z;

        long aOff = aBase;
        long bOff = bBase;
        for (int k = 0; k < kc; k++) {
            var b0 = DoubleVector.fromMemorySegment(SPECIESDB, pB, bOff, NATIVE);
            var b1 = DoubleVector.fromMemorySegment(SPECIESDB, pB, bOff + STRIDE_VEC_BYTES_F64, NATIVE);

            var a0 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff));
            var a1 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff + 8L));
            c00 = a0.fma(b0, c00); c01 = a0.fma(b1, c01);
            c10 = a1.fma(b0, c10); c11 = a1.fma(b1, c11);

            a0 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff + 16L));
            a1 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff + 24L));
            c20 = a0.fma(b0, c20); c21 = a0.fma(b1, c21);
            c30 = a1.fma(b0, c30); c31 = a1.fma(b1, c31);

            a0 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff + 32L));
            a1 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff + 40L));
            c40 = a0.fma(b0, c40); c41 = a0.fma(b1, c41);
            c50 = a1.fma(b0, c50); c51 = a1.fma(b1, c51);

            aOff += STEP_A_ROW_BYTES_F64;
            bOff += STEP_B_ROW_BYTES_F64;
        }

        final long cStride = (long) N * 8L;
        long row0 = ((long) ci * N + cj) * 8L;
        storeMaskedRow_Double(C, row0, c00, c01, mask0, mask1, lanes1 > 0, isFirstKBlock);
        row0 += cStride;
        storeMaskedRow_Double(C, row0, c10, c11, mask0, mask1, lanes1 > 0, isFirstKBlock);
        row0 += cStride;
        storeMaskedRow_Double(C, row0, c20, c21, mask0, mask1, lanes1 > 0, isFirstKBlock);
        row0 += cStride;
        storeMaskedRow_Double(C, row0, c30, c31, mask0, mask1, lanes1 > 0, isFirstKBlock);
        row0 += cStride;
        storeMaskedRow_Double(C, row0, c40, c41, mask0, mask1, lanes1 > 0, isFirstKBlock);
        row0 += cStride;
        storeMaskedRow_Double(C, row0, c50, c51, mask0, mask1, lanes1 > 0, isFirstKBlock);
    }

    private static void storeMaskedRow_Double(MemorySegment C, long rowOff, DoubleVector v0, DoubleVector v1,
                                              VectorMask<Double> mask0, VectorMask<Double> mask1,
                                              boolean hasSecondVec, boolean isFirstKBlock) {
        if (!isFirstKBlock) {
            v0 = v0.add(DoubleVector.fromMemorySegment(SPECIESDB, C, rowOff, NATIVE, mask0));
        }
        v0.intoMemorySegment(C, rowOff, NATIVE, mask0);
        if (hasSecondVec) {
            long off1 = rowOff + STRIDE_VEC_BYTES_F64;
            if (!isFirstKBlock) {
                v1 = v1.add(DoubleVector.fromMemorySegment(SPECIESDB, C, off1, NATIVE, mask1));
            }
            v1.intoMemorySegment(C, off1, NATIVE, mask1);
        }
    }

    /**
     * Microkernel 2x(2*V) Double - 2-way unrolled.
     */
    public static void microKernel2x16_Double(MemorySegment pA, long aBase, int rOff,
                                              MemorySegment pB, long bBase,
                                              MemorySegment C, int ci, int cj,
                                              int kc, int N, boolean isFirstKBlock) {
        final long row0 = ((long) ci * N + cj) * 8L;
        final long row1 = row0 + (long) N * 8L;

        DoubleVector c00, c01, c10, c11;
        if (isFirstKBlock) {
            var z = DoubleVector.zero(SPECIESDB);
            c00 = z; c01 = z; c10 = z; c11 = z;
        } else {
            c00 = DoubleVector.fromMemorySegment(SPECIESDB, C, row0, NATIVE);
            c01 = DoubleVector.fromMemorySegment(SPECIESDB, C, row0 + STRIDE_VEC_BYTES_F64, NATIVE);
            c10 = DoubleVector.fromMemorySegment(SPECIESDB, C, row1, NATIVE);
            c11 = DoubleVector.fromMemorySegment(SPECIESDB, C, row1 + STRIDE_VEC_BYTES_F64, NATIVE);
        }

        long aOff = aBase + (long) rOff * 8L;
        long bOff = bBase;
        int k = 0;
        final int kUnroll = kc - 1;

        for (; k < kUnroll; k += 2) {
            var b0 = DoubleVector.fromMemorySegment(SPECIESDB, pB, bOff, NATIVE);
            var b1 = DoubleVector.fromMemorySegment(SPECIESDB, pB, bOff + STRIDE_VEC_BYTES_F64, NATIVE);
            var a0 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff));
            var a1 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff + 8L));
            c00 = a0.fma(b0, c00); c01 = a0.fma(b1, c01);
            c10 = a1.fma(b0, c10); c11 = a1.fma(b1, c11);

            long aOff1 = aOff + STEP_A_ROW_BYTES_F64;
            long bOff1 = bOff + STEP_B_ROW_BYTES_F64;
            b0 = DoubleVector.fromMemorySegment(SPECIESDB, pB, bOff1, NATIVE);
            b1 = DoubleVector.fromMemorySegment(SPECIESDB, pB, bOff1 + STRIDE_VEC_BYTES_F64, NATIVE);
            a0 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff1));
            a1 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff1 + 8L));
            c00 = a0.fma(b0, c00); c01 = a0.fma(b1, c01);
            c10 = a1.fma(b0, c10); c11 = a1.fma(b1, c11);

            aOff = aOff1 + STEP_A_ROW_BYTES_F64;
            bOff = bOff1 + STEP_B_ROW_BYTES_F64;
        }

        for (; k < kc; k++) {
            var b0 = DoubleVector.fromMemorySegment(SPECIESDB, pB, bOff, NATIVE);
            var b1 = DoubleVector.fromMemorySegment(SPECIESDB, pB, bOff + STRIDE_VEC_BYTES_F64, NATIVE);
            var a0 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff));
            var a1 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff + 8L));
            c00 = a0.fma(b0, c00); c01 = a0.fma(b1, c01);
            c10 = a1.fma(b0, c10); c11 = a1.fma(b1, c11);

            aOff += STEP_A_ROW_BYTES_F64;
            bOff += STEP_B_ROW_BYTES_F64;
        }

        c00.intoMemorySegment(C, row0, NATIVE); c01.intoMemorySegment(C, row0 + STRIDE_VEC_BYTES_F64, NATIVE);
        c10.intoMemorySegment(C, row1, NATIVE); c11.intoMemorySegment(C, row1 + STRIDE_VEC_BYTES_F64, NATIVE);
    }

    private static void microKernel2x16Masked_Double(MemorySegment pA, long aBase, int rOff,
                                                     MemorySegment pB, long bBase,
                                                     MemorySegment C, int ci, int cj,
                                                     int kc, int N, int actualNR, boolean isFirstKBlock) {
        final int vLen = VL_F64;
        final int lanes0 = Math.min(vLen, actualNR);
        final int lanes1 = Math.max(0, actualNR - vLen);
        final VectorMask<Double> mask0 = SPECIESDB.indexInRange(0, lanes0);
        final VectorMask<Double> mask1 = SPECIESDB.indexInRange(0, lanes1);

        var z = DoubleVector.zero(SPECIESDB);
        DoubleVector c00 = z, c01 = z, c10 = z, c11 = z;

        long aOff = aBase + (long) rOff * 8L;
        long bOff = bBase;
        for (int k = 0; k < kc; k++) {
            var b0 = DoubleVector.fromMemorySegment(SPECIESDB, pB, bOff, NATIVE);
            var b1 = DoubleVector.fromMemorySegment(SPECIESDB, pB, bOff + STRIDE_VEC_BYTES_F64, NATIVE);
            var a0 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff));
            var a1 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff + 8L));
            c00 = a0.fma(b0, c00); c01 = a0.fma(b1, c01);
            c10 = a1.fma(b0, c10); c11 = a1.fma(b1, c11);

            aOff += STEP_A_ROW_BYTES_F64;
            bOff += STEP_B_ROW_BYTES_F64;
        }

        long row0 = ((long) ci * N + cj) * 8L;
        storeMaskedRow_Double(C, row0, c00, c01, mask0, mask1, lanes1 > 0, isFirstKBlock);
        storeMaskedRow_Double(C, row0 + (long) N * 8L, c10, c11, mask0, mask1, lanes1 > 0, isFirstKBlock);
    }

    /**
     * Microkernel 1x(2*V) Double - 2-way unrolled.
     */
    public static void microKernel1x16_Double(MemorySegment pA, long aBase, int rOff,
                                              MemorySegment pB, long bBase,
                                              MemorySegment C, int ci, int cj,
                                              int kc, int N, boolean isFirstKBlock) {
        final long row = ((long) ci * N + cj) * 8L;

        DoubleVector c00, c01;
        if (isFirstKBlock) {
            var z = DoubleVector.zero(SPECIESDB);
            c00 = z; c01 = z;
        } else {
            c00 = DoubleVector.fromMemorySegment(SPECIESDB, C, row, NATIVE);
            c01 = DoubleVector.fromMemorySegment(SPECIESDB, C, row + STRIDE_VEC_BYTES_F64, NATIVE);
        }

        long aOff = aBase + (long) rOff * 8L;
        long bOff = bBase;
        int k = 0;
        final int kUnroll = kc - 1;

        for (; k < kUnroll; k += 2) {
            var b0 = DoubleVector.fromMemorySegment(SPECIESDB, pB, bOff, NATIVE);
            var b1 = DoubleVector.fromMemorySegment(SPECIESDB, pB, bOff + STRIDE_VEC_BYTES_F64, NATIVE);
            var a0 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff));
            c00 = a0.fma(b0, c00); c01 = a0.fma(b1, c01);

            long aOff1 = aOff + STEP_A_ROW_BYTES_F64;
            long bOff1 = bOff + STEP_B_ROW_BYTES_F64;
            b0 = DoubleVector.fromMemorySegment(SPECIESDB, pB, bOff1, NATIVE);
            b1 = DoubleVector.fromMemorySegment(SPECIESDB, pB, bOff1 + STRIDE_VEC_BYTES_F64, NATIVE);
            a0 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff1));
            c00 = a0.fma(b0, c00); c01 = a0.fma(b1, c01);

            aOff = aOff1 + STEP_A_ROW_BYTES_F64;
            bOff = bOff1 + STEP_B_ROW_BYTES_F64;
        }

        for (; k < kc; k++) {
            var b0 = DoubleVector.fromMemorySegment(SPECIESDB, pB, bOff, NATIVE);
            var b1 = DoubleVector.fromMemorySegment(SPECIESDB, pB, bOff + STRIDE_VEC_BYTES_F64, NATIVE);
            var a0 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff));
            c00 = a0.fma(b0, c00); c01 = a0.fma(b1, c01);

            aOff += STEP_A_ROW_BYTES_F64;
            bOff += STEP_B_ROW_BYTES_F64;
        }

        c00.intoMemorySegment(C, row, NATIVE);
        c01.intoMemorySegment(C, row + STRIDE_VEC_BYTES_F64, NATIVE);
    }

    private static void microKernel1x16Masked_Double(MemorySegment pA, long aBase, int rOff,
                                                     MemorySegment pB, long bBase,
                                                     MemorySegment C, int ci, int cj,
                                                     int kc, int N, int actualNR, boolean isFirstKBlock) {
        final int vLen = VL_F64;
        final int lanes0 = Math.min(vLen, actualNR);
        final int lanes1 = Math.max(0, actualNR - vLen);
        final VectorMask<Double> mask0 = SPECIESDB.indexInRange(0, lanes0);
        final VectorMask<Double> mask1 = SPECIESDB.indexInRange(0, lanes1);

        var z = DoubleVector.zero(SPECIESDB);
        DoubleVector c00 = z, c01 = z;

        long aOff = aBase + (long) rOff * 8L;
        long bOff = bBase;
        for (int k = 0; k < kc; k++) {
            var b0 = DoubleVector.fromMemorySegment(SPECIESDB, pB, bOff, NATIVE);
            var b1 = DoubleVector.fromMemorySegment(SPECIESDB, pB, bOff + STRIDE_VEC_BYTES_F64, NATIVE);
            var a0 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff));
            c00 = a0.fma(b0, c00); c01 = a0.fma(b1, c01);

            aOff += STEP_A_ROW_BYTES_F64;
            bOff += STEP_B_ROW_BYTES_F64;
        }

        long row = ((long) ci * N + cj) * 8L;
        storeMaskedRow_Double(C, row, c00, c01, mask0, mask1, lanes1 > 0, isFirstKBlock);
    }

    /**
     * Microkernel 8x12 Double - 2-way unrolled with VectorMask tail store.
     */
    public static void microKernel8x12_Double(MemorySegment pA, long aBase, MemorySegment pB, long bBase,
                                              MemorySegment C, int ci, int cj, int kc, int N, boolean isFirstKBlock) {
        var z = DoubleVector.zero(SPECIESDB);
        DoubleVector c00 = z, c01 = z, c10 = z, c11 = z, c20 = z, c21 = z, c30 = z, c31 = z;
        DoubleVector c40 = z, c41 = z, c50 = z, c51 = z, c60 = z, c61 = z, c70 = z, c71 = z;

        final long stride = STRIDE_VEC_BYTES_F64;
        long aOff = aBase;
        long bOff = bBase;
        int k = 0;
        final int kUnroll = kc - 1;

        for (; k < kUnroll; k += 2) {
            var b0 = DoubleVector.fromMemorySegment(SPECIESDB, pB, bOff, NATIVE);
            var b1 = DoubleVector.fromMemorySegment(SPECIESDB, pB, bOff + stride, NATIVE);

            var a0 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff));
            var a1 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff + 8L));
            c00 = a0.fma(b0, c00); c01 = a0.fma(b1, c01);
            c10 = a1.fma(b0, c10); c11 = a1.fma(b1, c11);

            a0 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff + 16L));
            a1 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff + 24L));
            c20 = a0.fma(b0, c20); c21 = a0.fma(b1, c21);
            c30 = a1.fma(b0, c30); c31 = a1.fma(b1, c31);

            a0 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff + 32L));
            a1 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff + 40L));
            c40 = a0.fma(b0, c40); c41 = a0.fma(b1, c41);
            c50 = a1.fma(b0, c50); c51 = a1.fma(b1, c51);

            a0 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff + 48L));
            a1 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff + 56L));
            c60 = a0.fma(b0, c60); c61 = a0.fma(b1, c61);
            c70 = a1.fma(b0, c70); c71 = a1.fma(b1, c71);

            long aOff1 = aOff + 64L;
            long bOff1 = bOff + 128L;
            b0 = DoubleVector.fromMemorySegment(SPECIESDB, pB, bOff1, NATIVE);
            b1 = DoubleVector.fromMemorySegment(SPECIESDB, pB, bOff1 + stride, NATIVE);

            a0 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff1));
            a1 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff1 + 8L));
            c00 = a0.fma(b0, c00); c01 = a0.fma(b1, c01);
            c10 = a1.fma(b0, c10); c11 = a1.fma(b1, c11);

            a0 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff1 + 16L));
            a1 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff1 + 24L));
            c20 = a0.fma(b0, c20); c21 = a0.fma(b1, c21);
            c30 = a1.fma(b0, c30); c31 = a1.fma(b1, c31);

            a0 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff1 + 32L));
            a1 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff1 + 40L));
            c40 = a0.fma(b0, c40); c41 = a0.fma(b1, c41);
            c50 = a1.fma(b0, c50); c51 = a1.fma(b1, c51);

            a0 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff1 + 48L));
            a1 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff1 + 56L));
            c60 = a0.fma(b0, c60); c61 = a0.fma(b1, c61);
            c70 = a1.fma(b0, c70); c71 = a1.fma(b1, c71);

            aOff = aOff1 + 64L;
            bOff = bOff1 + 128L;
        }

        for (; k < kc; k++) {
            var b0 = DoubleVector.fromMemorySegment(SPECIESDB, pB, bOff, NATIVE);
            var b1 = DoubleVector.fromMemorySegment(SPECIESDB, pB, bOff + stride, NATIVE);

            var a0 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff));
            var a1 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff + 8L));
            c00 = a0.fma(b0, c00); c01 = a0.fma(b1, c01);
            c10 = a1.fma(b0, c10); c11 = a1.fma(b1, c11);

            a0 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff + 16L));
            a1 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff + 24L));
            c20 = a0.fma(b0, c20); c21 = a0.fma(b1, c21);
            c30 = a1.fma(b0, c30); c31 = a1.fma(b1, c31);

            a0 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff + 32L));
            a1 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff + 40L));
            c40 = a0.fma(b0, c40); c41 = a0.fma(b1, c41);
            c50 = a1.fma(b0, c50); c51 = a1.fma(b1, c51);

            a0 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff + 48L));
            a1 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff + 56L));
            c60 = a0.fma(b0, c60); c61 = a0.fma(b1, c61);
            c70 = a1.fma(b0, c70); c71 = a1.fma(b1, c71);

            aOff += 64L;
            bOff += 128L;
        }

        DoubleVector[] acc0 = {c00, c10, c20, c30, c40, c50, c60, c70};
        DoubleVector[] acc1 = {c01, c11, c21, c31, c41, c51, c61, c71};

        for (int r = 0; r < 8; r++) {
            long row = ((long) (ci + r) * N + cj) * 8L;
            if (isFirstKBlock) {
                acc0[r].intoMemorySegment(C, row, NATIVE);
                for (int lane = 0; lane < 4; lane++) {
                    long idx = (long) (ci + r) * N + cj + 8 + lane;
                    C.setAtIndex(VD, idx, acc1[r].lane(lane));
                }
            } else {
                DoubleVector.fromMemorySegment(SPECIESDB, C, row, NATIVE).add(acc0[r]).intoMemorySegment(C, row, NATIVE);
                for (int lane = 0; lane < 4; lane++) {
                    long idx = (long) (ci + r) * N + cj + 8 + lane;
                    C.setAtIndex(VD, idx, C.getAtIndex(VD, idx) + acc1[r].lane(lane));
                }
            }
        }
    }

    /**
     * Zero-allocation scalar fallback micro-kernel for Double.
     */
    public static void microKernelScalar_Double(MemorySegment pA, long aBase, int rOff,
                                                MemorySegment pB, long bBase, MemorySegment C,
                                                int ci, int cj, int kc, int N, int mr, int nr,
                                                int MR_dim, int NR_dim, boolean isFirstKBlock) {
        final long aStepBytes = (long) MR_dim * 8L;
        final long bStepBytes = (long) NR_dim * 8L;

        for (int r = 0; r < mr; r++) {
            long aRowBase = aBase + (long) (rOff + r) * 8L;
            long cRowBase = ((long) (ci + r) * N + cj) * 8L;
            for (int c = 0; c < nr; c++) {
                long cAddr = cRowBase + (long) c * 8L;
                double sum = isFirstKBlock ? 0.0d : C.get(VD, cAddr);
                long aOff = aRowBase;
                long bOff = bBase + (long) c * 8L;
                for (int k = 0; k < kc; k++) {
                    sum = Math.fma(pA.get(VD, aOff), pB.get(VD, bOff), sum);
                    aOff += aStepBytes;
                    bOff += bStepBytes;
                }
                C.set(VD, cAddr, sum);
            }
        }
    }

    // =========================================================================
    // 3. INT (INT32) MACRO & MICRO KERNELS
    // =========================================================================

    public static void gebpMacroKernel_Arm_Int(MemorySegment pA, MemorySegment pB, MemorySegment C,
                                               int rowStart, int mc, int jc, int nc, int kc, int p,
                                               boolean isFirstKBlock) {
        int nrPanels = (nc + NR_INT - 1) / NR_INT;
        int fullIPanels = mc / MR;
        int tailRows = mc % MR;

        for (int jp = 0; jp < nrPanels; jp++) {
            int jr = jp * NR_INT;
            int actualNR = Math.min(NR_INT, nc - jr);
            long bBase = (long) jp * NR_INT * kc * 4L;
            boolean fullNR = (actualNR == NR_INT);

            for (int ip = 0; ip < fullIPanels; ip++) {
                long aBase = (long) ip * MR * kc * 4L;
                int ci = rowStart + ip * MR;
                int cj = jc + jr;
                if (fullNR) {
                    microKernel6x16_Int(pA, aBase, pB, bBase, C, ci, cj, kc, p, isFirstKBlock);
                } else {
                    microKernel6x16Masked_Int(pA, aBase, pB, bBase, C, ci, cj, kc, p, actualNR, isFirstKBlock);
                }
            }

            if (tailRows > 0) {
                long aBase = (long) fullIPanels * MR * kc * 4L;
                int ci = rowStart + fullIPanels * MR;
                int cj = jc + jr;
                int rOff = 0;
                while (rOff + 2 <= tailRows) {
                    if (fullNR) {
                        microKernel2x16_Int(pA, aBase, rOff, pB, bBase, C, ci + rOff, cj, kc, p, isFirstKBlock);
                    } else {
                        microKernel2x16Masked_Int(pA, aBase, rOff, pB, bBase, C, ci + rOff, cj, kc, p, actualNR, isFirstKBlock);
                    }
                    rOff += 2;
                }
                if (rOff < tailRows) {
                    if (fullNR) {
                        microKernel1x16_Int(pA, aBase, rOff, pB, bBase, C, ci + rOff, cj, kc, p, isFirstKBlock);
                    } else {
                        microKernel1x16Masked_Int(pA, aBase, rOff, pB, bBase, C, ci + rOff, cj, kc, p, actualNR, isFirstKBlock);
                    }
                }
            }
        }
    }

    public static void gebpMacroKernel_Aarch_Int(MemorySegment pA, MemorySegment pB, MemorySegment C,
                                                 int rowStart, int mc, int jc, int nc, int kc, int p,
                                                 boolean isFirstKBlock) {
        int nrPanels = (nc + 11) / 12;
        int fullIPanels = mc / 8;
        int tailRows = mc % 8;

        for (int jp = 0; jp < nrPanels; jp++) {
            int jr = jp * 12;
            int actualNR = Math.min(12, nc - jr);
            long bBase = (long) jp * 16 * kc * 4L;
            if (actualNR == 12) {
                for (int ip = 0; ip < fullIPanels; ip++) {
                    microKernel8x12_Int(pA, (long) ip * 8 * kc * 4L, pB, bBase, C, rowStart + ip * 8, jc + jr, kc, p, isFirstKBlock);
                }
                if (tailRows > 0) {
                    microKernelScalar_Int(pA, (long) fullIPanels * 8 * kc * 4L, 0, pB, bBase, C, rowStart + fullIPanels * 8, jc + jr, kc, p, tailRows, 12, 8, 16, isFirstKBlock);
                }
            } else {
                for (int ip = 0; ip < fullIPanels; ip++) {
                    microKernelScalar_Int(pA, (long) ip * 8 * kc * 4L, 0, pB, bBase, C, rowStart + ip * 8, jc + jr, kc, p, 8, actualNR, 8, 16, isFirstKBlock);
                }
                if (tailRows > 0) {
                    microKernelScalar_Int(pA, (long) fullIPanels * 8 * kc * 4L, 0, pB, bBase, C, rowStart + fullIPanels * 8, jc + jr, kc, p, tailRows, actualNR, 8, 16, isFirstKBlock);
                }
            }
        }
    }

    /**
     * Microkernel 6x(2*V) Int - 2-way unrolled with 16 live vector locals.
     */
    public static void microKernel6x16_Int(MemorySegment pA, long aBase, MemorySegment pB, long bBase,
                                           MemorySegment C, int ci, int cj, int kc, int N, boolean isFirstKBlock) {
        final long cStride = (long) N * 4L;
        final long row0 = ((long) ci * N + cj) * 4L;
        final long row1 = row0 + cStride;
        final long row2 = row1 + cStride;
        final long row3 = row2 + cStride;
        final long row4 = row3 + cStride;
        final long row5 = row4 + cStride;

        IntVector c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51;
        if (isFirstKBlock) {
            var z = IntVector.zero(SPECIESINT);
            c00 = z; c01 = z;
            c10 = z; c11 = z;
            c20 = z; c21 = z;
            c30 = z; c31 = z;
            c40 = z; c41 = z;
            c50 = z; c51 = z;
        } else {
            c00 = IntVector.fromMemorySegment(SPECIESINT, C, row0, NATIVE);
            c01 = IntVector.fromMemorySegment(SPECIESINT, C, row0 + STRIDE_VEC_BYTES_I32, NATIVE);
            c10 = IntVector.fromMemorySegment(SPECIESINT, C, row1, NATIVE);
            c11 = IntVector.fromMemorySegment(SPECIESINT, C, row1 + STRIDE_VEC_BYTES_I32, NATIVE);
            c20 = IntVector.fromMemorySegment(SPECIESINT, C, row2, NATIVE);
            c21 = IntVector.fromMemorySegment(SPECIESINT, C, row2 + STRIDE_VEC_BYTES_I32, NATIVE);
            c30 = IntVector.fromMemorySegment(SPECIESINT, C, row3, NATIVE);
            c31 = IntVector.fromMemorySegment(SPECIESINT, C, row3 + STRIDE_VEC_BYTES_I32, NATIVE);
            c40 = IntVector.fromMemorySegment(SPECIESINT, C, row4, NATIVE);
            c41 = IntVector.fromMemorySegment(SPECIESINT, C, row4 + STRIDE_VEC_BYTES_I32, NATIVE);
            c50 = IntVector.fromMemorySegment(SPECIESINT, C, row5, NATIVE);
            c51 = IntVector.fromMemorySegment(SPECIESINT, C, row5 + STRIDE_VEC_BYTES_I32, NATIVE);
        }

        long aOff = aBase;
        long bOff = bBase;
        int k = 0;
        final int kUnroll = kc - 1;

        for (; k < kUnroll; k += 2) {
            // Iteration 0
            var b0 = IntVector.fromMemorySegment(SPECIESINT, pB, bOff, NATIVE);
            var b1 = IntVector.fromMemorySegment(SPECIESINT, pB, bOff + STRIDE_VEC_BYTES_I32, NATIVE);

            var a0 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff));
            var a1 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff + 4L));
            c00 = c00.add(a0.mul(b0)); c01 = c01.add(a0.mul(b1));
            c10 = c10.add(a1.mul(b0)); c11 = c11.add(a1.mul(b1));

            a0 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff + 8L));
            a1 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff + 12L));
            c20 = c20.add(a0.mul(b0)); c21 = c21.add(a0.mul(b1));
            c30 = c30.add(a1.mul(b0)); c31 = c31.add(a1.mul(b1));

            a0 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff + 16L));
            a1 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff + 20L));
            c40 = c40.add(a0.mul(b0)); c41 = c41.add(a0.mul(b1));
            c50 = c50.add(a1.mul(b0)); c51 = c51.add(a1.mul(b1));

            // Iteration 1
            long aOff1 = aOff + STEP_A_ROW_BYTES_I32;
            long bOff1 = bOff + STEP_B_ROW_BYTES_I32;
            b0 = IntVector.fromMemorySegment(SPECIESINT, pB, bOff1, NATIVE);
            b1 = IntVector.fromMemorySegment(SPECIESINT, pB, bOff1 + STRIDE_VEC_BYTES_I32, NATIVE);

            a0 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff1));
            a1 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff1 + 4L));
            c00 = c00.add(a0.mul(b0)); c01 = c01.add(a0.mul(b1));
            c10 = c10.add(a1.mul(b0)); c11 = c11.add(a1.mul(b1));

            a0 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff1 + 8L));
            a1 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff1 + 12L));
            c20 = c20.add(a0.mul(b0)); c21 = c21.add(a0.mul(b1));
            c30 = c30.add(a1.mul(b0)); c31 = c31.add(a1.mul(b1));

            a0 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff1 + 16L));
            a1 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff1 + 20L));
            c40 = c40.add(a0.mul(b0)); c41 = c41.add(a0.mul(b1));
            c50 = c50.add(a1.mul(b0)); c51 = c51.add(a1.mul(b1));

            aOff = aOff1 + STEP_A_ROW_BYTES_I32;
            bOff = bOff1 + STEP_B_ROW_BYTES_I32;
        }

        for (; k < kc; k++) {
            var b0 = IntVector.fromMemorySegment(SPECIESINT, pB, bOff, NATIVE);
            var b1 = IntVector.fromMemorySegment(SPECIESINT, pB, bOff + STRIDE_VEC_BYTES_I32, NATIVE);

            var a0 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff));
            var a1 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff + 4L));
            c00 = c00.add(a0.mul(b0)); c01 = c01.add(a0.mul(b1));
            c10 = c10.add(a1.mul(b0)); c11 = c11.add(a1.mul(b1));

            a0 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff + 8L));
            a1 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff + 12L));
            c20 = c20.add(a0.mul(b0)); c21 = c21.add(a0.mul(b1));
            c30 = c30.add(a1.mul(b0)); c31 = c31.add(a1.mul(b1));

            a0 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff + 16L));
            a1 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff + 20L));
            c40 = c40.add(a0.mul(b0)); c41 = c41.add(a0.mul(b1));
            c50 = c50.add(a1.mul(b0)); c51 = c51.add(a1.mul(b1));

            aOff += STEP_A_ROW_BYTES_I32;
            bOff += STEP_B_ROW_BYTES_I32;
        }

        c00.intoMemorySegment(C, row0, NATIVE); c01.intoMemorySegment(C, row0 + STRIDE_VEC_BYTES_I32, NATIVE);
        c10.intoMemorySegment(C, row1, NATIVE); c11.intoMemorySegment(C, row1 + STRIDE_VEC_BYTES_I32, NATIVE);
        c20.intoMemorySegment(C, row2, NATIVE); c21.intoMemorySegment(C, row2 + STRIDE_VEC_BYTES_I32, NATIVE);
        c30.intoMemorySegment(C, row3, NATIVE); c31.intoMemorySegment(C, row3 + STRIDE_VEC_BYTES_I32, NATIVE);
        c40.intoMemorySegment(C, row4, NATIVE); c41.intoMemorySegment(C, row4 + STRIDE_VEC_BYTES_I32, NATIVE);
        c50.intoMemorySegment(C, row5, NATIVE); c51.intoMemorySegment(C, row5 + STRIDE_VEC_BYTES_I32, NATIVE);
    }

    private static void microKernel6x16Masked_Int(MemorySegment pA, long aBase, MemorySegment pB, long bBase,
                                                  MemorySegment C, int ci, int cj, int kc, int N, int actualNR,
                                                  boolean isFirstKBlock) {
        final int vLen = VL_I32;
        final int lanes0 = Math.min(vLen, actualNR);
        final int lanes1 = Math.max(0, actualNR - vLen);
        final VectorMask<Integer> mask0 = SPECIESINT.indexInRange(0, lanes0);
        final VectorMask<Integer> mask1 = SPECIESINT.indexInRange(0, lanes1);

        var z = IntVector.zero(SPECIESINT);
        IntVector c00 = z, c01 = z, c10 = z, c11 = z, c20 = z, c21 = z;
        IntVector c30 = z, c31 = z, c40 = z, c41 = z, c50 = z, c51 = z;

        long aOff = aBase;
        long bOff = bBase;
        for (int k = 0; k < kc; k++) {
            var b0 = IntVector.fromMemorySegment(SPECIESINT, pB, bOff, NATIVE);
            var b1 = IntVector.fromMemorySegment(SPECIESINT, pB, bOff + STRIDE_VEC_BYTES_I32, NATIVE);

            var a0 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff));
            var a1 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff + 4L));
            c00 = c00.add(a0.mul(b0)); c01 = c01.add(a0.mul(b1));
            c10 = c10.add(a1.mul(b0)); c11 = c11.add(a1.mul(b1));

            a0 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff + 8L));
            a1 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff + 12L));
            c20 = c20.add(a0.mul(b0)); c21 = c21.add(a0.mul(b1));
            c30 = c30.add(a1.mul(b0)); c31 = c31.add(a1.mul(b1));

            a0 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff + 16L));
            a1 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff + 20L));
            c40 = c40.add(a0.mul(b0)); c41 = c41.add(a0.mul(b1));
            c50 = c50.add(a1.mul(b0)); c51 = c51.add(a1.mul(b1));

            aOff += STEP_A_ROW_BYTES_I32;
            bOff += STEP_B_ROW_BYTES_I32;
        }

        final long cStride = (long) N * 4L;
        long row0 = ((long) ci * N + cj) * 4L;
        storeMaskedRow_Int(C, row0, c00, c01, mask0, mask1, lanes1 > 0, isFirstKBlock);
        row0 += cStride;
        storeMaskedRow_Int(C, row0, c10, c11, mask0, mask1, lanes1 > 0, isFirstKBlock);
        row0 += cStride;
        storeMaskedRow_Int(C, row0, c20, c21, mask0, mask1, lanes1 > 0, isFirstKBlock);
        row0 += cStride;
        storeMaskedRow_Int(C, row0, c30, c31, mask0, mask1, lanes1 > 0, isFirstKBlock);
        row0 += cStride;
        storeMaskedRow_Int(C, row0, c40, c41, mask0, mask1, lanes1 > 0, isFirstKBlock);
        row0 += cStride;
        storeMaskedRow_Int(C, row0, c50, c51, mask0, mask1, lanes1 > 0, isFirstKBlock);
    }

    private static void storeMaskedRow_Int(MemorySegment C, long rowOff, IntVector v0, IntVector v1,
                                           VectorMask<Integer> mask0, VectorMask<Integer> mask1,
                                           boolean hasSecondVec, boolean isFirstKBlock) {
        if (!isFirstKBlock) {
            v0 = v0.add(IntVector.fromMemorySegment(SPECIESINT, C, rowOff, NATIVE, mask0));
        }
        v0.intoMemorySegment(C, rowOff, NATIVE, mask0);
        if (hasSecondVec) {
            long off1 = rowOff + STRIDE_VEC_BYTES_I32;
            if (!isFirstKBlock) {
                v1 = v1.add(IntVector.fromMemorySegment(SPECIESINT, C, off1, NATIVE, mask1));
            }
            v1.intoMemorySegment(C, off1, NATIVE, mask1);
        }
    }

    /**
     * Microkernel 2x(2*V) Int - 2-way unrolled.
     */
    public static void microKernel2x16_Int(MemorySegment pA, long aBase, int rOff,
                                           MemorySegment pB, long bBase,
                                           MemorySegment C, int ci, int cj,
                                           int kc, int N, boolean isFirstKBlock) {
        final long row0 = ((long) ci * N + cj) * 4L;
        final long row1 = row0 + (long) N * 4L;

        IntVector c00, c01, c10, c11;
        if (isFirstKBlock) {
            var z = IntVector.zero(SPECIESINT);
            c00 = z; c01 = z; c10 = z; c11 = z;
        } else {
            c00 = IntVector.fromMemorySegment(SPECIESINT, C, row0, NATIVE);
            c01 = IntVector.fromMemorySegment(SPECIESINT, C, row0 + STRIDE_VEC_BYTES_I32, NATIVE);
            c10 = IntVector.fromMemorySegment(SPECIESINT, C, row1, NATIVE);
            c11 = IntVector.fromMemorySegment(SPECIESINT, C, row1 + STRIDE_VEC_BYTES_I32, NATIVE);
        }

        long aOff = aBase + (long) rOff * 4L;
        long bOff = bBase;
        int k = 0;
        final int kUnroll = kc - 1;

        for (; k < kUnroll; k += 2) {
            var b0 = IntVector.fromMemorySegment(SPECIESINT, pB, bOff, NATIVE);
            var b1 = IntVector.fromMemorySegment(SPECIESINT, pB, bOff + STRIDE_VEC_BYTES_I32, NATIVE);
            var a0 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff));
            var a1 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff + 4L));
            c00 = c00.add(a0.mul(b0)); c01 = c01.add(a0.mul(b1));
            c10 = c10.add(a1.mul(b0)); c11 = c11.add(a1.mul(b1));

            long aOff1 = aOff + STEP_A_ROW_BYTES_I32;
            long bOff1 = bOff + STEP_B_ROW_BYTES_I32;
            b0 = IntVector.fromMemorySegment(SPECIESINT, pB, bOff1, NATIVE);
            b1 = IntVector.fromMemorySegment(SPECIESINT, pB, bOff1 + STRIDE_VEC_BYTES_I32, NATIVE);
            a0 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff1));
            a1 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff1 + 4L));
            c00 = c00.add(a0.mul(b0)); c01 = c01.add(a0.mul(b1));
            c10 = c10.add(a1.mul(b0)); c11 = c11.add(a1.mul(b1));

            aOff = aOff1 + STEP_A_ROW_BYTES_I32;
            bOff = bOff1 + STEP_B_ROW_BYTES_I32;
        }

        for (; k < kc; k++) {
            var b0 = IntVector.fromMemorySegment(SPECIESINT, pB, bOff, NATIVE);
            var b1 = IntVector.fromMemorySegment(SPECIESINT, pB, bOff + STRIDE_VEC_BYTES_I32, NATIVE);
            var a0 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff));
            var a1 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff + 4L));
            c00 = c00.add(a0.mul(b0)); c01 = c01.add(a0.mul(b1));
            c10 = c10.add(a1.mul(b0)); c11 = c11.add(a1.mul(b1));

            aOff += STEP_A_ROW_BYTES_I32;
            bOff += STEP_B_ROW_BYTES_I32;
        }

        c00.intoMemorySegment(C, row0, NATIVE); c01.intoMemorySegment(C, row0 + STRIDE_VEC_BYTES_I32, NATIVE);
        c10.intoMemorySegment(C, row1, NATIVE); c11.intoMemorySegment(C, row1 + STRIDE_VEC_BYTES_I32, NATIVE);
    }

    private static void microKernel2x16Masked_Int(MemorySegment pA, long aBase, int rOff,
                                                  MemorySegment pB, long bBase,
                                                  MemorySegment C, int ci, int cj,
                                                  int kc, int N, int actualNR, boolean isFirstKBlock) {
        final int vLen = VL_I32;
        final int lanes0 = Math.min(vLen, actualNR);
        final int lanes1 = Math.max(0, actualNR - vLen);
        final VectorMask<Integer> mask0 = SPECIESINT.indexInRange(0, lanes0);
        final VectorMask<Integer> mask1 = SPECIESINT.indexInRange(0, lanes1);

        var z = IntVector.zero(SPECIESINT);
        IntVector c00 = z, c01 = z, c10 = z, c11 = z;

        long aOff = aBase + (long) rOff * 4L;
        long bOff = bBase;
        for (int k = 0; k < kc; k++) {
            var b0 = IntVector.fromMemorySegment(SPECIESINT, pB, bOff, NATIVE);
            var b1 = IntVector.fromMemorySegment(SPECIESINT, pB, bOff + STRIDE_VEC_BYTES_I32, NATIVE);
            var a0 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff));
            var a1 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff + 4L));
            c00 = c00.add(a0.mul(b0)); c01 = c01.add(a0.mul(b1));
            c10 = c10.add(a1.mul(b0)); c11 = c11.add(a1.mul(b1));

            aOff += STEP_A_ROW_BYTES_I32;
            bOff += STEP_B_ROW_BYTES_I32;
        }

        long row0 = ((long) ci * N + cj) * 4L;
        storeMaskedRow_Int(C, row0, c00, c01, mask0, mask1, lanes1 > 0, isFirstKBlock);
        storeMaskedRow_Int(C, row0 + (long) N * 4L, c10, c11, mask0, mask1, lanes1 > 0, isFirstKBlock);
    }

    /**
     * Microkernel 1x(2*V) Int - 2-way unrolled.
     */
    public static void microKernel1x16_Int(MemorySegment pA, long aBase, int rOff,
                                           MemorySegment pB, long bBase,
                                           MemorySegment C, int ci, int cj,
                                           int kc, int N, boolean isFirstKBlock) {
        final long row = ((long) ci * N + cj) * 4L;

        IntVector c00, c01;
        if (isFirstKBlock) {
            var z = IntVector.zero(SPECIESINT);
            c00 = z; c01 = z;
        } else {
            c00 = IntVector.fromMemorySegment(SPECIESINT, C, row, NATIVE);
            c01 = IntVector.fromMemorySegment(SPECIESINT, C, row + STRIDE_VEC_BYTES_I32, NATIVE);
        }

        long aOff = aBase + (long) rOff * 4L;
        long bOff = bBase;
        int k = 0;
        final int kUnroll = kc - 1;

        for (; k < kUnroll; k += 2) {
            var b0 = IntVector.fromMemorySegment(SPECIESINT, pB, bOff, NATIVE);
            var b1 = IntVector.fromMemorySegment(SPECIESINT, pB, bOff + STRIDE_VEC_BYTES_I32, NATIVE);
            var a0 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff));
            c00 = c00.add(a0.mul(b0)); c01 = c01.add(a0.mul(b1));

            long aOff1 = aOff + STEP_A_ROW_BYTES_I32;
            long bOff1 = bOff + STEP_B_ROW_BYTES_I32;
            b0 = IntVector.fromMemorySegment(SPECIESINT, pB, bOff1, NATIVE);
            b1 = IntVector.fromMemorySegment(SPECIESINT, pB, bOff1 + STRIDE_VEC_BYTES_I32, NATIVE);
            a0 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff1));
            c00 = c00.add(a0.mul(b0)); c01 = c01.add(a0.mul(b1));

            aOff = aOff1 + STEP_A_ROW_BYTES_I32;
            bOff = bOff1 + STEP_B_ROW_BYTES_I32;
        }

        for (; k < kc; k++) {
            var b0 = IntVector.fromMemorySegment(SPECIESINT, pB, bOff, NATIVE);
            var b1 = IntVector.fromMemorySegment(SPECIESINT, pB, bOff + STRIDE_VEC_BYTES_I32, NATIVE);
            var a0 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff));
            c00 = c00.add(a0.mul(b0)); c01 = c01.add(a0.mul(b1));

            aOff += STEP_A_ROW_BYTES_I32;
            bOff += STEP_B_ROW_BYTES_I32;
        }

        c00.intoMemorySegment(C, row, NATIVE);
        c01.intoMemorySegment(C, row + STRIDE_VEC_BYTES_I32, NATIVE);
    }

    private static void microKernel1x16Masked_Int(MemorySegment pA, long aBase, int rOff,
                                                  MemorySegment pB, long bBase,
                                                  MemorySegment C, int ci, int cj,
                                                  int kc, int N, int actualNR, boolean isFirstKBlock) {
        final int vLen = VL_I32;
        final int lanes0 = Math.min(vLen, actualNR);
        final int lanes1 = Math.max(0, actualNR - vLen);
        final VectorMask<Integer> mask0 = SPECIESINT.indexInRange(0, lanes0);
        final VectorMask<Integer> mask1 = SPECIESINT.indexInRange(0, lanes1);

        var z = IntVector.zero(SPECIESINT);
        IntVector c00 = z, c01 = z;

        long aOff = aBase + (long) rOff * 4L;
        long bOff = bBase;
        for (int k = 0; k < kc; k++) {
            var b0 = IntVector.fromMemorySegment(SPECIESINT, pB, bOff, NATIVE);
            var b1 = IntVector.fromMemorySegment(SPECIESINT, pB, bOff + STRIDE_VEC_BYTES_I32, NATIVE);
            var a0 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff));
            c00 = c00.add(a0.mul(b0)); c01 = c01.add(a0.mul(b1));

            aOff += STEP_A_ROW_BYTES_I32;
            bOff += STEP_B_ROW_BYTES_I32;
        }

        long row = ((long) ci * N + cj) * 4L;
        storeMaskedRow_Int(C, row, c00, c01, mask0, mask1, lanes1 > 0, isFirstKBlock);
    }

    /**
     * Microkernel 8x12 Int - 2-way unrolled with VectorMask tail store.
     */
    public static void microKernel8x12_Int(MemorySegment pA, long aBase, MemorySegment pB, long bBase,
                                           MemorySegment C, int ci, int cj, int kc, int N, boolean isFirstKBlock) {
        var z = IntVector.zero(SPECIESINT);
        IntVector c00 = z, c01 = z, c10 = z, c11 = z, c20 = z, c21 = z, c30 = z, c31 = z;
        IntVector c40 = z, c41 = z, c50 = z, c51 = z, c60 = z, c61 = z, c70 = z, c71 = z;

        final long stride = STRIDE_VEC_BYTES_I32;
        long aOff = aBase;
        long bOff = bBase;
        int k = 0;
        final int kUnroll = kc - 1;

        for (; k < kUnroll; k += 2) {
            var b0 = IntVector.fromMemorySegment(SPECIESINT, pB, bOff, NATIVE);
            var b1 = IntVector.fromMemorySegment(SPECIESINT, pB, bOff + stride, NATIVE);

            var a0 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff));
            var a1 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff + 4L));
            c00 = c00.add(a0.mul(b0)); c01 = c01.add(a0.mul(b1));
            c10 = c10.add(a1.mul(b0)); c11 = c11.add(a1.mul(b1));

            a0 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff + 8L));
            a1 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff + 12L));
            c20 = c20.add(a0.mul(b0)); c21 = c21.add(a0.mul(b1));
            c30 = c30.add(a1.mul(b0)); c31 = c31.add(a1.mul(b1));

            a0 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff + 16L));
            a1 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff + 20L));
            c40 = c40.add(a0.mul(b0)); c41 = c41.add(a0.mul(b1));
            c50 = c50.add(a1.mul(b0)); c51 = c51.add(a1.mul(b1));

            a0 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff + 24L));
            a1 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff + 28L));
            c60 = c60.add(a0.mul(b0)); c61 = c61.add(a0.mul(b1));
            c70 = c70.add(a1.mul(b0)); c71 = c71.add(a1.mul(b1));

            long aOff1 = aOff + 32L;
            long bOff1 = bOff + 64L;
            b0 = IntVector.fromMemorySegment(SPECIESINT, pB, bOff1, NATIVE);
            b1 = IntVector.fromMemorySegment(SPECIESINT, pB, bOff1 + stride, NATIVE);

            a0 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff1));
            a1 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff1 + 4L));
            c00 = c00.add(a0.mul(b0)); c01 = c01.add(a0.mul(b1));
            c10 = c10.add(a1.mul(b0)); c11 = c11.add(a1.mul(b1));

            a0 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff1 + 8L));
            a1 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff1 + 12L));
            c20 = c20.add(a0.mul(b0)); c21 = c21.add(a0.mul(b1));
            c30 = c30.add(a1.mul(b0)); c31 = c31.add(a1.mul(b1));

            a0 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff1 + 16L));
            a1 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff1 + 20L));
            c40 = c40.add(a0.mul(b0)); c41 = c41.add(a0.mul(b1));
            c50 = c50.add(a1.mul(b0)); c51 = c51.add(a1.mul(b1));

            a0 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff1 + 24L));
            a1 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff1 + 28L));
            c60 = c60.add(a0.mul(b0)); c61 = c61.add(a0.mul(b1));
            c70 = c70.add(a1.mul(b0)); c71 = c71.add(a1.mul(b1));

            aOff = aOff1 + 32L;
            bOff = bOff1 + 64L;
        }

        for (; k < kc; k++) {
            var b0 = IntVector.fromMemorySegment(SPECIESINT, pB, bOff, NATIVE);
            var b1 = IntVector.fromMemorySegment(SPECIESINT, pB, bOff + stride, NATIVE);

            var a0 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff));
            var a1 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff + 4L));
            c00 = c00.add(a0.mul(b0)); c01 = c01.add(a0.mul(b1));
            c10 = c10.add(a1.mul(b0)); c11 = c11.add(a1.mul(b1));

            a0 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff + 8L));
            a1 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff + 12L));
            c20 = c20.add(a0.mul(b0)); c21 = c21.add(a0.mul(b1));
            c30 = c30.add(a1.mul(b0)); c31 = c31.add(a1.mul(b1));

            a0 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff + 16L));
            a1 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff + 20L));
            c40 = c40.add(a0.mul(b0)); c41 = c41.add(a0.mul(b1));
            c50 = c50.add(a1.mul(b0)); c51 = c51.add(a1.mul(b1));

            a0 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff + 24L));
            a1 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff + 28L));
            c60 = c60.add(a0.mul(b0)); c61 = c61.add(a0.mul(b1));
            c70 = c70.add(a1.mul(b0)); c71 = c71.add(a1.mul(b1));

            aOff += 32L;
            bOff += 64L;
        }

        final VectorMask<Integer> mask0 = SPECIESINT.indexInRange(0, Math.min(VL_I32, 12));
        final int rem = Math.max(0, 12 - VL_I32);
        final VectorMask<Integer> mask1 = SPECIESINT.indexInRange(0, rem);
        final boolean hasRem = rem > 0;
        final long cStride = (long) N * 4L;
        long row = ((long) ci * N + cj) * 4L;

        storeMaskedRow_Int(C, row, c00, c01, mask0, mask1, hasRem, isFirstKBlock); row += cStride;
        storeMaskedRow_Int(C, row, c10, c11, mask0, mask1, hasRem, isFirstKBlock); row += cStride;
        storeMaskedRow_Int(C, row, c20, c21, mask0, mask1, hasRem, isFirstKBlock); row += cStride;
        storeMaskedRow_Int(C, row, c30, c31, mask0, mask1, hasRem, isFirstKBlock); row += cStride;
        storeMaskedRow_Int(C, row, c40, c41, mask0, mask1, hasRem, isFirstKBlock); row += cStride;
        storeMaskedRow_Int(C, row, c50, c51, mask0, mask1, hasRem, isFirstKBlock); row += cStride;
        storeMaskedRow_Int(C, row, c60, c61, mask0, mask1, hasRem, isFirstKBlock); row += cStride;
        storeMaskedRow_Int(C, row, c70, c71, mask0, mask1, hasRem, isFirstKBlock);
    }

    /**
     * Zero-allocation scalar fallback micro-kernel for Int.
     */
    public static void microKernelScalar_Int(MemorySegment pA, long aBase, int rOff,
                                             MemorySegment pB, long bBase, MemorySegment C,
                                             int ci, int cj, int kc, int N, int mr, int nr,
                                             int MR_dim, int NR_dim, boolean isFirstKBlock) {
        final long aStepBytes = (long) MR_dim * 4L;
        final long bStepBytes = (long) NR_dim * 4L;

        for (int r = 0; r < mr; r++) {
            long aRowBase = aBase + (long) (rOff + r) * 4L;
            long cRowBase = ((long) (ci + r) * N + cj) * 4L;
            for (int c = 0; c < nr; c++) {
                long cAddr = cRowBase + (long) c * 4L;
                int sum = isFirstKBlock ? 0 : C.get(VI, cAddr);
                long aOff = aRowBase;
                long bOff = bBase + (long) c * 4L;
                for (int k = 0; k < kc; k++) {
                    sum += pA.get(VI, aOff) * pB.get(VI, bOff);
                    aOff += aStepBytes;
                    bOff += bStepBytes;
                }
                C.set(VI, cAddr, sum);
            }
        }
    }

    // =========================================================================
    // BACKWARD-COMPATIBILITY ALIASES FOR TESTS AND EXTERNAL CALLERS
    // =========================================================================

    public static void gebpMacroKernel_Float(MemorySegment pA, MemorySegment pB, MemorySegment C,
                                             int rowStart, int mc, int jc, int nc, int kc, int p, boolean isFirstKBlock) {
        gebpMacroKernel_Arm_Float(pA, pB, C, rowStart, mc, jc, nc, kc, p, isFirstKBlock);
    }

    public static void microKernel6x2vec_Float(MemorySegment pA, long aBase, MemorySegment pB, long bBase,
                                               MemorySegment C, int ci, int cj, int kc, int N, boolean isFirstKBlock) {
        microKernel6x16_Float(pA, aBase, pB, bBase, C, ci, cj, kc, N, isFirstKBlock);
    }

    public static void microKernel2x2vec_Float(MemorySegment pA, long aBase, int rOff, MemorySegment pB, long bBase,
                                               MemorySegment C, int ci, int cj, int kc, int N, boolean isFirstKBlock) {
        microKernel2x16_Float(pA, aBase, rOff, pB, bBase, C, ci, cj, kc, N, isFirstKBlock);
    }

    public static void microKernel1x2vec_Float(MemorySegment pA, long aBase, int rOff, MemorySegment pB, long bBase,
                                               MemorySegment C, int ci, int cj, int kc, int N, boolean isFirstKBlock) {
        microKernel1x16_Float(pA, aBase, rOff, pB, bBase, C, ci, cj, kc, N, isFirstKBlock);
    }

    public static void gebpMacroKernel_Double(MemorySegment pA, MemorySegment pB, MemorySegment C,
                                              int rowStart, int mc, int jc, int nc, int kc, int p, boolean isFirstKBlock) {
        gebpMacroKernel_Arm_Double(pA, pB, C, rowStart, mc, jc, nc, kc, p, isFirstKBlock);
    }

    public static void microKernel6x2vec_Double(MemorySegment pA, long aBase, MemorySegment pB, long bBase,
                                                MemorySegment C, int ci, int cj, int kc, int N, boolean isFirstKBlock) {
        microKernel6x16_Double(pA, aBase, pB, bBase, C, ci, cj, kc, N, isFirstKBlock);
    }

    public static void microKernel2x2vec_Double(MemorySegment pA, long aBase, int rOff, MemorySegment pB, long bBase,
                                                MemorySegment C, int ci, int cj, int kc, int N, boolean isFirstKBlock) {
        microKernel2x16_Double(pA, aBase, rOff, pB, bBase, C, ci, cj, kc, N, isFirstKBlock);
    }

    public static void microKernel1x2vec_Double(MemorySegment pA, long aBase, int rOff, MemorySegment pB, long bBase,
                                                MemorySegment C, int ci, int cj, int kc, int N, boolean isFirstKBlock) {
        microKernel1x16_Double(pA, aBase, rOff, pB, bBase, C, ci, cj, kc, N, isFirstKBlock);
    }

    public static void gebpMacroKernel_Int(MemorySegment pA, MemorySegment pB, MemorySegment C,
                                           int rowStart, int mc, int jc, int nc, int kc, int p, boolean isFirstKBlock) {
        gebpMacroKernel_Arm_Int(pA, pB, C, rowStart, mc, jc, nc, kc, p, isFirstKBlock);
    }

    public static void microKernel6x2vec_Int(MemorySegment pA, long aBase, MemorySegment pB, long bBase,
                                             MemorySegment C, int ci, int cj, int kc, int N, boolean isFirstKBlock) {
        microKernel6x16_Int(pA, aBase, pB, bBase, C, ci, cj, kc, N, isFirstKBlock);
    }

    public static void microKernel2x2vec_Int(MemorySegment pA, long aBase, int rOff, MemorySegment pB, long bBase,
                                             MemorySegment C, int ci, int cj, int kc, int N, boolean isFirstKBlock) {
        microKernel2x16_Int(pA, aBase, rOff, pB, bBase, C, ci, cj, kc, N, isFirstKBlock);
    }

    public static void microKernel1x2vec_Int(MemorySegment pA, long aBase, int rOff, MemorySegment pB, long bBase,
                                             MemorySegment C, int ci, int cj, int kc, int N, boolean isFirstKBlock) {
        microKernel1x16_Int(pA, aBase, rOff, pB, bBase, C, ci, cj, kc, N, isFirstKBlock);
    }
}