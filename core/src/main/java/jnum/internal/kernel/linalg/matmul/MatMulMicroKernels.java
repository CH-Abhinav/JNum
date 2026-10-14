package jnum.internal.kernel.linalg.matmul;

import static jnum.internal.Constants.*;

import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorSpecies;
import jdk.incubator.vector.VectorOperators;

/**
 * BLIS Compute Microkernels and GEBP Macro-kernels.
 * High-performance register-blocked kernels with 4-way unrolling.
 */
public final class MatMulMicroKernels {

    private static final ByteOrder NATIVE = NATIVE_ORDER;
    private static final VectorSpecies<Float> SPECIES = SPECIES_F32;
    private static final VectorSpecies<Double> SPECIESDB = SPECIES_F64;
    private static final VectorSpecies<Integer> SPECIESINT = SPECIES_I32;

    private MatMulMicroKernels() {
        throw new AssertionError("Utility class");
    }

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
                    microKernelScalar_Float(pA, aBase, 0, pB, bBase, C, ci, cj, kc, p, MR, actualNR, MR, NR, isFirstKBlock);
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
                        microKernelScalar_Float(pA, aBase, rOff, pB, bBase, C, ci + rOff, cj, kc, p, 2, actualNR, MR, NR, isFirstKBlock);
                    }
                    rOff += 2;
                }
                if (rOff < tailRows) {
                    if (fullNR) {
                        microKernel1x16_Float(pA, aBase, rOff, pB, bBase, C, ci + rOff, cj, kc, p, isFirstKBlock);
                    } else {
                        microKernelScalar_Float(pA, aBase, rOff, pB, bBase, C, ci + rOff, cj, kc, p, 1, actualNR, MR, NR, isFirstKBlock);
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
                microKernelScalar_Float(pA, (long) fullIPanels * 8 * kc * 4L, 0, pB, bBase, C, rowStart, jc + jr, kc, p, mc, actualNR, 8, 16, isFirstKBlock);
            }
        }
    }

    // Microkernel 6x16 Float - 4-way unrolled
    public static void microKernel6x16_Float(MemorySegment pA, long aBase, MemorySegment pB, long bBase,
                                              MemorySegment C, int ci, int cj, int kc, int N, boolean isFirstKBlock) {
        var c00 = FloatVector.zero(SPECIES); var c01 = FloatVector.zero(SPECIES);
        var c10 = FloatVector.zero(SPECIES); var c11 = FloatVector.zero(SPECIES);
        var c20 = FloatVector.zero(SPECIES); var c21 = FloatVector.zero(SPECIES);
        var c30 = FloatVector.zero(SPECIES); var c31 = FloatVector.zero(SPECIES);
        var c40 = FloatVector.zero(SPECIES); var c41 = FloatVector.zero(SPECIES);
        var c50 = FloatVector.zero(SPECIES); var c51 = FloatVector.zero(SPECIES);

        int k = 0;
        for (; k <= kc - 4; k += 4) {
            // k + 0
            long aOff0 = aBase + (long) k * STEP_A_ROW_BYTES_F32;
            long bOff0 = bBase + (long) k * STEP_B_ROW_BYTES_F32;
            var b0_0 = FloatVector.fromMemorySegment(SPECIES, pB, bOff0, NATIVE);
            var b1_0 = FloatVector.fromMemorySegment(SPECIES, pB, bOff0 + STRIDE_VEC_BYTES_F32, NATIVE);

            var a0_0 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff0));
            c00 = a0_0.fma(b0_0, c00); c01 = a0_0.fma(b1_0, c01);
            var a1_0 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff0 + 4L));
            c10 = a1_0.fma(b0_0, c10); c11 = a1_0.fma(b1_0, c11);
            var a2_0 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff0 + 8L));
            c20 = a2_0.fma(b0_0, c20); c21 = a2_0.fma(b1_0, c21);
            var a3_0 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff0 + 12L));
            c30 = a3_0.fma(b0_0, c30); c31 = a3_0.fma(b1_0, c31);
            var a4_0 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff0 + 16L));
            c40 = a4_0.fma(b0_0, c40); c41 = a4_0.fma(b1_0, c41);
            var a5_0 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff0 + 20L));
            c50 = a5_0.fma(b0_0, c50); c51 = a5_0.fma(b1_0, c51);

            // k + 1
            long aOff1 = aOff0 + STEP_A_ROW_BYTES_F32;
            long bOff1 = bOff0 + STEP_B_ROW_BYTES_F32;
            var b0_1 = FloatVector.fromMemorySegment(SPECIES, pB, bOff1, NATIVE);
            var b1_1 = FloatVector.fromMemorySegment(SPECIES, pB, bOff1 + STRIDE_VEC_BYTES_F32, NATIVE);

            var a0_1 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff1));
            c00 = a0_1.fma(b0_1, c00); c01 = a0_1.fma(b1_1, c01);
            var a1_1 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff1 + 4L));
            c10 = a1_1.fma(b0_1, c10); c11 = a1_1.fma(b1_1, c11);
            var a2_1 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff1 + 8L));
            c20 = a2_1.fma(b0_1, c20); c21 = a2_1.fma(b1_1, c21);
            var a3_1 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff1 + 12L));
            c30 = a3_1.fma(b0_1, c30); c31 = a3_1.fma(b1_1, c31);
            var a4_1 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff1 + 16L));
            c40 = a4_1.fma(b0_1, c40); c41 = a4_1.fma(b1_1, c41);
            var a5_1 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff1 + 20L));
            c50 = a5_1.fma(b0_1, c50); c51 = a5_1.fma(b1_1, c51);

            // k + 2
            long aOff2 = aOff1 + STEP_A_ROW_BYTES_F32;
            long bOff2 = bOff1 + STEP_B_ROW_BYTES_F32;
            var b0_2 = FloatVector.fromMemorySegment(SPECIES, pB, bOff2, NATIVE);
            var b1_2 = FloatVector.fromMemorySegment(SPECIES, pB, bOff2 + STRIDE_VEC_BYTES_F32, NATIVE);

            var a0_2 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff2));
            c00 = a0_2.fma(b0_2, c00); c01 = a0_2.fma(b1_2, c01);
            var a1_2 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff2 + 4L));
            c10 = a1_2.fma(b0_2, c10); c11 = a1_2.fma(b1_2, c11);
            var a2_2 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff2 + 8L));
            c20 = a2_2.fma(b0_2, c20); c21 = a2_2.fma(b1_2, c21);
            var a3_2 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff2 + 12L));
            c30 = a3_2.fma(b0_2, c30); c31 = a3_2.fma(b1_2, c31);
            var a4_2 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff2 + 16L));
            c40 = a4_2.fma(b0_2, c40); c41 = a4_2.fma(b1_2, c41);
            var a5_2 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff2 + 20L));
            c50 = a5_2.fma(b0_2, c50); c51 = a5_2.fma(b1_2, c51);

            // k + 3
            long aOff3 = aOff2 + STEP_A_ROW_BYTES_F32;
            long bOff3 = bOff2 + STEP_B_ROW_BYTES_F32;
            var b0_3 = FloatVector.fromMemorySegment(SPECIES, pB, bOff3, NATIVE);
            var b1_3 = FloatVector.fromMemorySegment(SPECIES, pB, bOff3 + STRIDE_VEC_BYTES_F32, NATIVE);

            var a0_3 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff3));
            c00 = a0_3.fma(b0_3, c00); c01 = a0_3.fma(b1_3, c01);
            var a1_3 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff3 + 4L));
            c10 = a1_3.fma(b0_3, c10); c11 = a1_3.fma(b1_3, c11);
            var a2_3 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff3 + 8L));
            c20 = a2_3.fma(b0_3, c20); c21 = a2_3.fma(b1_3, c21);
            var a3_3 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff3 + 12L));
            c30 = a3_3.fma(b0_3, c30); c31 = a3_3.fma(b1_3, c31);
            var a4_3 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff3 + 16L));
            c40 = a4_3.fma(b0_3, c40); c41 = a4_3.fma(b1_3, c41);
            var a5_3 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff3 + 20L));
            c50 = a5_3.fma(b0_3, c50); c51 = a5_3.fma(b1_3, c51);
        }
        for (; k < kc; k++) {
            long aOff = aBase + (long) k * STEP_A_ROW_BYTES_F32;
            long bOff = bBase + (long) k * STEP_B_ROW_BYTES_F32;

            var b0 = FloatVector.fromMemorySegment(SPECIES, pB, bOff, NATIVE);
            var b1 = FloatVector.fromMemorySegment(SPECIES, pB, bOff + STRIDE_VEC_BYTES_F32, NATIVE);

            var a0 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff));
            c00 = a0.fma(b0, c00); c01 = a0.fma(b1, c01);
            var a1 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff + 4L));
            c10 = a1.fma(b0, c10); c11 = a1.fma(b1, c11);
            var a2 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff + 8L));
            c20 = a2.fma(b0, c20); c21 = a2.fma(b1, c21);
            var a3 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff + 12L));
            c30 = a3.fma(b0, c30); c31 = a3.fma(b1, c31);
            var a4 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff + 16L));
            c40 = a4.fma(b0, c40); c41 = a4.fma(b1, c41);
            var a5 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff + 20L));
            c50 = a5.fma(b0, c50); c51 = a5.fma(b1, c51);
        }

        long row0 = ((long) ci * N + cj) * 4L;
        long row1 = ((long)(ci + 1) * N + cj) * 4L;
        long row2 = ((long)(ci + 2) * N + cj) * 4L;
        long row3 = ((long)(ci + 3) * N + cj) * 4L;
        long row4 = ((long)(ci + 4) * N + cj) * 4L;
        long row5 = ((long)(ci + 5) * N + cj) * 4L;

        if (isFirstKBlock) {
            c00.intoMemorySegment(C, row0, NATIVE); c01.intoMemorySegment(C, row0 + STRIDE_VEC_BYTES_F32, NATIVE);
            c10.intoMemorySegment(C, row1, NATIVE); c11.intoMemorySegment(C, row1 + STRIDE_VEC_BYTES_F32, NATIVE);
            c20.intoMemorySegment(C, row2, NATIVE); c21.intoMemorySegment(C, row2 + STRIDE_VEC_BYTES_F32, NATIVE);
            c30.intoMemorySegment(C, row3, NATIVE); c31.intoMemorySegment(C, row3 + STRIDE_VEC_BYTES_F32, NATIVE);
            c40.intoMemorySegment(C, row4, NATIVE); c41.intoMemorySegment(C, row4 + STRIDE_VEC_BYTES_F32, NATIVE);
            c50.intoMemorySegment(C, row5, NATIVE); c51.intoMemorySegment(C, row5 + STRIDE_VEC_BYTES_F32, NATIVE);
        } else {
            FloatVector.fromMemorySegment(SPECIES, C, row0, NATIVE).add(c00).intoMemorySegment(C, row0, NATIVE);
            FloatVector.fromMemorySegment(SPECIES, C, row0 + STRIDE_VEC_BYTES_F32, NATIVE).add(c01).intoMemorySegment(C, row0 + STRIDE_VEC_BYTES_F32, NATIVE);
            FloatVector.fromMemorySegment(SPECIES, C, row1, NATIVE).add(c10).intoMemorySegment(C, row1, NATIVE);
            FloatVector.fromMemorySegment(SPECIES, C, row1 + STRIDE_VEC_BYTES_F32, NATIVE).add(c11).intoMemorySegment(C, row1 + STRIDE_VEC_BYTES_F32, NATIVE);
            FloatVector.fromMemorySegment(SPECIES, C, row2, NATIVE).add(c20).intoMemorySegment(C, row2, NATIVE);
            FloatVector.fromMemorySegment(SPECIES, C, row2 + STRIDE_VEC_BYTES_F32, NATIVE).add(c21).intoMemorySegment(C, row2 + STRIDE_VEC_BYTES_F32, NATIVE);
            FloatVector.fromMemorySegment(SPECIES, C, row3, NATIVE).add(c30).intoMemorySegment(C, row3, NATIVE);
            FloatVector.fromMemorySegment(SPECIES, C, row3 + STRIDE_VEC_BYTES_F32, NATIVE).add(c31).intoMemorySegment(C, row3 + STRIDE_VEC_BYTES_F32, NATIVE);
            FloatVector.fromMemorySegment(SPECIES, C, row4, NATIVE).add(c40).intoMemorySegment(C, row4, NATIVE);
            FloatVector.fromMemorySegment(SPECIES, C, row4 + STRIDE_VEC_BYTES_F32, NATIVE).add(c41).intoMemorySegment(C, row4 + STRIDE_VEC_BYTES_F32, NATIVE);
            FloatVector.fromMemorySegment(SPECIES, C, row5, NATIVE).add(c50).intoMemorySegment(C, row5, NATIVE);
            FloatVector.fromMemorySegment(SPECIES, C, row5 + STRIDE_VEC_BYTES_F32, NATIVE).add(c51).intoMemorySegment(C, row5 + STRIDE_VEC_BYTES_F32, NATIVE);
        }
    }

    // Microkernel 2x16 Float - 4-way unrolled
    public static void microKernel2x16_Float(MemorySegment pA, long aBase, int rOff,
                                              MemorySegment pB, long bBase,
                                              MemorySegment C, int ci, int cj,
                                              int kc, int N, boolean isFirstKBlock) {
        var c00 = FloatVector.zero(SPECIES); var c01 = FloatVector.zero(SPECIES);
        var c10 = FloatVector.zero(SPECIES); var c11 = FloatVector.zero(SPECIES);

        int k = 0;
        for (; k <= kc - 4; k += 4) {
            // k + 0
            long aOff0 = aBase + (long) k * STEP_A_ROW_BYTES_F32 + (long) rOff * 4L;
            long bOff0 = bBase + (long) k * STEP_B_ROW_BYTES_F32;
            var b0_0 = FloatVector.fromMemorySegment(SPECIES, pB, bOff0, NATIVE);
            var b1_0 = FloatVector.fromMemorySegment(SPECIES, pB, bOff0 + STRIDE_VEC_BYTES_F32, NATIVE);
            var a0_0 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff0));
            c00 = a0_0.fma(b0_0, c00); c01 = a0_0.fma(b1_0, c01);
            var a1_0 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff0 + 4L));
            c10 = a1_0.fma(b0_0, c10); c11 = a1_0.fma(b1_0, c11);

            // k + 1
            long aOff1 = aOff0 + STEP_A_ROW_BYTES_F32;
            long bOff1 = bOff0 + STEP_B_ROW_BYTES_F32;
            var b0_1 = FloatVector.fromMemorySegment(SPECIES, pB, bOff1, NATIVE);
            var b1_1 = FloatVector.fromMemorySegment(SPECIES, pB, bOff1 + STRIDE_VEC_BYTES_F32, NATIVE);
            var a0_1 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff1));
            c00 = a0_1.fma(b0_1, c00); c01 = a0_1.fma(b1_1, c01);
            var a1_1 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff1 + 4L));
            c10 = a1_1.fma(b0_1, c10); c11 = a1_1.fma(b1_1, c11);

            // k + 2
            long aOff2 = aOff1 + STEP_A_ROW_BYTES_F32;
            long bOff2 = bOff1 + STEP_B_ROW_BYTES_F32;
            var b0_2 = FloatVector.fromMemorySegment(SPECIES, pB, bOff2, NATIVE);
            var b1_2 = FloatVector.fromMemorySegment(SPECIES, pB, bOff2 + STRIDE_VEC_BYTES_F32, NATIVE);
            var a0_2 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff2));
            c00 = a0_2.fma(b0_2, c00); c01 = a0_2.fma(b1_2, c01);
            var a1_2 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff2 + 4L));
            c10 = a1_2.fma(b0_2, c10); c11 = a1_2.fma(b1_2, c11);

            // k + 3
            long aOff3 = aOff2 + STEP_A_ROW_BYTES_F32;
            long bOff3 = bOff2 + STEP_B_ROW_BYTES_F32;
            var b0_3 = FloatVector.fromMemorySegment(SPECIES, pB, bOff3, NATIVE);
            var b1_3 = FloatVector.fromMemorySegment(SPECIES, pB, bOff3 + STRIDE_VEC_BYTES_F32, NATIVE);
            var a0_3 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff3));
            c00 = a0_3.fma(b0_3, c00); c01 = a0_3.fma(b1_3, c01);
            var a1_3 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff3 + 4L));
            c10 = a1_3.fma(b0_3, c10); c11 = a1_3.fma(b1_3, c11);
        }

        for (; k < kc; k++) {
            long aOff = aBase + (long) k * STEP_A_ROW_BYTES_F32 + (long) rOff * 4L;
            long bOff = bBase + (long) k * STEP_B_ROW_BYTES_F32;

            var b0 = FloatVector.fromMemorySegment(SPECIES, pB, bOff, NATIVE);
            var b1 = FloatVector.fromMemorySegment(SPECIES, pB, bOff + STRIDE_VEC_BYTES_F32, NATIVE);

            var a0 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff));
            c00 = a0.fma(b0, c00); c01 = a0.fma(b1, c01);

            var a1 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff + 4L));
            c10 = a1.fma(b0, c10); c11 = a1.fma(b1, c11);
        }

        long row0 = ((long) ci * N + cj) * 4L;
        long row1 = ((long)(ci + 1) * N + cj) * 4L;

        if (isFirstKBlock) {
            c00.intoMemorySegment(C, row0, NATIVE); c01.intoMemorySegment(C, row0 + STRIDE_VEC_BYTES_F32, NATIVE);
            c10.intoMemorySegment(C, row1, NATIVE); c11.intoMemorySegment(C, row1 + STRIDE_VEC_BYTES_F32, NATIVE);
        } else {
            FloatVector.fromMemorySegment(SPECIES, C, row0, NATIVE).add(c00).intoMemorySegment(C, row0, NATIVE);
            FloatVector.fromMemorySegment(SPECIES, C, row0 + STRIDE_VEC_BYTES_F32, NATIVE).add(c01).intoMemorySegment(C, row0 + STRIDE_VEC_BYTES_F32, NATIVE);
            FloatVector.fromMemorySegment(SPECIES, C, row1, NATIVE).add(c10).intoMemorySegment(C, row1, NATIVE);
            FloatVector.fromMemorySegment(SPECIES, C, row1 + STRIDE_VEC_BYTES_F32, NATIVE).add(c11).intoMemorySegment(C, row1 + STRIDE_VEC_BYTES_F32, NATIVE);
        }
    }

    // Microkernel 1x16 Float - 4-way unrolled
    public static void microKernel1x16_Float(MemorySegment pA, long aBase, int rOff,
                                              MemorySegment pB, long bBase,
                                              MemorySegment C, int ci, int cj,
                                              int kc, int N, boolean isFirstKBlock) {
        var c00 = FloatVector.zero(SPECIES); var c01 = FloatVector.zero(SPECIES);

        int k = 0;
        for (; k <= kc - 4; k += 4) {
            // k + 0
            long aOff0 = aBase + (long) k * STEP_A_ROW_BYTES_F32 + (long) rOff * 4L;
            long bOff0 = bBase + (long) k * STEP_B_ROW_BYTES_F32;
            var b0_0 = FloatVector.fromMemorySegment(SPECIES, pB, bOff0, NATIVE);
            var b1_0 = FloatVector.fromMemorySegment(SPECIES, pB, bOff0 + STRIDE_VEC_BYTES_F32, NATIVE);
            var a_0 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff0));
            c00 = a_0.fma(b0_0, c00); c01 = a_0.fma(b1_0, c01);

            // k + 1
            long aOff1 = aOff0 + STEP_A_ROW_BYTES_F32;
            long bOff1 = bOff0 + STEP_B_ROW_BYTES_F32;
            var b0_1 = FloatVector.fromMemorySegment(SPECIES, pB, bOff1, NATIVE);
            var b1_1 = FloatVector.fromMemorySegment(SPECIES, pB, bOff1 + STRIDE_VEC_BYTES_F32, NATIVE);
            var a_1 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff1));
            c00 = a_1.fma(b0_1, c00); c01 = a_1.fma(b1_1, c01);

            // k + 2
            long aOff2 = aOff1 + STEP_A_ROW_BYTES_F32;
            long bOff2 = bOff1 + STEP_B_ROW_BYTES_F32;
            var b0_2 = FloatVector.fromMemorySegment(SPECIES, pB, bOff2, NATIVE);
            var b1_2 = FloatVector.fromMemorySegment(SPECIES, pB, bOff2 + STRIDE_VEC_BYTES_F32, NATIVE);
            var a_2 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff2));
            c00 = a_2.fma(b0_2, c00); c01 = a_2.fma(b1_2, c01);

            // k + 3
            long aOff3 = aOff2 + STEP_A_ROW_BYTES_F32;
            long bOff3 = bOff2 + STEP_B_ROW_BYTES_F32;
            var b0_3 = FloatVector.fromMemorySegment(SPECIES, pB, bOff3, NATIVE);
            var b1_3 = FloatVector.fromMemorySegment(SPECIES, pB, bOff3 + STRIDE_VEC_BYTES_F32, NATIVE);
            var a_3 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff3));
            c00 = a_3.fma(b0_3, c00); c01 = a_3.fma(b1_3, c01);
        }

        for (; k < kc; k++) {
            long aOff = aBase + (long) k * STEP_A_ROW_BYTES_F32 + (long) rOff * 4L;
            long bOff = bBase + (long) k * STEP_B_ROW_BYTES_F32;

            var b0 = FloatVector.fromMemorySegment(SPECIES, pB, bOff, NATIVE);
            var b1 = FloatVector.fromMemorySegment(SPECIES, pB, bOff + STRIDE_VEC_BYTES_F32, NATIVE);

            var a = FloatVector.broadcast(SPECIES, pA.get(VF, aOff));
            c00 = a.fma(b0, c00); c01 = a.fma(b1, c01);
        }

        long row = ((long) ci * N + cj) * 4L;
        if (isFirstKBlock) {
            c00.intoMemorySegment(C, row, NATIVE);
            c01.intoMemorySegment(C, row + STRIDE_VEC_BYTES_F32, NATIVE);
        } else {
            FloatVector.fromMemorySegment(SPECIES, C, row, NATIVE).add(c00).intoMemorySegment(C, row, NATIVE);
            FloatVector.fromMemorySegment(SPECIES, C, row + STRIDE_VEC_BYTES_F32, NATIVE).add(c01).intoMemorySegment(C, row + STRIDE_VEC_BYTES_F32, NATIVE);
        }
    }

    // Microkernel 8x12 Float - 4-way unrolled
    public static void microKernel8x12_Float(MemorySegment pA, long aBase, MemorySegment pB, long bBase,
                                              MemorySegment C, int ci, int cj, int kc, int N, boolean isFirstKBlock) {
        var c00 = FloatVector.zero(SPECIES); var c01 = FloatVector.zero(SPECIES);
        var c10 = FloatVector.zero(SPECIES); var c11 = FloatVector.zero(SPECIES);
        var c20 = FloatVector.zero(SPECIES); var c21 = FloatVector.zero(SPECIES);
        var c30 = FloatVector.zero(SPECIES); var c31 = FloatVector.zero(SPECIES);
        var c40 = FloatVector.zero(SPECIES); var c41 = FloatVector.zero(SPECIES);
        var c50 = FloatVector.zero(SPECIES); var c51 = FloatVector.zero(SPECIES);
        var c60 = FloatVector.zero(SPECIES); var c61 = FloatVector.zero(SPECIES);
        var c70 = FloatVector.zero(SPECIES); var c71 = FloatVector.zero(SPECIES);

        long stride = (long) SPECIES.length() * 4L;

        int k = 0;
        for (; k <= kc - 4; k += 4) {
            // k + 0
            long aOff0 = aBase + (long)(k + 0) * 8 * 4L;
            long bOff0 = bBase + (long)(k + 0) * 16 * 4L;
            var b0_0 = FloatVector.fromMemorySegment(SPECIES, pB, bOff0, NATIVE);
            var b1_0 = FloatVector.fromMemorySegment(SPECIES, pB, bOff0 + stride, NATIVE);
            var a0_0 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff0 + 0 * 4L));
            c00 = a0_0.fma(b0_0, c00); c01 = a0_0.fma(b1_0, c01);
            var a1_0 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff0 + 1 * 4L));
            c10 = a1_0.fma(b0_0, c10); c11 = a1_0.fma(b1_0, c11);
            var a2_0 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff0 + 2 * 4L));
            c20 = a2_0.fma(b0_0, c20); c21 = a2_0.fma(b1_0, c21);
            var a3_0 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff0 + 3 * 4L));
            c30 = a3_0.fma(b0_0, c30); c31 = a3_0.fma(b1_0, c31);
            var a4_0 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff0 + 4 * 4L));
            c40 = a4_0.fma(b0_0, c40); c41 = a4_0.fma(b1_0, c41);
            var a5_0 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff0 + 5 * 4L));
            c50 = a5_0.fma(b0_0, c50); c51 = a5_0.fma(b1_0, c51);
            var a6_0 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff0 + 6 * 4L));
            c60 = a6_0.fma(b0_0, c60); c61 = a6_0.fma(b1_0, c61);
            var a7_0 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff0 + 7 * 4L));
            c70 = a7_0.fma(b0_0, c70); c71 = a7_0.fma(b1_0, c71);

            // k + 1
            long aOff1 = aBase + (long)(k + 1) * 8 * 4L;
            long bOff1 = bBase + (long)(k + 1) * 16 * 4L;
            var b0_1 = FloatVector.fromMemorySegment(SPECIES, pB, bOff1, NATIVE);
            var b1_1 = FloatVector.fromMemorySegment(SPECIES, pB, bOff1 + stride, NATIVE);
            var a0_1 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff1 + 0 * 4L));
            c00 = a0_1.fma(b0_1, c00); c01 = a0_1.fma(b1_1, c01);
            var a1_1 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff1 + 1 * 4L));
            c10 = a1_1.fma(b0_1, c10); c11 = a1_1.fma(b1_1, c11);
            var a2_1 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff1 + 2 * 4L));
            c20 = a2_1.fma(b0_1, c20); c21 = a2_1.fma(b1_1, c21);
            var a3_1 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff1 + 3 * 4L));
            c30 = a3_1.fma(b0_1, c30); c31 = a3_1.fma(b1_1, c31);
            var a4_1 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff1 + 4 * 4L));
            c40 = a4_1.fma(b0_1, c40); c41 = a4_1.fma(b1_1, c41);
            var a5_1 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff1 + 5 * 4L));
            c50 = a5_1.fma(b0_1, c50); c51 = a5_1.fma(b1_1, c51);
            var a6_1 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff1 + 6 * 4L));
            c60 = a6_1.fma(b0_1, c60); c61 = a6_1.fma(b1_1, c61);
            var a7_1 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff1 + 7 * 4L));
            c70 = a7_1.fma(b0_1, c70); c71 = a7_1.fma(b1_1, c71);

            // k + 2
            long aOff2 = aBase + (long)(k + 2) * 8 * 4L;
            long bOff2 = bBase + (long)(k + 2) * 16 * 4L;
            var b0_2 = FloatVector.fromMemorySegment(SPECIES, pB, bOff2, NATIVE);
            var b1_2 = FloatVector.fromMemorySegment(SPECIES, pB, bOff2 + stride, NATIVE);
            var a0_2 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff2 + 0 * 4L));
            c00 = a0_2.fma(b0_2, c00); c01 = a0_2.fma(b1_2, c01);
            var a1_2 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff2 + 1 * 4L));
            c10 = a1_2.fma(b0_2, c10); c11 = a1_2.fma(b1_2, c11);
            var a2_2 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff2 + 2 * 4L));
            c20 = a2_2.fma(b0_2, c20); c21 = a2_2.fma(b1_2, c21);
            var a3_2 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff2 + 3 * 4L));
            c30 = a3_2.fma(b0_2, c30); c31 = a3_2.fma(b1_2, c31);
            var a4_2 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff2 + 4 * 4L));
            c40 = a4_2.fma(b0_2, c40); c41 = a4_2.fma(b1_2, c41);
            var a5_2 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff2 + 5 * 4L));
            c50 = a5_2.fma(b0_2, c50); c51 = a5_2.fma(b1_2, c51);
            var a6_2 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff2 + 6 * 4L));
            c60 = a6_2.fma(b0_2, c60); c61 = a6_2.fma(b1_2, c61);
            var a7_2 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff2 + 7 * 4L));
            c70 = a7_2.fma(b0_2, c70); c71 = a7_2.fma(b1_2, c71);

            // k + 3
            long aOff3 = aBase + (long)(k + 3) * 8 * 4L;
            long bOff3 = bBase + (long)(k + 3) * 16 * 4L;
            var b0_3 = FloatVector.fromMemorySegment(SPECIES, pB, bOff3, NATIVE);
            var b1_3 = FloatVector.fromMemorySegment(SPECIES, pB, bOff3 + stride, NATIVE);
            var a0_3 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff3 + 0 * 4L));
            c00 = a0_3.fma(b0_3, c00); c01 = a0_3.fma(b1_3, c01);
            var a1_3 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff3 + 1 * 4L));
            c10 = a1_3.fma(b0_3, c10); c11 = a1_3.fma(b1_3, c11);
            var a2_3 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff3 + 2 * 4L));
            c20 = a2_3.fma(b0_3, c20); c21 = a2_3.fma(b1_3, c21);
            var a3_3 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff3 + 3 * 4L));
            c30 = a3_3.fma(b0_3, c30); c31 = a3_3.fma(b1_3, c31);
            var a4_3 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff3 + 4 * 4L));
            c40 = a4_3.fma(b0_3, c40); c41 = a4_3.fma(b1_3, c41);
            var a5_3 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff3 + 5 * 4L));
            c50 = a5_3.fma(b0_3, c50); c51 = a5_3.fma(b1_3, c51);
            var a6_3 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff3 + 6 * 4L));
            c60 = a6_3.fma(b0_3, c60); c61 = a6_3.fma(b1_3, c61);
            var a7_3 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff3 + 7 * 4L));
            c70 = a7_3.fma(b0_3, c70); c71 = a7_3.fma(b1_3, c71);
        }

        for (; k < kc; k++) {
            long aOff = aBase + (long) k * 8 * 4L;
            long bOff = bBase + (long) k * 16 * 4L;

            var b0 = FloatVector.fromMemorySegment(SPECIES, pB, bOff, NATIVE);
            var b1 = FloatVector.fromMemorySegment(SPECIES, pB, bOff + stride, NATIVE);

            var a0 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff + 0 * 4L));
            c00 = a0.fma(b0, c00); c01 = a0.fma(b1, c01);
            var a1 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff + 1 * 4L));
            c10 = a1.fma(b0, c10); c11 = a1.fma(b1, c11);
            var a2 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff + 2 * 4L));
            c20 = a2.fma(b0, c20); c21 = a2.fma(b1, c21);
            var a3 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff + 3 * 4L));
            c30 = a3.fma(b0, c30); c31 = a3.fma(b1, c31);
            var a4 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff + 4 * 4L));
            c40 = a4.fma(b0, c40); c41 = a4.fma(b1, c41);
            var a5 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff + 5 * 4L));
            c50 = a5.fma(b0, c50); c51 = a5.fma(b1, c51);
            var a6 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff + 6 * 4L));
            c60 = a6.fma(b0, c60); c61 = a6.fma(b1, c61);
            var a7 = FloatVector.broadcast(SPECIES, pA.get(VF, aOff + 7 * 4L));
            c70 = a7.fma(b0, c70); c71 = a7.fma(b1, c71);
        }

        FloatVector[] acc0 = {c00, c10, c20, c30, c40, c50, c60, c70};
        FloatVector[] acc1 = {c01, c11, c21, c31, c41, c51, c61, c71};

        for (int r = 0; r < 8; r++) {
            long row = ((long)(ci + r) * N + cj) * 4L;
            if (isFirstKBlock) {
                acc0[r].intoMemorySegment(C, row, NATIVE);
                for (int lane = 0; lane < 4; lane++) {
                    long idx = (long)(ci + r) * N + cj + 8 + lane;
                    C.setAtIndex(VF, idx, acc1[r].lane(lane));
                }
            } else {
                FloatVector.fromMemorySegment(SPECIES, C, row, NATIVE).add(acc0[r]).intoMemorySegment(C, row, NATIVE);
                for (int lane = 0; lane < 4; lane++) {
                    long idx = (long)(ci + r) * N + cj + 8 + lane;
                    C.setAtIndex(VF, idx, C.getAtIndex(VF, idx) + acc1[r].lane(lane));
                }
            }
        }
    }

    public static void microKernelScalar_Float(MemorySegment pA, long aBase, int rOff,
                                                MemorySegment pB, long bBase, MemorySegment C,
                                                int ci, int cj, int kc, int N, int mr, int nr,
                                                int MR_dim, int NR_dim, boolean isFirstKBlock) {
        float[] acc = new float[mr * nr];
        for (int k = 0; k < kc; k++) {
            long aOff = aBase + (long) k * MR_dim * 4L + (long) rOff * 4L;
            long bOff = bBase + (long) k * NR_dim * 4L;
            for (int r = 0; r < mr; r++) {
                float aVal = pA.get(VF, aOff + (long) r * 4L);
                for (int c = 0; c < nr; c++) {
                    acc[r * nr + c] += aVal * pB.get(VF, bOff + (long) c * 4L);
                }
            }
        }
        for (int r = 0; r < mr; r++) {
            for (int c = 0; c < nr; c++) {
                long cIdx = (long)(ci + r) * N + cj + c;
                if (isFirstKBlock) {
                    C.setAtIndex(VF, cIdx, acc[r * nr + c]);
                } else {
                    C.setAtIndex(VF, cIdx, C.getAtIndex(VF, cIdx) + acc[r * nr + c]);
                }
            }
        }
    }
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
                    microKernelScalar_Double(pA, aBase, 0, pB, bBase, C, ci, cj, kc, p, MR, actualNR, MR, NR_DB, isFirstKBlock);
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
                        microKernelScalar_Double(pA, aBase, rOff, pB, bBase, C, ci + rOff, cj, kc, p, 2, actualNR, MR, NR_DB, isFirstKBlock);
                    }
                    rOff += 2;
                }
                if (rOff < tailRows) {
                    if (fullNR) {
                        microKernel1x16_Double(pA, aBase, rOff, pB, bBase, C, ci + rOff, cj, kc, p, isFirstKBlock);
                    } else {
                        microKernelScalar_Double(pA, aBase, rOff, pB, bBase, C, ci + rOff, cj, kc, p, 1, actualNR, MR, NR_DB, isFirstKBlock);
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
                microKernelScalar_Double(pA, (long) fullIPanels * 8 * kc * 8L, 0, pB, bBase, C, rowStart, jc + jr, kc, p, mc, actualNR, 8, 16, isFirstKBlock);
            }
        }
    }

    // Microkernel 6x16 Double - Explicitly 4-way unrolled (Action 4)
    public static void microKernel6x16_Double(MemorySegment pA, long aBase, MemorySegment pB, long bBase,
                                               MemorySegment C, int ci, int cj, int kc, int N, boolean isFirstKBlock) {
        var c00 = DoubleVector.zero(SPECIESDB); var c01 = DoubleVector.zero(SPECIESDB);
        var c10 = DoubleVector.zero(SPECIESDB); var c11 = DoubleVector.zero(SPECIESDB);
        var c20 = DoubleVector.zero(SPECIESDB); var c21 = DoubleVector.zero(SPECIESDB);
        var c30 = DoubleVector.zero(SPECIESDB); var c31 = DoubleVector.zero(SPECIESDB);
        var c40 = DoubleVector.zero(SPECIESDB); var c41 = DoubleVector.zero(SPECIESDB);
        var c50 = DoubleVector.zero(SPECIESDB); var c51 = DoubleVector.zero(SPECIESDB);

        int k = 0;
        for (; k <= kc - 4; k += 4) {
            // k + 0
            long aOff0 = aBase + (long) k * STEP_A_ROW_BYTES_F64;
            long bOff0 = bBase + (long) k * STEP_B_ROW_BYTES_F64;
            var b0_0 = DoubleVector.fromMemorySegment(SPECIESDB, pB, bOff0, NATIVE);
            var b1_0 = DoubleVector.fromMemorySegment(SPECIESDB, pB, bOff0 + STRIDE_VEC_BYTES_F64, NATIVE);

            var a0_0 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff0 + 0L));
            c00 = a0_0.fma(b0_0, c00); c01 = a0_0.fma(b1_0, c01);
            var a1_0 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff0 + 8L));
            c10 = a1_0.fma(b0_0, c10); c11 = a1_0.fma(b1_0, c11);
            var a2_0 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff0 + 16L));
            c20 = a2_0.fma(b0_0, c20); c21 = a2_0.fma(b1_0, c21);
            var a3_0 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff0 + 24L));
            c30 = a3_0.fma(b0_0, c30); c31 = a3_0.fma(b1_0, c31);
            var a4_0 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff0 + 32L));
            c40 = a4_0.fma(b0_0, c40); c41 = a4_0.fma(b1_0, c41);
            var a5_0 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff0 + 40L));
            c50 = a5_0.fma(b0_0, c50); c51 = a5_0.fma(b1_0, c51);

            // k + 1
            long aOff1 = aOff0 + STEP_A_ROW_BYTES_F64;
            long bOff1 = bOff0 + STEP_B_ROW_BYTES_F64;
            var b0_1 = DoubleVector.fromMemorySegment(SPECIESDB, pB, bOff1, NATIVE);
            var b1_1 = DoubleVector.fromMemorySegment(SPECIESDB, pB, bOff1 + STRIDE_VEC_BYTES_F64, NATIVE);

            var a0_1 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff1 + 0L));
            c00 = a0_1.fma(b0_1, c00); c01 = a0_1.fma(b1_1, c01);
            var a1_1 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff1 + 8L));
            c10 = a1_1.fma(b0_1, c10); c11 = a1_1.fma(b1_1, c11);
            var a2_1 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff1 + 16L));
            c20 = a2_1.fma(b0_1, c20); c21 = a2_1.fma(b1_1, c21);
            var a3_1 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff1 + 24L));
            c30 = a3_1.fma(b0_1, c30); c31 = a3_1.fma(b1_1, c31);
            var a4_1 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff1 + 32L));
            c40 = a4_1.fma(b0_1, c40); c41 = a4_1.fma(b1_1, c41);
            var a5_1 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff1 + 40L));
            c50 = a5_1.fma(b0_1, c50); c51 = a5_1.fma(b1_1, c51);

            // k + 2
            long aOff2 = aOff1 + STEP_A_ROW_BYTES_F64;
            long bOff2 = bOff1 + STEP_B_ROW_BYTES_F64;
            var b0_2 = DoubleVector.fromMemorySegment(SPECIESDB, pB, bOff2, NATIVE);
            var b1_2 = DoubleVector.fromMemorySegment(SPECIESDB, pB, bOff2 + STRIDE_VEC_BYTES_F64, NATIVE);

            var a0_2 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff2 + 0L));
            c00 = a0_2.fma(b0_2, c00); c01 = a0_2.fma(b1_2, c01);
            var a1_2 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff2 + 8L));
            c10 = a1_2.fma(b0_2, c10); c11 = a1_2.fma(b1_2, c11);
            var a2_2 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff2 + 16L));
            c20 = a2_2.fma(b0_2, c20); c21 = a2_2.fma(b1_2, c21);
            var a3_2 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff2 + 24L));
            c30 = a3_2.fma(b0_2, c30); c31 = a3_2.fma(b1_2, c31);
            var a4_2 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff2 + 32L));
            c40 = a4_2.fma(b0_2, c40); c41 = a4_2.fma(b1_2, c41);
            var a5_2 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff2 + 40L));
            c50 = a5_2.fma(b0_2, c50); c51 = a5_2.fma(b1_2, c51);

            // k + 3
            long aOff3 = aOff2 + STEP_A_ROW_BYTES_F64;
            long bOff3 = bOff2 + STEP_B_ROW_BYTES_F64;
            var b0_3 = DoubleVector.fromMemorySegment(SPECIESDB, pB, bOff3, NATIVE);
            var b1_3 = DoubleVector.fromMemorySegment(SPECIESDB, pB, bOff3 + STRIDE_VEC_BYTES_F64, NATIVE);

            var a0_3 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff3 + 0L));
            c00 = a0_3.fma(b0_3, c00); c01 = a0_3.fma(b1_3, c01);
            var a1_3 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff3 + 8L));
            c10 = a1_3.fma(b0_3, c10); c11 = a1_3.fma(b1_3, c11);
            var a2_3 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff3 + 16L));
            c20 = a2_3.fma(b0_3, c20); c21 = a2_3.fma(b1_3, c21);
            var a3_3 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff3 + 24L));
            c30 = a3_3.fma(b0_3, c30); c31 = a3_3.fma(b1_3, c31);
            var a4_3 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff3 + 32L));
            c40 = a4_3.fma(b0_3, c40); c41 = a4_3.fma(b1_3, c41);
            var a5_3 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff3 + 40L));
            c50 = a5_3.fma(b0_3, c50); c51 = a5_3.fma(b1_3, c51);
        }

        for (; k < kc; k++) {
            long aOff = aBase + (long) k * STEP_A_ROW_BYTES_F64;
            long bOff = bBase + (long) k * STEP_B_ROW_BYTES_F64;

            var b0 = DoubleVector.fromMemorySegment(SPECIESDB, pB, bOff, NATIVE);
            var b1 = DoubleVector.fromMemorySegment(SPECIESDB, pB, bOff + STRIDE_VEC_BYTES_F64, NATIVE);

            var a0 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff + 0L));
            c00 = a0.fma(b0, c00); c01 = a0.fma(b1, c01);
            var a1 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff + 8L));
            c10 = a1.fma(b0, c10); c11 = a1.fma(b1, c11);
            var a2 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff + 16L));
            c20 = a2.fma(b0, c20); c21 = a2.fma(b1, c21);
            var a3 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff + 24L));
            c30 = a3.fma(b0, c30); c31 = a3.fma(b1, c31);
            var a4 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff + 32L));
            c40 = a4.fma(b0, c40); c41 = a4.fma(b1, c41);
            var a5 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff + 40L));
            c50 = a5.fma(b0, c50); c51 = a5.fma(b1, c51);
        }

        long row0 = ((long) ci * N + cj) * 8L;
        long row1 = ((long)(ci + 1) * N + cj) * 8L;
        long row2 = ((long)(ci + 2) * N + cj) * 8L;
        long row3 = ((long)(ci + 3) * N + cj) * 8L;
        long row4 = ((long)(ci + 4) * N + cj) * 8L;
        long row5 = ((long)(ci + 5) * N + cj) * 8L;

        if (isFirstKBlock) {
            c00.intoMemorySegment(C, row0, NATIVE); c01.intoMemorySegment(C, row0 + STRIDE_VEC_BYTES_F64, NATIVE);
            c10.intoMemorySegment(C, row1, NATIVE); c11.intoMemorySegment(C, row1 + STRIDE_VEC_BYTES_F64, NATIVE);
            c20.intoMemorySegment(C, row2, NATIVE); c21.intoMemorySegment(C, row2 + STRIDE_VEC_BYTES_F64, NATIVE);
            c30.intoMemorySegment(C, row3, NATIVE); c31.intoMemorySegment(C, row3 + STRIDE_VEC_BYTES_F64, NATIVE);
            c40.intoMemorySegment(C, row4, NATIVE); c41.intoMemorySegment(C, row4 + STRIDE_VEC_BYTES_F64, NATIVE);
            c50.intoMemorySegment(C, row5, NATIVE); c51.intoMemorySegment(C, row5 + STRIDE_VEC_BYTES_F64, NATIVE);
        } else {
            DoubleVector.fromMemorySegment(SPECIESDB, C, row0, NATIVE).add(c00).intoMemorySegment(C, row0, NATIVE);
            DoubleVector.fromMemorySegment(SPECIESDB, C, row0 + STRIDE_VEC_BYTES_F64, NATIVE).add(c01).intoMemorySegment(C, row0 + STRIDE_VEC_BYTES_F64, NATIVE);
            DoubleVector.fromMemorySegment(SPECIESDB, C, row1, NATIVE).add(c10).intoMemorySegment(C, row1, NATIVE);
            DoubleVector.fromMemorySegment(SPECIESDB, C, row1 + STRIDE_VEC_BYTES_F64, NATIVE).add(c11).intoMemorySegment(C, row1 + STRIDE_VEC_BYTES_F64, NATIVE);
            DoubleVector.fromMemorySegment(SPECIESDB, C, row2, NATIVE).add(c20).intoMemorySegment(C, row2, NATIVE);
            DoubleVector.fromMemorySegment(SPECIESDB, C, row2 + STRIDE_VEC_BYTES_F64, NATIVE).add(c21).intoMemorySegment(C, row2 + STRIDE_VEC_BYTES_F64, NATIVE);
            DoubleVector.fromMemorySegment(SPECIESDB, C, row3, NATIVE).add(c30).intoMemorySegment(C, row3, NATIVE);
            DoubleVector.fromMemorySegment(SPECIESDB, C, row3 + STRIDE_VEC_BYTES_F64, NATIVE).add(c31).intoMemorySegment(C, row3 + STRIDE_VEC_BYTES_F64, NATIVE);
            DoubleVector.fromMemorySegment(SPECIESDB, C, row4, NATIVE).add(c40).intoMemorySegment(C, row4, NATIVE);
            DoubleVector.fromMemorySegment(SPECIESDB, C, row4 + STRIDE_VEC_BYTES_F64, NATIVE).add(c41).intoMemorySegment(C, row4 + STRIDE_VEC_BYTES_F64, NATIVE);
            DoubleVector.fromMemorySegment(SPECIESDB, C, row5, NATIVE).add(c50).intoMemorySegment(C, row5, NATIVE);
            DoubleVector.fromMemorySegment(SPECIESDB, C, row5 + STRIDE_VEC_BYTES_F64, NATIVE).add(c51).intoMemorySegment(C, row5 + STRIDE_VEC_BYTES_F64, NATIVE);
        }
    }

    // Microkernel 2x16 Double - Explicitly 4-way unrolled (Action 4)
    public static void microKernel2x16_Double(MemorySegment pA, long aBase, int rOff,
                                               MemorySegment pB, long bBase,
                                               MemorySegment C, int ci, int cj,
                                               int kc, int N, boolean isFirstKBlock) {
        var c00 = DoubleVector.zero(SPECIESDB); var c01 = DoubleVector.zero(SPECIESDB);
        var c10 = DoubleVector.zero(SPECIESDB); var c11 = DoubleVector.zero(SPECIESDB);

        int k = 0;
        for (; k <= kc - 4; k += 4) {
            // k + 0
            long aOff0 = aBase + (long) k * STEP_A_ROW_BYTES_F64 + (long) rOff * 8L;
            long bOff0 = bBase + (long) k * STEP_B_ROW_BYTES_F64;
            var b0_0 = DoubleVector.fromMemorySegment(SPECIESDB, pB, bOff0, NATIVE);
            var b1_0 = DoubleVector.fromMemorySegment(SPECIESDB, pB, bOff0 + STRIDE_VEC_BYTES_F64, NATIVE);
            var a0_0 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff0 + 0L));
            c00 = a0_0.fma(b0_0, c00); c01 = a0_0.fma(b1_0, c01);
            var a1_0 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff0 + 8L));
            c10 = a1_0.fma(b0_0, c10); c11 = a1_0.fma(b1_0, c11);

            // k + 1
            long aOff1 = aOff0 + STEP_A_ROW_BYTES_F64;
            long bOff1 = bOff0 + STEP_B_ROW_BYTES_F64;
            var b0_1 = DoubleVector.fromMemorySegment(SPECIESDB, pB, bOff1, NATIVE);
            var b1_1 = DoubleVector.fromMemorySegment(SPECIESDB, pB, bOff1 + STRIDE_VEC_BYTES_F64, NATIVE);
            var a0_1 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff1 + 0L));
            c00 = a0_1.fma(b0_1, c00); c01 = a0_1.fma(b1_1, c01);
            var a1_1 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff1 + 8L));
            c10 = a1_1.fma(b0_1, c10); c11 = a1_1.fma(b1_1, c11);

            // k + 2
            long aOff2 = aOff1 + STEP_A_ROW_BYTES_F64;
            long bOff2 = bOff1 + STEP_B_ROW_BYTES_F64;
            var b0_2 = DoubleVector.fromMemorySegment(SPECIESDB, pB, bOff2, NATIVE);
            var b1_2 = DoubleVector.fromMemorySegment(SPECIESDB, pB, bOff2 + STRIDE_VEC_BYTES_F64, NATIVE);
            var a0_2 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff2 + 0L));
            c00 = a0_2.fma(b0_2, c00); c01 = a0_2.fma(b1_2, c01);
            var a1_2 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff2 + 8L));
            c10 = a1_2.fma(b0_2, c10); c11 = a1_2.fma(b1_2, c11);

            // k + 3
            long aOff3 = aOff2 + STEP_A_ROW_BYTES_F64;
            long bOff3 = bOff2 + STEP_B_ROW_BYTES_F64;
            var b0_3 = DoubleVector.fromMemorySegment(SPECIESDB, pB, bOff3, NATIVE);
            var b1_3 = DoubleVector.fromMemorySegment(SPECIESDB, pB, bOff3 + STRIDE_VEC_BYTES_F64, NATIVE);
            var a0_3 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff3 + 0L));
            c00 = a0_3.fma(b0_3, c00); c01 = a0_3.fma(b1_3, c01);
            var a1_3 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff3 + 8L));
            c10 = a1_3.fma(b0_3, c10); c11 = a1_3.fma(b1_3, c11);
        }

        for (; k < kc; k++) {
            long aOff = aBase + (long) k * STEP_A_ROW_BYTES_F64 + (long) rOff * 8L;
            long bOff = bBase + (long) k * STEP_B_ROW_BYTES_F64;

            var b0 = DoubleVector.fromMemorySegment(SPECIESDB, pB, bOff, NATIVE);
            var b1 = DoubleVector.fromMemorySegment(SPECIESDB, pB, bOff + STRIDE_VEC_BYTES_F64, NATIVE);

            var a0 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff + 0L));
            c00 = a0.fma(b0, c00); c01 = a0.fma(b1, c01);

            var a1 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff + 8L));
            c10 = a1.fma(b0, c10); c11 = a1.fma(b1, c11);
        }

        long row0 = ((long) ci * N + cj) * 8L;
        long row1 = ((long)(ci + 1) * N + cj) * 8L;

        if (isFirstKBlock) {
            c00.intoMemorySegment(C, row0, NATIVE); c01.intoMemorySegment(C, row0 + STRIDE_VEC_BYTES_F64, NATIVE);
            c10.intoMemorySegment(C, row1, NATIVE); c11.intoMemorySegment(C, row1 + STRIDE_VEC_BYTES_F64, NATIVE);
        } else {
            DoubleVector.fromMemorySegment(SPECIESDB, C, row0, NATIVE).add(c00).intoMemorySegment(C, row0, NATIVE);
            DoubleVector.fromMemorySegment(SPECIESDB, C, row0 + STRIDE_VEC_BYTES_F64, NATIVE).add(c01).intoMemorySegment(C, row0 + STRIDE_VEC_BYTES_F64, NATIVE);
            DoubleVector.fromMemorySegment(SPECIESDB, C, row1, NATIVE).add(c10).intoMemorySegment(C, row1, NATIVE);
            DoubleVector.fromMemorySegment(SPECIESDB, C, row1 + STRIDE_VEC_BYTES_F64, NATIVE).add(c11).intoMemorySegment(C, row1 + STRIDE_VEC_BYTES_F64, NATIVE);
        }
    }

    // Microkernel 1x16 Double - Explicitly 4-way unrolled (Action 4)
    public static void microKernel1x16_Double(MemorySegment pA, long aBase, int rOff,
                                               MemorySegment pB, long bBase,
                                               MemorySegment C, int ci, int cj,
                                               int kc, int N, boolean isFirstKBlock) {
        var c00 = DoubleVector.zero(SPECIESDB); var c01 = DoubleVector.zero(SPECIESDB);

        int k = 0;
        for (; k <= kc - 4; k += 4) {
            // k + 0
            long aOff0 = aBase + (long) k * STEP_A_ROW_BYTES_F64 + (long) rOff * 8L;
            long bOff0 = bBase + (long) k * STEP_B_ROW_BYTES_F64;
            var b0_0 = DoubleVector.fromMemorySegment(SPECIESDB, pB, bOff0, NATIVE);
            var b1_0 = DoubleVector.fromMemorySegment(SPECIESDB, pB, bOff0 + STRIDE_VEC_BYTES_F64, NATIVE);
            var a_0 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff0));
            c00 = a_0.fma(b0_0, c00); c01 = a_0.fma(b1_0, c01);

            // k + 1
            long aOff1 = aOff0 + STEP_A_ROW_BYTES_F64;
            long bOff1 = bOff0 + STEP_B_ROW_BYTES_F64;
            var b0_1 = DoubleVector.fromMemorySegment(SPECIESDB, pB, bOff1, NATIVE);
            var b1_1 = DoubleVector.fromMemorySegment(SPECIESDB, pB, bOff1 + STRIDE_VEC_BYTES_F64, NATIVE);
            var a_1 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff1));
            c00 = a_1.fma(b0_1, c00); c01 = a_1.fma(b1_1, c01);

            // k + 2
            long aOff2 = aOff1 + STEP_A_ROW_BYTES_F64;
            long bOff2 = bOff1 + STEP_B_ROW_BYTES_F64;
            var b0_2 = DoubleVector.fromMemorySegment(SPECIESDB, pB, bOff2, NATIVE);
            var b1_2 = DoubleVector.fromMemorySegment(SPECIESDB, pB, bOff2 + STRIDE_VEC_BYTES_F64, NATIVE);
            var a_2 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff2));
            c00 = a_2.fma(b0_2, c00); c01 = a_2.fma(b1_2, c01);

            // k + 3
            long aOff3 = aOff2 + STEP_A_ROW_BYTES_F64;
            long bOff3 = bOff2 + STEP_B_ROW_BYTES_F64;
            var b0_3 = DoubleVector.fromMemorySegment(SPECIESDB, pB, bOff3, NATIVE);
            var b1_3 = DoubleVector.fromMemorySegment(SPECIESDB, pB, bOff3 + STRIDE_VEC_BYTES_F64, NATIVE);
            var a_3 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff3));
            c00 = a_3.fma(b0_3, c00); c01 = a_3.fma(b1_3, c01);
        }

        for (; k < kc; k++) {
            long aOff = aBase + (long) k * STEP_A_ROW_BYTES_F64 + (long) rOff * 8L;
            long bOff = bBase + (long) k * STEP_B_ROW_BYTES_F64;

            var b0 = DoubleVector.fromMemorySegment(SPECIESDB, pB, bOff, NATIVE);
            var b1 = DoubleVector.fromMemorySegment(SPECIESDB, pB, bOff + STRIDE_VEC_BYTES_F64, NATIVE);

            var a = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff));
            c00 = a.fma(b0, c00); c01 = a.fma(b1, c01);
        }

        long row = ((long) ci * N + cj) * 8L;
        if (isFirstKBlock) {
            c00.intoMemorySegment(C, row, NATIVE);
            c01.intoMemorySegment(C, row + STRIDE_VEC_BYTES_F64, NATIVE);
        } else {
            DoubleVector.fromMemorySegment(SPECIESDB, C, row, NATIVE).add(c00).intoMemorySegment(C, row, NATIVE);
            DoubleVector.fromMemorySegment(SPECIESDB, C, row + STRIDE_VEC_BYTES_F64, NATIVE).add(c01).intoMemorySegment(C, row + STRIDE_VEC_BYTES_F64, NATIVE);
        }
    }

    // Microkernel 8x12 Double - Explicitly 4-way unrolled (Action 4)
    public static void microKernel8x12_Double(MemorySegment pA, long aBase, MemorySegment pB, long bBase,
                                               MemorySegment C, int ci, int cj, int kc, int N, boolean isFirstKBlock) {
        var c00 = DoubleVector.zero(SPECIESDB); var c01 = DoubleVector.zero(SPECIESDB);
        var c10 = DoubleVector.zero(SPECIESDB); var c11 = DoubleVector.zero(SPECIESDB);
        var c20 = DoubleVector.zero(SPECIESDB); var c21 = DoubleVector.zero(SPECIESDB);
        var c30 = DoubleVector.zero(SPECIESDB); var c31 = DoubleVector.zero(SPECIESDB);
        var c40 = DoubleVector.zero(SPECIESDB); var c41 = DoubleVector.zero(SPECIESDB);
        var c50 = DoubleVector.zero(SPECIESDB); var c51 = DoubleVector.zero(SPECIESDB);
        var c60 = DoubleVector.zero(SPECIESDB); var c61 = DoubleVector.zero(SPECIESDB);
        var c70 = DoubleVector.zero(SPECIESDB); var c71 = DoubleVector.zero(SPECIESDB);

        long stride = (long) SPECIESDB.length() * 8L;

        int k = 0;
        for (; k <= kc - 4; k += 4) {
            // k + 0
            long aOff0 = aBase + (long)(k + 0) * 8 * 8L;
            long bOff0 = bBase + (long)(k + 0) * 16 * 8L;
            var b0_0 = DoubleVector.fromMemorySegment(SPECIESDB, pB, bOff0, NATIVE);
            var b1_0 = DoubleVector.fromMemorySegment(SPECIESDB, pB, bOff0 + stride, NATIVE);
            var a0_0 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff0 + 0 * 8L));
            c00 = a0_0.fma(b0_0, c00); c01 = a0_0.fma(b1_0, c01);
            var a1_0 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff0 + 1 * 8L));
            c10 = a1_0.fma(b0_0, c10); c11 = a1_0.fma(b1_0, c11);
            var a2_0 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff0 + 2 * 8L));
            c20 = a2_0.fma(b0_0, c20); c21 = a2_0.fma(b1_0, c21);
            var a3_0 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff0 + 3 * 8L));
            c30 = a3_0.fma(b0_0, c30); c31 = a3_0.fma(b1_0, c31);
            var a4_0 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff0 + 4 * 8L));
            c40 = a4_0.fma(b0_0, c40); c41 = a4_0.fma(b1_0, c41);
            var a5_0 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff0 + 5 * 8L));
            c50 = a5_0.fma(b0_0, c50); c51 = a5_0.fma(b1_0, c51);
            var a6_0 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff0 + 6 * 8L));
            c60 = a6_0.fma(b0_0, c60); c61 = a6_0.fma(b1_0, c61);
            var a7_0 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff0 + 7 * 8L));
            c70 = a7_0.fma(b0_0, c70); c71 = a7_0.fma(b1_0, c71);

            // k + 1
            long aOff1 = aBase + (long)(k + 1) * 8 * 8L;
            long bOff1 = bBase + (long)(k + 1) * 16 * 8L;
            var b0_1 = DoubleVector.fromMemorySegment(SPECIESDB, pB, bOff1, NATIVE);
            var b1_1 = DoubleVector.fromMemorySegment(SPECIESDB, pB, bOff1 + stride, NATIVE);
            var a0_1 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff1 + 0 * 8L));
            c00 = a0_1.fma(b0_1, c00); c01 = a0_1.fma(b1_1, c01);
            var a1_1 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff1 + 1 * 8L));
            c10 = a1_1.fma(b0_1, c10); c11 = a1_1.fma(b1_1, c11);
            var a2_1 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff1 + 2 * 8L));
            c20 = a2_1.fma(b0_1, c20); c21 = a2_1.fma(b1_1, c21);
            var a3_1 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff1 + 3 * 8L));
            c30 = a3_1.fma(b0_1, c30); c31 = a3_1.fma(b1_1, c31);
            var a4_1 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff1 + 4 * 8L));
            c40 = a4_1.fma(b0_1, c40); c41 = a4_1.fma(b1_1, c41);
            var a5_1 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff1 + 5 * 8L));
            c50 = a5_1.fma(b0_1, c50); c51 = a5_1.fma(b1_1, c51);
            var a6_1 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff1 + 6 * 8L));
            c60 = a6_1.fma(b0_1, c60); c61 = a6_1.fma(b1_1, c61);
            var a7_1 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff1 + 7 * 8L));
            c70 = a7_1.fma(b0_1, c70); c71 = a7_1.fma(b1_1, c71);

            // k + 2
            long aOff2 = aBase + (long)(k + 2) * 8 * 8L;
            long bOff2 = bBase + (long)(k + 2) * 16 * 8L;
            var b0_2 = DoubleVector.fromMemorySegment(SPECIESDB, pB, bOff2, NATIVE);
            var b1_2 = DoubleVector.fromMemorySegment(SPECIESDB, pB, bOff2 + stride, NATIVE);
            var a0_2 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff2 + 0 * 8L));
            c00 = a0_2.fma(b0_2, c00); c01 = a0_2.fma(b1_2, c01);
            var a1_2 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff2 + 1 * 8L));
            c10 = a1_2.fma(b0_2, c10); c11 = a1_2.fma(b1_2, c11);
            var a2_2 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff2 + 2 * 8L));
            c20 = a2_2.fma(b0_2, c20); c21 = a2_2.fma(b1_2, c21);
            var a3_2 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff2 + 3 * 8L));
            c30 = a3_2.fma(b0_2, c30); c31 = a3_2.fma(b1_2, c31);
            var a4_2 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff2 + 4 * 8L));
            c40 = a4_2.fma(b0_2, c40); c41 = a4_2.fma(b1_2, c41);
            var a5_2 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff2 + 5 * 8L));
            c50 = a5_2.fma(b0_2, c50); c51 = a5_2.fma(b1_2, c51);
            var a6_2 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff2 + 6 * 8L));
            c60 = a6_2.fma(b0_2, c60); c61 = a6_2.fma(b1_2, c61);
            var a7_2 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff2 + 7 * 8L));
            c70 = a7_2.fma(b0_2, c70); c71 = a7_2.fma(b1_2, c71);

            // k + 3
            long aOff3 = aBase + (long)(k + 3) * 8 * 8L;
            long bOff3 = bBase + (long)(k + 3) * 16 * 8L;
            var b0_3 = DoubleVector.fromMemorySegment(SPECIESDB, pB, bOff3, NATIVE);
            var b1_3 = DoubleVector.fromMemorySegment(SPECIESDB, pB, bOff3 + stride, NATIVE);
            var a0_3 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff3 + 0 * 8L));
            c00 = a0_3.fma(b0_3, c00); c01 = a0_3.fma(b1_3, c01);
            var a1_3 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff3 + 1 * 8L));
            c10 = a1_3.fma(b0_3, c10); c11 = a1_3.fma(b1_3, c11);
            var a2_3 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff3 + 2 * 8L));
            c20 = a2_3.fma(b0_3, c20); c21 = a2_3.fma(b1_3, c21);
            var a3_3 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff3 + 3 * 8L));
            c30 = a3_3.fma(b0_3, c30); c31 = a3_3.fma(b1_3, c31);
            var a4_3 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff3 + 4 * 8L));
            c40 = a4_3.fma(b0_3, c40); c41 = a4_3.fma(b1_3, c41);
            var a5_3 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff3 + 5 * 8L));
            c50 = a5_3.fma(b0_3, c50); c51 = a5_3.fma(b1_3, c51);
            var a6_3 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff3 + 6 * 8L));
            c60 = a6_3.fma(b0_3, c60); c61 = a6_3.fma(b1_3, c61);
            var a7_3 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff3 + 7 * 8L));
            c70 = a7_3.fma(b0_3, c70); c71 = a7_3.fma(b1_3, c71);
        }

        for (; k < kc; k++) {
            long aOff = aBase + (long) k * 8 * 8L;
            long bOff = bBase + (long) k * 16 * 8L;

            var b0 = DoubleVector.fromMemorySegment(SPECIESDB, pB, bOff, NATIVE);
            var b1 = DoubleVector.fromMemorySegment(SPECIESDB, pB, bOff + stride, NATIVE);

            var a0 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff + 0 * 8L));
            c00 = a0.fma(b0, c00); c01 = a0.fma(b1, c01);
            var a1 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff + 1 * 8L));
            c10 = a1.fma(b0, c10); c11 = a1.fma(b1, c11);
            var a2 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff + 2 * 8L));
            c20 = a2.fma(b0, c20); c21 = a2.fma(b1, c21);
            var a3 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff + 3 * 8L));
            c30 = a3.fma(b0, c30); c31 = a3.fma(b1, c31);
            var a4 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff + 4 * 8L));
            c40 = a4.fma(b0, c40); c41 = a4.fma(b1, c41);
            var a5 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff + 5 * 8L));
            c50 = a5.fma(b0, c50); c51 = a5.fma(b1, c51);
            var a6 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff + 6 * 8L));
            c60 = a6.fma(b0, c60); c61 = a6.fma(b1, c61);
            var a7 = DoubleVector.broadcast(SPECIESDB, pA.get(VD, aOff + 7 * 8L));
            c70 = a7.fma(b0, c70); c71 = a7.fma(b1, c71);
        }

        DoubleVector[] acc0 = {c00, c10, c20, c30, c40, c50, c60, c70};
        DoubleVector[] acc1 = {c01, c11, c21, c31, c41, c51, c61, c71};

        for (int r = 0; r < 8; r++) {
            long row = ((long)(ci + r) * N + cj) * 8L;
            if (isFirstKBlock) {
                acc0[r].intoMemorySegment(C, row, NATIVE);
                for (int lane = 0; lane < 4; lane++) {
                    long idx = (long)(ci + r) * N + cj + 8 + lane;
                    C.setAtIndex(VD, idx, acc1[r].lane(lane));
                }
            } else {
                DoubleVector.fromMemorySegment(SPECIESDB, C, row, NATIVE).add(acc0[r]).intoMemorySegment(C, row, NATIVE);
                for (int lane = 0; lane < 4; lane++) {
                    long idx = (long)(ci + r) * N + cj + 8 + lane;
                    C.setAtIndex(VD, idx, C.getAtIndex(VD, idx) + acc1[r].lane(lane));
                }
            }
        }
    }

    public static void microKernelScalar_Double(MemorySegment pA, long aBase, int rOff,
                                                 MemorySegment pB, long bBase, MemorySegment C,
                                                 int ci, int cj, int kc, int N, int mr, int nr,
                                                 int MR_dim, int NR_dim, boolean isFirstKBlock) {
        double[] acc = new double[mr * nr];
        for (int k = 0; k < kc; k++) {
            long aOff = aBase + (long) k * MR_dim * 8L + (long) rOff * 8L;
            long bOff = bBase + (long) k * NR_dim * 8L;
            for (int r = 0; r < mr; r++) {
                double aVal = pA.get(VD, aOff + (long) r * 8L);
                for (int c = 0; c < nr; c++) {
                    acc[r * nr + c] += aVal * pB.get(VD, bOff + (long) c * 8L);
                }
            }
        }
        for (int r = 0; r < mr; r++) {
            for (int c = 0; c < nr; c++) {
                long cIdx = (long)(ci + r) * N + cj + c;
                if (isFirstKBlock) {
                    C.setAtIndex(VD, cIdx, acc[r * nr + c]);
                } else {
                    C.setAtIndex(VD, cIdx, C.getAtIndex(VD, cIdx) + acc[r * nr + c]);
                }
            }
        }
    }
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
                    microKernelScalar_Int(pA, aBase, 0, pB, bBase, C, ci, cj, kc, p, MR, actualNR, MR, NR_INT, isFirstKBlock);
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
                        microKernelScalar_Int(pA, aBase, rOff, pB, bBase, C, ci + rOff, cj, kc, p, 2, actualNR, MR, NR_INT, isFirstKBlock);
                    }
                    rOff += 2;
                }
                if (rOff < tailRows) {
                    if (fullNR) {
                        microKernel1x16_Int(pA, aBase, rOff, pB, bBase, C, ci + rOff, cj, kc, p, isFirstKBlock);
                    } else {
                        microKernelScalar_Int(pA, aBase, rOff, pB, bBase, C, ci + rOff, cj, kc, p, 1, actualNR, MR, NR_INT, isFirstKBlock);
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
                microKernelScalar_Int(pA, (long) fullIPanels * 8 * kc * 4L, 0, pB, bBase, C, rowStart, jc + jr, kc, p, mc, actualNR, 8, 16, isFirstKBlock);
            }
        }
    }

    // Microkernel 6x16 Int - 4-way unrolled
    public static void microKernel6x16_Int(MemorySegment pA, long aBase, MemorySegment pB, long bBase,
                                            MemorySegment C, int ci, int cj, int kc, int N, boolean isFirstKBlock) {
        var c00 = IntVector.zero(SPECIESINT); var c01 = IntVector.zero(SPECIESINT);
        var c10 = IntVector.zero(SPECIESINT); var c11 = IntVector.zero(SPECIESINT);
        var c20 = IntVector.zero(SPECIESINT); var c21 = IntVector.zero(SPECIESINT);
        var c30 = IntVector.zero(SPECIESINT); var c31 = IntVector.zero(SPECIESINT);
        var c40 = IntVector.zero(SPECIESINT); var c41 = IntVector.zero(SPECIESINT);
        var c50 = IntVector.zero(SPECIESINT); var c51 = IntVector.zero(SPECIESINT);

        int k = 0;
        for (; k <= kc - 4; k += 4) {
            // k + 0
            long aOff0 = aBase + (long) k * STEP_A_ROW_BYTES_I32;
            long bOff0 = bBase + (long) k * STEP_B_ROW_BYTES_I32;
            var b0_0 = IntVector.fromMemorySegment(SPECIESINT, pB, bOff0, NATIVE);
            var b1_0 = IntVector.fromMemorySegment(SPECIESINT, pB, bOff0 + STRIDE_VEC_BYTES_I32, NATIVE);

            var a0_0 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff0 + 0L));
            c00 = c00.add(a0_0.mul(b0_0)); c01 = c01.add(a0_0.mul(b1_0));
            var a1_0 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff0 + 4L));
            c10 = c10.add(a1_0.mul(b0_0)); c11 = c11.add(a1_0.mul(b1_0));
            var a2_0 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff0 + 8L));
            c20 = c20.add(a2_0.mul(b0_0)); c21 = c21.add(a2_0.mul(b1_0));
            var a3_0 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff0 + 12L));
            c30 = c30.add(a3_0.mul(b0_0)); c31 = c31.add(a3_0.mul(b1_0));
            var a4_0 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff0 + 16L));
            c40 = c40.add(a4_0.mul(b0_0)); c41 = c41.add(a4_0.mul(b1_0));
            var a5_0 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff0 + 20L));
            c50 = c50.add(a5_0.mul(b0_0)); c51 = c51.add(a5_0.mul(b1_0));

            // k + 1
            long aOff1 = aOff0 + STEP_A_ROW_BYTES_I32;
            long bOff1 = bOff0 + STEP_B_ROW_BYTES_I32;
            var b0_1 = IntVector.fromMemorySegment(SPECIESINT, pB, bOff1, NATIVE);
            var b1_1 = IntVector.fromMemorySegment(SPECIESINT, pB, bOff1 + STRIDE_VEC_BYTES_I32, NATIVE);

            var a0_1 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff1 + 0L));
            c00 = c00.add(a0_1.mul(b0_1)); c01 = c01.add(a0_1.mul(b1_1));
            var a1_1 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff1 + 4L));
            c10 = c10.add(a1_1.mul(b0_1)); c11 = c11.add(a1_1.mul(b1_1));
            var a2_1 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff1 + 8L));
            c20 = c20.add(a2_1.mul(b0_1)); c21 = c21.add(a2_1.mul(b1_1));
            var a3_1 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff1 + 12L));
            c30 = c30.add(a3_1.mul(b0_1)); c31 = c31.add(a3_1.mul(b1_1));
            var a4_1 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff1 + 16L));
            c40 = c40.add(a4_1.mul(b0_1)); c41 = c41.add(a4_1.mul(b1_1));
            var a5_1 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff1 + 20L));
            c50 = c50.add(a5_1.mul(b0_1)); c51 = c51.add(a5_1.mul(b1_1));

            // k + 2
            long aOff2 = aOff1 + STEP_A_ROW_BYTES_I32;
            long bOff2 = bOff1 + STEP_B_ROW_BYTES_I32;
            var b0_2 = IntVector.fromMemorySegment(SPECIESINT, pB, bOff2, NATIVE);
            var b1_2 = IntVector.fromMemorySegment(SPECIESINT, pB, bOff2 + STRIDE_VEC_BYTES_I32, NATIVE);

            var a0_2 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff2 + 0L));
            c00 = c00.add(a0_2.mul(b0_2)); c01 = c01.add(a0_2.mul(b1_2));
            var a1_2 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff2 + 4L));
            c10 = c10.add(a1_2.mul(b0_2)); c11 = c11.add(a1_2.mul(b1_2));
            var a2_2 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff2 + 8L));
            c20 = c20.add(a2_2.mul(b0_2)); c21 = c21.add(a2_2.mul(b1_2));
            var a3_2 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff2 + 12L));
            c30 = c30.add(a3_2.mul(b0_2)); c31 = c31.add(a3_2.mul(b1_2));
            var a4_2 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff2 + 16L));
            c40 = c40.add(a4_2.mul(b0_2)); c41 = c41.add(a4_2.mul(b1_2));
            var a5_2 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff2 + 20L));
            c50 = c50.add(a5_2.mul(b0_2)); c51 = c51.add(a5_2.mul(b1_2));

            // k + 3
            long aOff3 = aOff2 + STEP_A_ROW_BYTES_I32;
            long bOff3 = bOff2 + STEP_B_ROW_BYTES_I32;
            var b0_3 = IntVector.fromMemorySegment(SPECIESINT, pB, bOff3, NATIVE);
            var b1_3 = IntVector.fromMemorySegment(SPECIESINT, pB, bOff3 + STRIDE_VEC_BYTES_I32, NATIVE);

            var a0_3 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff3 + 0L));
            c00 = c00.add(a0_3.mul(b0_3)); c01 = c01.add(a0_3.mul(b1_3));
            var a1_3 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff3 + 4L));
            c10 = c10.add(a1_3.mul(b0_3)); c11 = c11.add(a1_3.mul(b1_3));
            var a2_3 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff3 + 8L));
            c20 = c20.add(a2_3.mul(b0_3)); c21 = c21.add(a2_3.mul(b1_3));
            var a3_3 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff3 + 12L));
            c30 = c30.add(a3_3.mul(b0_3)); c31 = c31.add(a3_3.mul(b1_3));
            var a4_3 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff3 + 16L));
            c40 = c40.add(a4_3.mul(b0_3)); c41 = c41.add(a4_3.mul(b1_3));
            var a5_3 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff3 + 20L));
            c50 = c50.add(a5_3.mul(b0_3)); c51 = c51.add(a5_3.mul(b1_3));
        }

        for (; k < kc; k++) {
            long aOff = aBase + (long) k * STEP_A_ROW_BYTES_I32;
            long bOff = bBase + (long) k * STEP_B_ROW_BYTES_I32;

            var b0 = IntVector.fromMemorySegment(SPECIESINT, pB, bOff, NATIVE);
            var b1 = IntVector.fromMemorySegment(SPECIESINT, pB, bOff + STRIDE_VEC_BYTES_I32, NATIVE);

            var a0 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff + 0L));
            c00 = c00.add(a0.mul(b0)); c01 = c01.add(a0.mul(b1));
            var a1 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff + 4L));
            c10 = c10.add(a1.mul(b0)); c11 = c11.add(a1.mul(b1));
            var a2 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff + 8L));
            c20 = c20.add(a2.mul(b0)); c21 = c21.add(a2.mul(b1));
            var a3 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff + 12L));
            c30 = c30.add(a3.mul(b0)); c31 = c31.add(a3.mul(b1));
            var a4 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff + 16L));
            c40 = c40.add(a4.mul(b0)); c41 = c41.add(a4.mul(b1));
            var a5 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff + 20L));
            c50 = c50.add(a5.mul(b0)); c51 = c51.add(a5.mul(b1));
        }

        long row0 = ((long) ci * N + cj) * 4L;
        long row1 = ((long)(ci + 1) * N + cj) * 4L;
        long row2 = ((long)(ci + 2) * N + cj) * 4L;
        long row3 = ((long)(ci + 3) * N + cj) * 4L;
        long row4 = ((long)(ci + 4) * N + cj) * 4L;
        long row5 = ((long)(ci + 5) * N + cj) * 4L;

        if (isFirstKBlock) {
            c00.intoMemorySegment(C, row0, NATIVE); c01.intoMemorySegment(C, row0 + STRIDE_VEC_BYTES_I32, NATIVE);
            c10.intoMemorySegment(C, row1, NATIVE); c11.intoMemorySegment(C, row1 + STRIDE_VEC_BYTES_I32, NATIVE);
            c20.intoMemorySegment(C, row2, NATIVE); c21.intoMemorySegment(C, row2 + STRIDE_VEC_BYTES_I32, NATIVE);
            c30.intoMemorySegment(C, row3, NATIVE); c31.intoMemorySegment(C, row3 + STRIDE_VEC_BYTES_I32, NATIVE);
            c40.intoMemorySegment(C, row4, NATIVE); c41.intoMemorySegment(C, row4 + STRIDE_VEC_BYTES_I32, NATIVE);
            c50.intoMemorySegment(C, row5, NATIVE); c51.intoMemorySegment(C, row5 + STRIDE_VEC_BYTES_I32, NATIVE);
        } else {
            IntVector.fromMemorySegment(SPECIESINT, C, row0, NATIVE).add(c00).intoMemorySegment(C, row0, NATIVE);
            IntVector.fromMemorySegment(SPECIESINT, C, row0 + STRIDE_VEC_BYTES_I32, NATIVE).add(c01).intoMemorySegment(C, row0 + STRIDE_VEC_BYTES_I32, NATIVE);
            IntVector.fromMemorySegment(SPECIESINT, C, row1, NATIVE).add(c10).intoMemorySegment(C, row1, NATIVE);
            IntVector.fromMemorySegment(SPECIESINT, C, row1 + STRIDE_VEC_BYTES_I32, NATIVE).add(c11).intoMemorySegment(C, row1 + STRIDE_VEC_BYTES_I32, NATIVE);
            IntVector.fromMemorySegment(SPECIESINT, C, row2, NATIVE).add(c20).intoMemorySegment(C, row2, NATIVE);
            IntVector.fromMemorySegment(SPECIESINT, C, row2 + STRIDE_VEC_BYTES_I32, NATIVE).add(c21).intoMemorySegment(C, row2 + STRIDE_VEC_BYTES_I32, NATIVE);
            IntVector.fromMemorySegment(SPECIESINT, C, row3, NATIVE).add(c30).intoMemorySegment(C, row3, NATIVE);
            IntVector.fromMemorySegment(SPECIESINT, C, row3 + STRIDE_VEC_BYTES_I32, NATIVE).add(c31).intoMemorySegment(C, row3 + STRIDE_VEC_BYTES_I32, NATIVE);
            IntVector.fromMemorySegment(SPECIESINT, C, row4, NATIVE).add(c40).intoMemorySegment(C, row4, NATIVE);
            IntVector.fromMemorySegment(SPECIESINT, C, row4 + STRIDE_VEC_BYTES_I32, NATIVE).add(c41).intoMemorySegment(C, row4 + STRIDE_VEC_BYTES_I32, NATIVE);
            IntVector.fromMemorySegment(SPECIESINT, C, row5, NATIVE).add(c50).intoMemorySegment(C, row5, NATIVE);
            IntVector.fromMemorySegment(SPECIESINT, C, row5 + STRIDE_VEC_BYTES_I32, NATIVE).add(c51).intoMemorySegment(C, row5 + STRIDE_VEC_BYTES_I32, NATIVE);
        }
    }

    // Microkernel 2x16 Int - 4-way unrolled
    public static void microKernel2x16_Int(MemorySegment pA, long aBase, int rOff,
                                            MemorySegment pB, long bBase,
                                            MemorySegment C, int ci, int cj,
                                            int kc, int N, boolean isFirstKBlock) {
        var c00 = IntVector.zero(SPECIESINT); var c01 = IntVector.zero(SPECIESINT);
        var c10 = IntVector.zero(SPECIESINT); var c11 = IntVector.zero(SPECIESINT);

        int k = 0;
        for (; k <= kc - 4; k += 4) {
            // k + 0
            long aOff0 = aBase + (long) k * STEP_A_ROW_BYTES_I32 + (long) rOff * 4L;
            long bOff0 = bBase + (long) k * STEP_B_ROW_BYTES_I32;
            var b0_0 = IntVector.fromMemorySegment(SPECIESINT, pB, bOff0, NATIVE);
            var b1_0 = IntVector.fromMemorySegment(SPECIESINT, pB, bOff0 + STRIDE_VEC_BYTES_I32, NATIVE);
            var a0_0 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff0 + 0L));
            c00 = c00.add(a0_0.mul(b0_0)); c01 = c01.add(a0_0.mul(b1_0));
            var a1_0 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff0 + 4L));
            c10 = c10.add(a1_0.mul(b0_0)); c11 = c11.add(a1_0.mul(b1_0));

            // k + 1
            long aOff1 = aOff0 + STEP_A_ROW_BYTES_I32;
            long bOff1 = bOff0 + STEP_B_ROW_BYTES_I32;
            var b0_1 = IntVector.fromMemorySegment(SPECIESINT, pB, bOff1, NATIVE);
            var b1_1 = IntVector.fromMemorySegment(SPECIESINT, pB, bOff1 + STRIDE_VEC_BYTES_I32, NATIVE);
            var a0_1 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff1 + 0L));
            c00 = c00.add(a0_1.mul(b0_1)); c01 = c01.add(a0_1.mul(b1_1));
            var a1_1 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff1 + 4L));
            c10 = c10.add(a1_1.mul(b0_1)); c11 = c11.add(a1_1.mul(b1_1));

            // k + 2
            long aOff2 = aOff1 + STEP_A_ROW_BYTES_I32;
            long bOff2 = bOff1 + STEP_B_ROW_BYTES_I32;
            var b0_2 = IntVector.fromMemorySegment(SPECIESINT, pB, bOff2, NATIVE);
            var b1_2 = IntVector.fromMemorySegment(SPECIESINT, pB, bOff2 + STRIDE_VEC_BYTES_I32, NATIVE);
            var a0_2 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff2 + 0L));
            c00 = c00.add(a0_2.mul(b0_2)); c01 = c01.add(a0_2.mul(b1_2));
            var a1_2 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff2 + 4L));
            c10 = c10.add(a1_2.mul(b0_2)); c11 = c11.add(a1_2.mul(b1_2));

            // k + 3
            long aOff3 = aOff2 + STEP_A_ROW_BYTES_I32;
            long bOff3 = bOff2 + STEP_B_ROW_BYTES_I32;
            var b0_3 = IntVector.fromMemorySegment(SPECIESINT, pB, bOff3, NATIVE);
            var b1_3 = IntVector.fromMemorySegment(SPECIESINT, pB, bOff3 + STRIDE_VEC_BYTES_I32, NATIVE);
            var a0_3 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff3 + 0L));
            c00 = c00.add(a0_3.mul(b0_3)); c01 = c01.add(a0_3.mul(b1_3));
            var a1_3 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff3 + 4L));
            c10 = c10.add(a1_3.mul(b0_3)); c11 = c11.add(a1_3.mul(b1_3));
        }

        for (; k < kc; k++) {
            long aOff = aBase + (long) k * STEP_A_ROW_BYTES_I32 + (long) rOff * 4L;
            long bOff = bBase + (long) k * STEP_B_ROW_BYTES_I32;

            var b0 = IntVector.fromMemorySegment(SPECIESINT, pB, bOff, NATIVE);
            var b1 = IntVector.fromMemorySegment(SPECIESINT, pB, bOff + STRIDE_VEC_BYTES_I32, NATIVE);

            var a0 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff + 0L));
            c00 = c00.add(a0.mul(b0)); c01 = c01.add(a0.mul(b1));

            var a1 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff + 4L));
            c10 = c10.add(a1.mul(b0)); c11 = c11.add(a1.mul(b1));
        }

        long row0 = ((long) ci * N + cj) * 4L;
        long row1 = ((long)(ci + 1) * N + cj) * 4L;

        if (isFirstKBlock) {
            c00.intoMemorySegment(C, row0, NATIVE); c01.intoMemorySegment(C, row0 + STRIDE_VEC_BYTES_I32, NATIVE);
            c10.intoMemorySegment(C, row1, NATIVE); c11.intoMemorySegment(C, row1 + STRIDE_VEC_BYTES_I32, NATIVE);
        } else {
            IntVector.fromMemorySegment(SPECIESINT, C, row0, NATIVE).add(c00).intoMemorySegment(C, row0, NATIVE);
            IntVector.fromMemorySegment(SPECIESINT, C, row0 + STRIDE_VEC_BYTES_I32, NATIVE).add(c01).intoMemorySegment(C, row0 + STRIDE_VEC_BYTES_I32, NATIVE);
            IntVector.fromMemorySegment(SPECIESINT, C, row1, NATIVE).add(c10).intoMemorySegment(C, row1, NATIVE);
            IntVector.fromMemorySegment(SPECIESINT, C, row1 + STRIDE_VEC_BYTES_I32, NATIVE).add(c11).intoMemorySegment(C, row1 + STRIDE_VEC_BYTES_I32, NATIVE);
        }
    }

    // Microkernel 1x16 Int - 4-way unrolled
    public static void microKernel1x16_Int(MemorySegment pA, long aBase, int rOff,
                                            MemorySegment pB, long bBase,
                                            MemorySegment C, int ci, int cj,
                                            int kc, int N, boolean isFirstKBlock) {
        var c00 = IntVector.zero(SPECIESINT); var c01 = IntVector.zero(SPECIESINT);

        int k = 0;
        for (; k <= kc - 4; k += 4) {
            // k + 0
            long aOff0 = aBase + (long) k * STEP_A_ROW_BYTES_I32 + (long) rOff * 4L;
            long bOff0 = bBase + (long) k * STEP_B_ROW_BYTES_I32;
            var b0_0 = IntVector.fromMemorySegment(SPECIESINT, pB, bOff0, NATIVE);
            var b1_0 = IntVector.fromMemorySegment(SPECIESINT, pB, bOff0 + STRIDE_VEC_BYTES_I32, NATIVE);
            var a_0 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff0));
            c00 = c00.add(a_0.mul(b0_0)); c01 = c01.add(a_0.mul(b1_0));

            // k + 1
            long aOff1 = aOff0 + STEP_A_ROW_BYTES_I32;
            long bOff1 = bOff0 + STEP_B_ROW_BYTES_I32;
            var b0_1 = IntVector.fromMemorySegment(SPECIESINT, pB, bOff1, NATIVE);
            var b1_1 = IntVector.fromMemorySegment(SPECIESINT, pB, bOff1 + STRIDE_VEC_BYTES_I32, NATIVE);
            var a_1 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff1));
            c00 = c00.add(a_1.mul(b0_1)); c01 = c01.add(a_1.mul(b1_1));

            // k + 2
            long aOff2 = aOff1 + STEP_A_ROW_BYTES_I32;
            long bOff2 = bOff1 + STEP_B_ROW_BYTES_I32;
            var b0_2 = IntVector.fromMemorySegment(SPECIESINT, pB, bOff2, NATIVE);
            var b1_2 = IntVector.fromMemorySegment(SPECIESINT, pB, bOff2 + STRIDE_VEC_BYTES_I32, NATIVE);
            var a_2 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff2));
            c00 = c00.add(a_2.mul(b0_2)); c01 = c01.add(a_2.mul(b1_2));

            // k + 3
            long aOff3 = aOff2 + STEP_A_ROW_BYTES_I32;
            long bOff3 = bOff2 + STEP_B_ROW_BYTES_I32;
            var b0_3 = IntVector.fromMemorySegment(SPECIESINT, pB, bOff3, NATIVE);
            var b1_3 = IntVector.fromMemorySegment(SPECIESINT, pB, bOff3 + STRIDE_VEC_BYTES_I32, NATIVE);
            var a_3 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff3));
            c00 = c00.add(a_3.mul(b0_3)); c01 = c01.add(a_3.mul(b1_3));
        }

        for (; k < kc; k++) {
            long aOff = aBase + (long) k * STEP_A_ROW_BYTES_I32 + (long) rOff * 4L;
            long bOff = bBase + (long) k * STEP_B_ROW_BYTES_I32;

            var b0 = IntVector.fromMemorySegment(SPECIESINT, pB, bOff, NATIVE);
            var b1 = IntVector.fromMemorySegment(SPECIESINT, pB, bOff + STRIDE_VEC_BYTES_I32, NATIVE);

            var a = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff));
            c00 = c00.add(a.mul(b0)); c01 = c01.add(a.mul(b1));
        }

        long row = ((long) ci * N + cj) * 4L;
        if (isFirstKBlock) {
            c00.intoMemorySegment(C, row, NATIVE);
            c01.intoMemorySegment(C, row + STRIDE_VEC_BYTES_I32, NATIVE);
        } else {
            IntVector.fromMemorySegment(SPECIESINT, C, row, NATIVE).add(c00).intoMemorySegment(C, row, NATIVE);
            IntVector.fromMemorySegment(SPECIESINT, C, row + STRIDE_VEC_BYTES_I32, NATIVE).add(c01).intoMemorySegment(C, row + STRIDE_VEC_BYTES_I32, NATIVE);
        }
    }

    // Microkernel 8x12 Int - 4-way unrolled
    public static void microKernel8x12_Int(MemorySegment pA, long aBase, MemorySegment pB, long bBase,
                                            MemorySegment C, int ci, int cj, int kc, int N, boolean isFirstKBlock) {
        var c00 = IntVector.zero(SPECIESINT); var c01 = IntVector.zero(SPECIESINT);
        var c10 = IntVector.zero(SPECIESINT); var c11 = IntVector.zero(SPECIESINT);
        var c20 = IntVector.zero(SPECIESINT); var c21 = IntVector.zero(SPECIESINT);
        var c30 = IntVector.zero(SPECIESINT); var c31 = IntVector.zero(SPECIESINT);
        var c40 = IntVector.zero(SPECIESINT); var c41 = IntVector.zero(SPECIESINT);
        var c50 = IntVector.zero(SPECIESINT); var c51 = IntVector.zero(SPECIESINT);
        var c60 = IntVector.zero(SPECIESINT); var c61 = IntVector.zero(SPECIESINT);
        var c70 = IntVector.zero(SPECIESINT); var c71 = IntVector.zero(SPECIESINT);

        long stride = (long) SPECIESINT.length() * 4L;

        int k = 0;
        for (; k <= kc - 4; k += 4) {
            // k + 0
            long aOff0 = aBase + (long)(k + 0) * 8 * 4L;
            long bOff0 = bBase + (long)(k + 0) * 16 * 4L;
            var b0_0 = IntVector.fromMemorySegment(SPECIESINT, pB, bOff0, NATIVE);
            var b1_0 = IntVector.fromMemorySegment(SPECIESINT, pB, bOff0 + stride, NATIVE);
            var a0_0 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff0 + 0 * 4L));
            c00 = c00.add(a0_0.mul(b0_0)); c01 = c01.add(a0_0.mul(b1_0));
            var a1_0 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff0 + 1 * 4L));
            c10 = c10.add(a1_0.mul(b0_0)); c11 = c11.add(a1_0.mul(b1_0));
            var a2_0 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff0 + 2 * 4L));
            c20 = c20.add(a2_0.mul(b0_0)); c21 = c21.add(a2_0.mul(b1_0));
            var a3_0 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff0 + 3 * 4L));
            c30 = c30.add(a3_0.mul(b0_0)); c31 = c31.add(a3_0.mul(b1_0));
            var a4_0 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff0 + 4 * 4L));
            c40 = c40.add(a4_0.mul(b0_0)); c41 = c41.add(a4_0.mul(b1_0));
            var a5_0 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff0 + 5 * 4L));
            c50 = c50.add(a5_0.mul(b0_0)); c51 = c51.add(a5_0.mul(b1_0));
            var a6_0 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff0 + 6 * 4L));
            c60 = c60.add(a6_0.mul(b0_0)); c61 = c61.add(a6_0.mul(b1_0));
            var a7_0 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff0 + 7 * 4L));
            c70 = c70.add(a7_0.mul(b0_0)); c71 = c71.add(a7_0.mul(b1_0));

            // k + 1
            long aOff1 = aBase + (long)(k + 1) * 8 * 4L;
            long bOff1 = bBase + (long)(k + 1) * 16 * 4L;
            var b0_1 = IntVector.fromMemorySegment(SPECIESINT, pB, bOff1, NATIVE);
            var b1_1 = IntVector.fromMemorySegment(SPECIESINT, pB, bOff1 + stride, NATIVE);
            var a0_1 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff1 + 0 * 4L));
            c00 = c00.add(a0_1.mul(b0_1)); c01 = c01.add(a0_1.mul(b1_1));
            var a1_1 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff1 + 1 * 4L));
            c10 = c10.add(a1_1.mul(b0_1)); c11 = c11.add(a1_1.mul(b1_1));
            var a2_1 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff1 + 2 * 4L));
            c20 = c20.add(a2_1.mul(b0_1)); c21 = c21.add(a2_1.mul(b1_1));
            var a3_1 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff1 + 3 * 4L));
            c30 = c30.add(a3_1.mul(b0_1)); c31 = c31.add(a3_1.mul(b1_1));
            var a4_1 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff1 + 4 * 4L));
            c40 = c40.add(a4_1.mul(b0_1)); c41 = c41.add(a4_1.mul(b1_1));
            var a5_1 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff1 + 5 * 4L));
            c50 = c50.add(a5_1.mul(b0_1)); c51 = c51.add(a5_1.mul(b1_1));
            var a6_1 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff1 + 6 * 4L));
            c60 = c60.add(a6_1.mul(b0_1)); c61 = c61.add(a6_1.mul(b1_1));
            var a7_1 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff1 + 7 * 4L));
            c70 = c70.add(a7_1.mul(b0_1)); c71 = c71.add(a7_1.mul(b1_1));

            // k + 2
            long aOff2 = aBase + (long)(k + 2) * 8 * 4L;
            long bOff2 = bBase + (long)(k + 2) * 16 * 4L;
            var b0_2 = IntVector.fromMemorySegment(SPECIESINT, pB, bOff2, NATIVE);
            var b1_2 = IntVector.fromMemorySegment(SPECIESINT, pB, bOff2 + stride, NATIVE);
            var a0_2 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff2 + 0 * 4L));
            c00 = c00.add(a0_2.mul(b0_2)); c01 = c01.add(a0_2.mul(b1_2));
            var a1_2 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff2 + 1 * 4L));
            c10 = c10.add(a1_2.mul(b0_2)); c11 = c11.add(a1_2.mul(b1_2));
            var a2_2 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff2 + 2 * 4L));
            c20 = c20.add(a2_2.mul(b0_2)); c21 = c21.add(a2_2.mul(b1_2));
            var a3_2 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff2 + 3 * 4L));
            c30 = c30.add(a3_2.mul(b0_2)); c31 = c31.add(a3_2.mul(b1_2));
            var a4_2 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff2 + 4 * 4L));
            c40 = c40.add(a4_2.mul(b0_2)); c41 = c41.add(a4_2.mul(b1_2));
            var a5_2 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff2 + 5 * 4L));
            c50 = c50.add(a5_2.mul(b0_2)); c51 = c51.add(a5_2.mul(b1_2));
            var a6_2 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff2 + 6 * 4L));
            c60 = c60.add(a6_2.mul(b0_2)); c61 = c61.add(a6_2.mul(b1_2));
            var a7_2 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff2 + 7 * 4L));
            c70 = c70.add(a7_2.mul(b0_2)); c71 = c71.add(a7_2.mul(b1_2));

            // k + 3
            long aOff3 = aBase + (long)(k + 3) * 8 * 4L;
            long bOff3 = bBase + (long)(k + 3) * 16 * 4L;
            var b0_3 = IntVector.fromMemorySegment(SPECIESINT, pB, bOff3, NATIVE);
            var b1_3 = IntVector.fromMemorySegment(SPECIESINT, pB, bOff3 + stride, NATIVE);
            var a0_3 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff3 + 0 * 4L));
            c00 = c00.add(a0_3.mul(b0_3)); c01 = c01.add(a0_3.mul(b1_3));
            var a1_3 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff3 + 1 * 4L));
            c10 = c10.add(a1_3.mul(b0_3)); c11 = c11.add(a1_3.mul(b1_3));
            var a2_3 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff3 + 2 * 4L));
            c20 = c20.add(a2_3.mul(b0_3)); c21 = c21.add(a2_3.mul(b1_3));
            var a3_3 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff3 + 3 * 4L));
            c30 = c30.add(a3_3.mul(b0_3)); c31 = c31.add(a3_3.mul(b1_3));
            var a4_3 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff3 + 4 * 4L));
            c40 = c40.add(a4_3.mul(b0_3)); c41 = c41.add(a4_3.mul(b1_3));
            var a5_3 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff3 + 5 * 4L));
            c50 = c50.add(a5_3.mul(b0_3)); c51 = c51.add(a5_3.mul(b1_3));
            var a6_3 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff3 + 6 * 4L));
            c60 = c60.add(a6_3.mul(b0_3)); c61 = c61.add(a6_3.mul(b1_3));
            var a7_3 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff3 + 7 * 4L));
            c70 = c70.add(a7_3.mul(b0_3)); c71 = c71.add(a7_3.mul(b1_3));
        }

        for (; k < kc; k++) {
            long aOff = aBase + (long) k * 8 * 4L;
            long bOff = bBase + (long) k * 16 * 4L;

            var b0 = IntVector.fromMemorySegment(SPECIESINT, pB, bOff, NATIVE);
            var b1 = IntVector.fromMemorySegment(SPECIESINT, pB, bOff + stride, NATIVE);

            var a0 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff + 0 * 4L));
            c00 = c00.add(a0.mul(b0)); c01 = c01.add(a0.mul(b1));
            var a1 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff + 1 * 4L));
            c10 = c10.add(a1.mul(b0)); c11 = c11.add(a1.mul(b1));
            var a2 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff + 2 * 4L));
            c20 = c20.add(a2.mul(b0)); c21 = c21.add(a2.mul(b1));
            var a3 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff + 3 * 4L));
            c30 = c30.add(a3.mul(b0)); c31 = c31.add(a3.mul(b1));
            var a4 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff + 4 * 4L));
            c40 = c40.add(a4.mul(b0)); c41 = c41.add(a4.mul(b1));
            var a5 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff + 5 * 4L));
            c50 = c50.add(a5.mul(b0)); c51 = c51.add(a5.mul(b1));
            var a6 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff + 6 * 4L));
            c60 = c60.add(a6.mul(b0)); c61 = c61.add(a6.mul(b1));
            var a7 = IntVector.broadcast(SPECIESINT, pA.get(VI, aOff + 7 * 4L));
            c70 = c70.add(a7.mul(b0)); c71 = c71.add(a7.mul(b1));
        }

        IntVector[] acc0 = {c00, c10, c20, c30, c40, c50, c60, c70};
        IntVector[] acc1 = {c01, c11, c21, c31, c41, c51, c61, c71};

        for (int r = 0; r < 8; r++) {
            long row = ((long)(ci + r) * N + cj) * 4L;
            if (isFirstKBlock) {
                acc0[r].intoMemorySegment(C, row, NATIVE);
                for (int lane = 0; lane < 4; lane++) {
                    long idx = (long)(ci + r) * N + cj + 8 + lane;
                    C.setAtIndex(VI, idx, acc1[r].lane(lane));
                }
            } else {
                IntVector.fromMemorySegment(SPECIESINT, C, row, NATIVE).add(acc0[r]).intoMemorySegment(C, row, NATIVE);
                for (int lane = 0; lane < 4; lane++) {
                    long idx = (long)(ci + r) * N + cj + 8 + lane;
                    C.setAtIndex(VI, idx, C.getAtIndex(VI, idx) + acc1[r].lane(lane));
                }
            }
        }
    }

    public static void microKernelScalar_Int(MemorySegment pA, long aBase, int rOff,
                                              MemorySegment pB, long bBase, MemorySegment C,
                                              int ci, int cj, int kc, int N, int mr, int nr,
                                              int MR_dim, int NR_dim, boolean isFirstKBlock) {
        int[] acc = new int[mr * nr];
        for (int k = 0; k < kc; k++) {
            long aOff = aBase + (long) k * MR_dim * 4L + (long) rOff * 4L;
            long bOff = bBase + (long) k * NR_dim * 4L;
            for (int r = 0; r < mr; r++) {
                int aVal = pA.get(VI, aOff + (long) r * 4L);
                for (int c = 0; c < nr; c++) {
                    acc[r * nr + c] += aVal * pB.get(VI, bOff + (long) c * 4L);
                }
            }
        }
        for (int r = 0; r < mr; r++) {
            for (int c = 0; c < nr; c++) {
                long cIdx = (long)(ci + r) * N + cj + c;
                if (isFirstKBlock) {
                    C.setAtIndex(VI, cIdx, acc[r * nr + c]);
                } else {
                    C.setAtIndex(VI, cIdx, C.getAtIndex(VI, cIdx) + acc[r * nr + c]);
                }
            }
        }
    }

    // Backward-compatibility aliases for tests and external callers
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