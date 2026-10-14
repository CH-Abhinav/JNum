package jnum.internal.kernel.linalg.matmul;

import static jnum.internal.Constants.*;

import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;
import java.util.concurrent.RecursiveAction;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * BLIS Panel Packing Routines for Matrices A and B.
 */
public final class MatMulPacking {

    private static final ByteOrder NATIVE = NATIVE_ORDER;
    private static final VectorSpecies<Float> SPECIES = SPECIES_F32;
    private static final VectorSpecies<Double> SPECIESDB = SPECIES_F64;
    private static final VectorSpecies<Integer> SPECIESINT = SPECIES_I32;

    private MatMulPacking() {
        throw new AssertionError("Utility class");
    }

    public static void packA_panel_Arm_Float(MemorySegment src, MemorySegment dst, int rowStart, int mc, int colStart, int kc, long a_s0, long a_s1) {
        int fullPanels = mc / MR;
        int tailRows = mc % MR;

        if (a_s1 == 1L) {
            for (int p = 0; p < fullPanels; p++) {
                long dstBase = (long) p * MR * kc * 4L;
                long r0 = (long)(rowStart + p * MR + 0) * a_s0 + colStart;
                long r1 = (long)(rowStart + p * MR + 1) * a_s0 + colStart;
                long r2 = (long)(rowStart + p * MR + 2) * a_s0 + colStart;
                long r3 = (long)(rowStart + p * MR + 3) * a_s0 + colStart;
                long r4 = (long)(rowStart + p * MR + 4) * a_s0 + colStart;
                long r5 = (long)(rowStart + p * MR + 5) * a_s0 + colStart;

                int k = 0;
                for (; k <= kc - 4; k += 4) {
                    long dOff0 = dstBase + (long) k * STEP_A_ROW_BYTES_F32;
                    dst.set(VF, dOff0,                    src.getAtIndex(VF, r0 + k));
                    dst.set(VF, dOff0 + 1L * 4L, src.getAtIndex(VF, r1 + k));
                    dst.set(VF, dOff0 + 2L * 4L, src.getAtIndex(VF, r2 + k));
                    dst.set(VF, dOff0 + 3L * 4L, src.getAtIndex(VF, r3 + k));
                    dst.set(VF, dOff0 + 4L * 4L, src.getAtIndex(VF, r4 + k));
                    dst.set(VF, dOff0 + 5L * 4L, src.getAtIndex(VF, r5 + k));

                    long dOff1 = dOff0 + STEP_A_ROW_BYTES_F32;
                    dst.set(VF, dOff1,                    src.getAtIndex(VF, r0 + k + 1));
                    dst.set(VF, dOff1 + 1L * 4L, src.getAtIndex(VF, r1 + k + 1));
                    dst.set(VF, dOff1 + 2L * 4L, src.getAtIndex(VF, r2 + k + 1));
                    dst.set(VF, dOff1 + 3L * 4L, src.getAtIndex(VF, r3 + k + 1));
                    dst.set(VF, dOff1 + 4L * 4L, src.getAtIndex(VF, r4 + k + 1));
                    dst.set(VF, dOff1 + 5L * 4L, src.getAtIndex(VF, r5 + k + 1));

                    long dOff2 = dOff1 + STEP_A_ROW_BYTES_F32;
                    dst.set(VF, dOff2,                    src.getAtIndex(VF, r0 + k + 2));
                    dst.set(VF, dOff2 + 1L * 4L, src.getAtIndex(VF, r1 + k + 2));
                    dst.set(VF, dOff2 + 2L * 4L, src.getAtIndex(VF, r2 + k + 2));
                    dst.set(VF, dOff2 + 3L * 4L, src.getAtIndex(VF, r3 + k + 2));
                    dst.set(VF, dOff2 + 4L * 4L, src.getAtIndex(VF, r4 + k + 2));
                    dst.set(VF, dOff2 + 5L * 4L, src.getAtIndex(VF, r5 + k + 2));

                    long dOff3 = dOff2 + STEP_A_ROW_BYTES_F32;
                    dst.set(VF, dOff3,                    src.getAtIndex(VF, r0 + k + 3));
                    dst.set(VF, dOff3 + 1L * 4L, src.getAtIndex(VF, r1 + k + 3));
                    dst.set(VF, dOff3 + 2L * 4L, src.getAtIndex(VF, r2 + k + 3));
                    dst.set(VF, dOff3 + 3L * 4L, src.getAtIndex(VF, r3 + k + 3));
                    dst.set(VF, dOff3 + 4L * 4L, src.getAtIndex(VF, r4 + k + 3));
                    dst.set(VF, dOff3 + 5L * 4L, src.getAtIndex(VF, r5 + k + 3));
                }
                for (; k < kc; k++) {
                    long dOff = dstBase + (long) k * STEP_A_ROW_BYTES_F32;
                    dst.set(VF, dOff,                    src.getAtIndex(VF, r0 + k));
                    dst.set(VF, dOff + 1L * 4L, src.getAtIndex(VF, r1 + k));
                    dst.set(VF, dOff + 2L * 4L, src.getAtIndex(VF, r2 + k));
                    dst.set(VF, dOff + 3L * 4L, src.getAtIndex(VF, r3 + k));
                    dst.set(VF, dOff + 4L * 4L, src.getAtIndex(VF, r4 + k));
                    dst.set(VF, dOff + 5L * 4L, src.getAtIndex(VF, r5 + k));
                }
            }
            if (tailRows > 0) {
                long dstBase = (long) fullPanels * MR * kc * 4L;
                for (int r = 0; r < MR; r++) {
                    if (r < tailRows) {
                        long srcRow = (long)(rowStart + fullPanels * MR + r) * a_s0 + colStart;
                        for (int k = 0; k < kc; k++) {
                            dst.set(VF,
                                dstBase + (long) k * STEP_A_ROW_BYTES_F32 + (long) r * 4L,
                                src.getAtIndex(VF, srcRow + k));
                        }
                    } else {
                        for (int k = 0; k < kc; k++) {
                            dst.set(VF,
                                dstBase + (long) k * STEP_A_ROW_BYTES_F32 + (long) r * 4L, 0.0f);
                        }
                    }
                }
            }
        } else {
            for (int p = 0; p < fullPanels; p++) {
                long dstBase = (long) p * MR * kc * 4L;
                long r0 = (long)(rowStart + p * MR + 0) * a_s0 + (long) colStart * a_s1;
                long r1 = (long)(rowStart + p * MR + 1) * a_s0 + (long) colStart * a_s1;
                long r2 = (long)(rowStart + p * MR + 2) * a_s0 + (long) colStart * a_s1;
                long r3 = (long)(rowStart + p * MR + 3) * a_s0 + (long) colStart * a_s1;
                long r4 = (long)(rowStart + p * MR + 4) * a_s0 + (long) colStart * a_s1;
                long r5 = (long)(rowStart + p * MR + 5) * a_s0 + (long) colStart * a_s1;

                for (int k = 0; k < kc; k++) {
                    long dOff = dstBase + (long) k * STEP_A_ROW_BYTES_F32;
                    long kOff = (long) k * a_s1;
                    dst.set(VF, dOff,                    src.getAtIndex(VF, r0 + kOff));
                    dst.set(VF, dOff + 1L * 4L, src.getAtIndex(VF, r1 + kOff));
                    dst.set(VF, dOff + 2L * 4L, src.getAtIndex(VF, r2 + kOff));
                    dst.set(VF, dOff + 3L * 4L, src.getAtIndex(VF, r3 + kOff));
                    dst.set(VF, dOff + 4L * 4L, src.getAtIndex(VF, r4 + kOff));
                    dst.set(VF, dOff + 5L * 4L, src.getAtIndex(VF, r5 + kOff));
                }
            }
            if (tailRows > 0) {
                long dstBase = (long) fullPanels * MR * kc * 4L;
                for (int r = 0; r < MR; r++) {
                    if (r < tailRows) {
                        long srcRow = (long)(rowStart + fullPanels * MR + r) * a_s0 + (long) colStart * a_s1;
                        for (int k = 0; k < kc; k++) {
                            dst.set(VF,
                                dstBase + (long) k * STEP_A_ROW_BYTES_F32 + (long) r * 4L,
                                src.getAtIndex(VF, srcRow + (long) k * a_s1));
                        }
                    } else {
                        for (int k = 0; k < kc; k++) {
                            dst.set(VF,
                                dstBase + (long) k * STEP_A_ROW_BYTES_F32 + (long) r * 4L, 0.0f);
                        }
                    }
                }
            }
        }
    }

    public static void packB_panel_Arm_Float(MemorySegment src, MemorySegment dst, int rowStart, int colStart, int kc, int nc, long b_s0, long b_s1) {
        int fullPanels = nc / NR;
        int tailCols = nc % NR;
        if (fullPanels >= 4 && AVAILABLE_CORES >= 2) {
            POOL.invoke(new PackBTask_Arm_Float(src, dst, rowStart, colStart, kc, nc, b_s0, b_s1, 0, fullPanels));
        } else {
            packB_panels_range_Arm_Float(src, dst, rowStart, colStart, kc, b_s0, b_s1, 0, fullPanels);
        }
        if (tailCols > 0) {
            packB_tail_Arm_Float(src, dst, rowStart, colStart, kc, nc, b_s0, b_s1, fullPanels, tailCols);
        }
    }

    public static void packB_panels_range_Arm_Float(MemorySegment src, MemorySegment dst, int rowStart, int colStart, int kc, long b_s0, long b_s1, int pStart, int pEnd) {
        if (b_s1 == 1L) {
            for (int p = pStart; p < pEnd; p++) {
                long dstBase = (long) p * NR * kc * 4L;
                for (int k = 0; k < kc; k++) {
                    long srcOff = ((long)(rowStart + k) * b_s0 + colStart + (long) p * NR) * 4L;
                    long dstOff = dstBase + (long) k * STEP_B_ROW_BYTES_F32;
                    FloatVector.fromMemorySegment(SPECIES, src, srcOff, NATIVE).intoMemorySegment(dst, dstOff, NATIVE);
                    FloatVector.fromMemorySegment(SPECIES, src, srcOff + STRIDE_VEC_BYTES_F32, NATIVE).intoMemorySegment(dst, dstOff + STRIDE_VEC_BYTES_F32, NATIVE);
                }
            }
        } else {
            for (int p = pStart; p < pEnd; p++) {
                long dstBase = (long) p * NR * kc * 4L;
                long colBase = colStart + (long) p * NR;
                for (int k = 0; k < kc; k++) {
                    long rowBase = (long)(rowStart + k) * b_s0;
                    long dstOff = dstBase + (long) k * STEP_B_ROW_BYTES_F32;
                    for (int c = 0; c < NR; c++) {
                        dst.set(VF, dstOff + (long) c * 4L,
                            src.get(VF, (rowBase + (colBase + c) * b_s1) * 4L));
                    }
                }
            }
        }
    }

    public static void packB_tail_Arm_Float(MemorySegment src, MemorySegment dst, int rowStart, int colStart, int kc, int nc, long b_s0, long b_s1, int fullPanels, int tailCols) {
        long dstBase = (long) fullPanels * NR * kc * 4L;
        if (b_s1 == 1L) {
            for (int k = 0; k < kc; k++) {
                long srcOff = ((long)(rowStart + k) * b_s0 + colStart + (long) fullPanels * NR) * 4L;
                long dstOff = dstBase + (long) k * STEP_B_ROW_BYTES_F32;
                for (int c = 0; c < tailCols; c++) {
                    dst.set(VF, dstOff + (long) c * 4L,
                            src.get(VF, srcOff + (long) c * 4L));
                }
                for (int c = tailCols; c < NR; c++) {
                    dst.set(VF, dstOff + (long) c * 4L, 0.0f);
                }
            }
        } else {
            long colBase = colStart + (long) fullPanels * NR;
            for (int k = 0; k < kc; k++) {
                long rowBase = (long)(rowStart + k) * b_s0;
                long dstOff = dstBase + (long) k * STEP_B_ROW_BYTES_F32;
                for (int c = 0; c < tailCols; c++) {
                    dst.set(VF, dstOff + (long) c * 4L,
                        src.get(VF, (rowBase + (colBase + c) * b_s1) * 4L));
                }
                for (int c = tailCols; c < NR; c++) {
                    dst.set(VF, dstOff + (long) c * 4L, 0.0f);
                }
            }
        }
    }

    public static final class PackBTask_Arm_Float extends RecursiveAction {
        final MemorySegment src, dst;
        final int rowStart, colStart, kc, nc;
        final long b_s0, b_s1;
        final int pStart, pEnd;

        PackBTask_Arm_Float(MemorySegment src, MemorySegment dst, int rowStart, int colStart,
                            int kc, int nc, long b_s0, long b_s1, int pStart, int pEnd) {
            this.src = src; this.dst = dst; this.rowStart = rowStart; this.colStart = colStart;
            this.kc = kc; this.nc = nc; this.b_s0 = b_s0; this.b_s1 = b_s1;
            this.pStart = pStart; this.pEnd = pEnd;
        }

        @Override
        protected void compute() {
            int panels = pEnd - pStart;
            if (panels <= 4) {
                packB_panels_range_Arm_Float(src, dst, rowStart, colStart, kc, b_s0, b_s1, pStart, pEnd);
            } else {
                int mid = pStart + panels / 2;
                invokeAll(
                    new PackBTask_Arm_Float(src, dst, rowStart, colStart, kc, nc, b_s0, b_s1, pStart, mid),
                    new PackBTask_Arm_Float(src, dst, rowStart, colStart, kc, nc, b_s0, b_s1, mid, pEnd)
                );
            }
        }
    }

    public static void packA_panel_Aarch_Float(MemorySegment src, MemorySegment dst, int rowStart, int mc, int colStart, int kc, long a_s0, long a_s1) {
        int fullPanels = mc / 8;
        int tailRows = mc % 8;

        if (a_s1 == 1L) {
            for (int p = 0; p < fullPanels; p++) {
                long dstBase = (long) p * 8 * kc * 4L;
                for (int r = 0; r < 8; r++) {
                    long srcRow = (long)(rowStart + p * 8 + r) * a_s0 + colStart;
                    for (int k = 0; k < kc; k++) {
                        float v = src.getAtIndex(VF, srcRow + k);
                        dst.set(VF, dstBase + (long) k * 8 * 4L + (long) r * 4L, v);
                    }
                }
            }
            if (tailRows > 0) {
                long dstBase = (long) fullPanels * 8 * kc * 4L;
                for (int r = 0; r < 8; r++) {
                    if (r < tailRows) {
                        long srcRow = (long)(rowStart + fullPanels * 8 + r) * a_s0 + colStart;
                        for (int k = 0; k < kc; k++) {
                            dst.set(VF,
                                dstBase + (long) k * 8 * 4L + (long) r * 4L,
                                src.getAtIndex(VF, srcRow + k));
                        }
                    } else {
                        for (int k = 0; k < kc; k++) {
                            dst.set(VF,
                                dstBase + (long) k * 8 * 4L + (long) r * 4L, 0.0f);
                        }
                    }
                }
            }
        } else {
            for (int p = 0; p < fullPanels; p++) {
                long dstBase = (long) p * 8 * kc * 4L;
                for (int r = 0; r < 8; r++) {
                    long srcRow = (long)(rowStart + p * 8 + r) * a_s0 + (long) colStart * a_s1;
                    for (int k = 0; k < kc; k++) {
                        float v = src.getAtIndex(VF, srcRow + (long) k * a_s1);
                        dst.set(VF, dstBase + (long) k * 8 * 4L + (long) r * 4L, v);
                    }
                }
            }
            if (tailRows > 0) {
                long dstBase = (long) fullPanels * 8 * kc * 4L;
                for (int r = 0; r < 8; r++) {
                    if (r < tailRows) {
                        long srcRow = (long)(rowStart + fullPanels * 8 + r) * a_s0 + (long) colStart * a_s1;
                        for (int k = 0; k < kc; k++) {
                            dst.set(VF,
                                dstBase + (long) k * 8 * 4L + (long) r * 4L,
                                src.getAtIndex(VF, srcRow + (long) k * a_s1));
                        }
                    } else {
                        for (int k = 0; k < kc; k++) {
                            dst.set(VF,
                                dstBase + (long) k * 8 * 4L + (long) r * 4L, 0.0f);
                        }
                    }
                }
            }
        }
    }

    public static void packB_panel_Aarch_Float(MemorySegment src, MemorySegment dst, int rowStart, int colStart, int kc, int nc, long b_s0, long b_s1) {
        int fullPanels = nc / 12;
        int tailCols = nc % 12;
        if (fullPanels >= 4 && AVAILABLE_CORES >= 2) {
            POOL.invoke(new PackBTask_Aarch_Float(src, dst, rowStart, colStart, kc, nc, b_s0, b_s1, 0, fullPanels));
        } else {
            packB_panels_range_Aarch_Float(src, dst, rowStart, colStart, kc, b_s0, b_s1, 0, fullPanels);
        }
        if (tailCols > 0) {
            packB_tail_Aarch_Float(src, dst, rowStart, colStart, kc, nc, b_s0, b_s1, fullPanels, tailCols);
        }
    }

    public static void packB_panels_range_Aarch_Float(MemorySegment src, MemorySegment dst, int rowStart, int colStart, int kc, long b_s0, long b_s1, int pStart, int pEnd) {
        if (b_s1 == 1L) {
            for (int p = pStart; p < pEnd; p++) {
                long dstBase = (long) p * 16 * kc * 4L;
                for (int k = 0; k < kc; k++) {
                    long srcOff = ((long)(rowStart + k) * b_s0 + colStart + (long) p * 12) * 4L;
                    long dstOff = dstBase + (long) k * 16 * 4L;
                    MemorySegment.copy(src, srcOff, dst, dstOff, 12L * 4L);
                    dst.set(VF, dstOff + 12 * 4L, 0.0f);
                    dst.set(VF, dstOff + 13 * 4L, 0.0f);
                    dst.set(VF, dstOff + 14 * 4L, 0.0f);
                    dst.set(VF, dstOff + 15 * 4L, 0.0f);
                }
            }
        } else {
            for (int p = pStart; p < pEnd; p++) {
                long dstBase = (long) p * 16 * kc * 4L;
                long colBase = colStart + (long) p * 12;
                for (int k = 0; k < kc; k++) {
                    long rowBase = (long)(rowStart + k) * b_s0;
                    long dstOff = dstBase + (long) k * 16 * 4L;
                    for (int c = 0; c < 12; c++) {
                        dst.set(VF, dstOff + (long) c * 4L,
                            src.get(VF, (rowBase + (colBase + c) * b_s1) * 4L));
                    }
                    dst.set(VF, dstOff + 12 * 4L, 0.0f);
                    dst.set(VF, dstOff + 13 * 4L, 0.0f);
                    dst.set(VF, dstOff + 14 * 4L, 0.0f);
                    dst.set(VF, dstOff + 15 * 4L, 0.0f);
                }
            }
        }
    }

    public static void packB_tail_Aarch_Float(MemorySegment src, MemorySegment dst, int rowStart, int colStart, int kc, int nc, long b_s0, long b_s1, int fullPanels, int tailCols) {
        long dstBase = (long) fullPanels * 16 * kc * 4L;
        if (b_s1 == 1L) {
            for (int k = 0; k < kc; k++) {
                long srcOff = ((long)(rowStart + k) * b_s0 + colStart + (long) fullPanels * 12) * 4L;
                long dstOff = dstBase + (long) k * 16 * 4L;
                MemorySegment.copy(src, srcOff, dst, dstOff, (long) tailCols * 4L);
                for (int c = tailCols; c < 16; c++) {
                    dst.set(VF, dstOff + (long) c * 4L, 0.0f);
                }
            }
        } else {
            long colBase = colStart + (long) fullPanels * 12;
            for (int k = 0; k < kc; k++) {
                long rowBase = (long)(rowStart + k) * b_s0;
                long dstOff = dstBase + (long) k * 16 * 4L;
                for (int c = 0; c < tailCols; c++) {
                    dst.set(VF, dstOff + (long) c * 4L,
                        src.get(VF, (rowBase + (colBase + c) * b_s1) * 4L));
                }
                for (int c = tailCols; c < 16; c++) {
                    dst.set(VF, dstOff + (long) c * 4L, 0.0f);
                }
            }
        }
    }

    public static final class PackBTask_Aarch_Float extends RecursiveAction {
        final MemorySegment src, dst;
        final int rowStart, colStart, kc, nc;
        final long b_s0, b_s1;
        final int pStart, pEnd;

        PackBTask_Aarch_Float(MemorySegment src, MemorySegment dst, int rowStart, int colStart,
                              int kc, int nc, long b_s0, long b_s1, int pStart, int pEnd) {
            this.src = src; this.dst = dst; this.rowStart = rowStart; this.colStart = colStart;
            this.kc = kc; this.nc = nc; this.b_s0 = b_s0; this.b_s1 = b_s1;
            this.pStart = pStart; this.pEnd = pEnd;
        }

        @Override
        protected void compute() {
            int panels = pEnd - pStart;
            if (panels <= 4) {
                packB_panels_range_Aarch_Float(src, dst, rowStart, colStart, kc, b_s0, b_s1, pStart, pEnd);
            } else {
                int mid = pStart + panels / 2;
                invokeAll(
                    new PackBTask_Aarch_Float(src, dst, rowStart, colStart, kc, nc, b_s0, b_s1, pStart, mid),
                    new PackBTask_Aarch_Float(src, dst, rowStart, colStart, kc, nc, b_s0, b_s1, mid, pEnd)
                );
            }
        }
    }
    public static void packA_panel_Arm_Double(MemorySegment src, MemorySegment dst, int rowStart, int mc, int colStart, int kc, long a_s0, long a_s1) {
        int fullPanels = mc / MR;
        int tailRows = mc % MR;

        if (a_s1 == 1L) {
            for (int p = 0; p < fullPanels; p++) {
                long dstBase = (long) p * MR * kc * 8L;
                long r0 = (long)(rowStart + p * MR + 0) * a_s0 + colStart;
                long r1 = (long)(rowStart + p * MR + 1) * a_s0 + colStart;
                long r2 = (long)(rowStart + p * MR + 2) * a_s0 + colStart;
                long r3 = (long)(rowStart + p * MR + 3) * a_s0 + colStart;
                long r4 = (long)(rowStart + p * MR + 4) * a_s0 + colStart;
                long r5 = (long)(rowStart + p * MR + 5) * a_s0 + colStart;

                int k = 0;
                for (; k <= kc - 4; k += 4) {
                    long dOff0 = dstBase + (long) k * STEP_A_ROW_BYTES_F64;
                    dst.set(VD, dOff0,                    src.getAtIndex(VD, r0 + k));
                    dst.set(VD, dOff0 + 1L * 8L, src.getAtIndex(VD, r1 + k));
                    dst.set(VD, dOff0 + 2L * 8L, src.getAtIndex(VD, r2 + k));
                    dst.set(VD, dOff0 + 3L * 8L, src.getAtIndex(VD, r3 + k));
                    dst.set(VD, dOff0 + 4L * 8L, src.getAtIndex(VD, r4 + k));
                    dst.set(VD, dOff0 + 5L * 8L, src.getAtIndex(VD, r5 + k));

                    long dOff1 = dOff0 + STEP_A_ROW_BYTES_F64;
                    dst.set(VD, dOff1,                    src.getAtIndex(VD, r0 + k + 1));
                    dst.set(VD, dOff1 + 1L * 8L, src.getAtIndex(VD, r1 + k + 1));
                    dst.set(VD, dOff1 + 2L * 8L, src.getAtIndex(VD, r2 + k + 1));
                    dst.set(VD, dOff1 + 3L * 8L, src.getAtIndex(VD, r3 + k + 1));
                    dst.set(VD, dOff1 + 4L * 8L, src.getAtIndex(VD, r4 + k + 1));
                    dst.set(VD, dOff1 + 5L * 8L, src.getAtIndex(VD, r5 + k + 1));

                    long dOff2 = dOff1 + STEP_A_ROW_BYTES_F64;
                    dst.set(VD, dOff2,                    src.getAtIndex(VD, r0 + k + 2));
                    dst.set(VD, dOff2 + 1L * 8L, src.getAtIndex(VD, r1 + k + 2));
                    dst.set(VD, dOff2 + 2L * 8L, src.getAtIndex(VD, r2 + k + 2));
                    dst.set(VD, dOff2 + 3L * 8L, src.getAtIndex(VD, r3 + k + 2));
                    dst.set(VD, dOff2 + 4L * 8L, src.getAtIndex(VD, r4 + k + 2));
                    dst.set(VD, dOff2 + 5L * 8L, src.getAtIndex(VD, r5 + k + 2));

                    long dOff3 = dOff2 + STEP_A_ROW_BYTES_F64;
                    dst.set(VD, dOff3,                    src.getAtIndex(VD, r0 + k + 3));
                    dst.set(VD, dOff3 + 1L * 8L, src.getAtIndex(VD, r1 + k + 3));
                    dst.set(VD, dOff3 + 2L * 8L, src.getAtIndex(VD, r2 + k + 3));
                    dst.set(VD, dOff3 + 3L * 8L, src.getAtIndex(VD, r3 + k + 3));
                    dst.set(VD, dOff3 + 4L * 8L, src.getAtIndex(VD, r4 + k + 3));
                    dst.set(VD, dOff3 + 5L * 8L, src.getAtIndex(VD, r5 + k + 3));
                }
                for (; k < kc; k++) {
                    long dOff = dstBase + (long) k * STEP_A_ROW_BYTES_F64;
                    dst.set(VD, dOff,                    src.getAtIndex(VD, r0 + k));
                    dst.set(VD, dOff + 1L * 8L, src.getAtIndex(VD, r1 + k));
                    dst.set(VD, dOff + 2L * 8L, src.getAtIndex(VD, r2 + k));
                    dst.set(VD, dOff + 3L * 8L, src.getAtIndex(VD, r3 + k));
                    dst.set(VD, dOff + 4L * 8L, src.getAtIndex(VD, r4 + k));
                    dst.set(VD, dOff + 5L * 8L, src.getAtIndex(VD, r5 + k));
                }
            }
            if (tailRows > 0) {
                long dstBase = (long) fullPanels * STEP_A_ROW_BYTES_F64 * kc;
                for (int r = 0; r < MR; r++) {
                    if (r < tailRows) {
                        long srcRow = (long)(rowStart + fullPanels * MR + r) * a_s0 + colStart;
                        for (int k = 0; k < kc; k++) {
                            dst.set(VD,
                                dstBase + (long) k * STEP_A_ROW_BYTES_F64 + (long) r * 8L,
                                src.getAtIndex(VD, srcRow + k));
                        }
                    } else {
                        for (int k = 0; k < kc; k++) {
                            dst.set(VD,
                                dstBase + (long) k * STEP_A_ROW_BYTES_F64 + (long) r * 8L, 0.0d);
                        }
                    }
                }
            }
        } else {
            for (int p = 0; p < fullPanels; p++) {
                long dstBase = (long) p * STEP_A_ROW_BYTES_F64 * kc;
                long r0 = (long)(rowStart + p * MR + 0) * a_s0 + (long) colStart * a_s1;
                long r1 = (long)(rowStart + p * MR + 1) * a_s0 + (long) colStart * a_s1;
                long r2 = (long)(rowStart + p * MR + 2) * a_s0 + (long) colStart * a_s1;
                long r3 = (long)(rowStart + p * MR + 3) * a_s0 + (long) colStart * a_s1;
                long r4 = (long)(rowStart + p * MR + 4) * a_s0 + (long) colStart * a_s1;
                long r5 = (long)(rowStart + p * MR + 5) * a_s0 + (long) colStart * a_s1;

                for (int k = 0; k < kc; k++) {
                    long dOff = dstBase + (long) k * STEP_A_ROW_BYTES_F64;
                    long kOff = (long) k * a_s1;
                    dst.set(VD, dOff,                    src.getAtIndex(VD, r0 + kOff));
                    dst.set(VD, dOff + 1L * 8L, src.getAtIndex(VD, r1 + kOff));
                    dst.set(VD, dOff + 2L * 8L, src.getAtIndex(VD, r2 + kOff));
                    dst.set(VD, dOff + 3L * 8L, src.getAtIndex(VD, r3 + kOff));
                    dst.set(VD, dOff + 4L * 8L, src.getAtIndex(VD, r4 + kOff));
                    dst.set(VD, dOff + 5L * 8L, src.getAtIndex(VD, r5 + kOff));
                }
            }
            if (tailRows > 0) {
                long dstBase = (long) fullPanels * STEP_A_ROW_BYTES_F64 * kc;
                for (int r = 0; r < MR; r++) {
                    if (r < tailRows) {
                        long srcRow = (long)(rowStart + fullPanels * MR + r) * a_s0 + (long) colStart * a_s1;
                        for (int k = 0; k < kc; k++) {
                            dst.set(VD,
                                dstBase + (long) k * STEP_A_ROW_BYTES_F64 + (long) r * 8L,
                                src.getAtIndex(VD, srcRow + (long) k * a_s1));
                        }
                    } else {
                        for (int k = 0; k < kc; k++) {
                            dst.set(VD,
                                dstBase + (long) k * STEP_A_ROW_BYTES_F64 + (long) r * 8L, 0.0d);
                        }
                    }
                }
            }
        }
    }

    public static void packB_panel_Arm_Double(MemorySegment src, MemorySegment dst, int rowStart, int colStart, int kc, int nc, long b_s0, long b_s1) {
        int fullPanels = nc / NR_DB;
        int tailCols = nc % NR_DB;
        if (fullPanels >= 4 && AVAILABLE_CORES >= 2) {
            POOL.invoke(new PackBTask_Arm_Double(src, dst, rowStart, colStart, kc, nc, b_s0, b_s1, 0, fullPanels));
        } else {
            packB_panels_range_Arm_Double(src, dst, rowStart, colStart, kc, b_s0, b_s1, 0, fullPanels);
        }
        if (tailCols > 0) {
            packB_tail_Arm_Double(src, dst, rowStart, colStart, kc, nc, b_s0, b_s1, fullPanels, tailCols);
        }
    }

    public static void packB_panels_range_Arm_Double(MemorySegment src, MemorySegment dst, int rowStart, int colStart, int kc, long b_s0, long b_s1, int pStart, int pEnd) {
        if (b_s1 == 1L) {
            for (int p = pStart; p < pEnd; p++) {
                long dstBase = (long) p * STEP_B_ROW_BYTES_F64 * kc;
                for (int k = 0; k < kc; k++) {
                    long srcOff = ((long)(rowStart + k) * b_s0 + colStart + (long) p * NR_DB) * 8L;
                    long dstOff = dstBase + (long) k * STEP_B_ROW_BYTES_F64;
                    DoubleVector.fromMemorySegment(SPECIESDB, src, srcOff, NATIVE).intoMemorySegment(dst, dstOff, NATIVE);
                    DoubleVector.fromMemorySegment(SPECIESDB, src, srcOff + STRIDE_VEC_BYTES_F64, NATIVE).intoMemorySegment(dst, dstOff + STRIDE_VEC_BYTES_F64, NATIVE);
                }
            }
        } else {
            for (int p = pStart; p < pEnd; p++) {
                long dstBase = (long) p * STEP_B_ROW_BYTES_F64 * kc;
                long colBase = colStart + (long) p * NR_DB;
                for (int k = 0; k < kc; k++) {
                    long rowBase = (long)(rowStart + k) * b_s0;
                    long dstOff = dstBase + (long) k * STEP_B_ROW_BYTES_F64;
                    for (int c = 0; c < NR_DB; c++) {
                        dst.set(VD, dstOff + (long) c * 8L,
                            src.get(VD, (rowBase + (colBase + c) * b_s1) * 8L));
                    }
                }
            }
        }
    }

    public static void packB_tail_Arm_Double(MemorySegment src, MemorySegment dst, int rowStart, int colStart, int kc, int nc, long b_s0, long b_s1, int fullPanels, int tailCols) {
        long dstBase = (long) fullPanels * STEP_B_ROW_BYTES_F64 * kc;
        if (b_s1 == 1L) {
            for (int k = 0; k < kc; k++) {
                long srcOff = ((long)(rowStart + k) * b_s0 + colStart + (long) fullPanels * NR_DB) * 8L;
                long dstOff = dstBase + (long) k * STEP_B_ROW_BYTES_F64;
                for (int c = 0; c < tailCols; c++) {
                    dst.set(VD, dstOff + (long) c * 8L,
                            src.get(VD, srcOff + (long) c * 8L));
                }
                for (int c = tailCols; c < NR_DB; c++) {
                    dst.set(VD, dstOff + (long) c * 8L, 0.0d);
                }
            }
        } else {
            long colBase = colStart + (long) fullPanels * NR_DB;
            for (int k = 0; k < kc; k++) {
                long rowBase = (long)(rowStart + k) * b_s0;
                long dstOff = dstBase + (long) k * STEP_B_ROW_BYTES_F64;
                for (int c = 0; c < tailCols; c++) {
                    dst.set(VD, dstOff + (long) c * 8L,
                        src.get(VD, (rowBase + (colBase + c) * b_s1) * 8L));
                }
                for (int c = tailCols; c < NR_DB; c++) {
                    dst.set(VD, dstOff + (long) c * 8L, 0.0d);
                }
            }
        }
    }

    public static final class PackBTask_Arm_Double extends RecursiveAction {
        final MemorySegment src, dst;
        final int rowStart, colStart, kc, nc;
        final long b_s0, b_s1;
        final int pStart, pEnd;

        PackBTask_Arm_Double(MemorySegment src, MemorySegment dst, int rowStart, int colStart,
                            int kc, int nc, long b_s0, long b_s1, int pStart, int pEnd) {
            this.src = src; this.dst = dst; this.rowStart = rowStart; this.colStart = colStart;
            this.kc = kc; this.nc = nc; this.b_s0 = b_s0; this.b_s1 = b_s1;
            this.pStart = pStart; this.pEnd = pEnd;
        }

        @Override
        protected void compute() {
            int panels = pEnd - pStart;
            if (panels <= 4) {
                packB_panels_range_Arm_Double(src, dst, rowStart, colStart, kc, b_s0, b_s1, pStart, pEnd);
            } else {
                int mid = pStart + panels / 2;
                invokeAll(
                    new PackBTask_Arm_Double(src, dst, rowStart, colStart, kc, nc, b_s0, b_s1, pStart, mid),
                    new PackBTask_Arm_Double(src, dst, rowStart, colStart, kc, nc, b_s0, b_s1, mid, pEnd)
                );
            }
        }
    }

    public static void packA_panel_Aarch_Double(MemorySegment src, MemorySegment dst, int rowStart, int mc, int colStart, int kc, long a_s0, long a_s1) {
        int fullPanels = mc / 8;
        int tailRows = mc % 8;

        if (a_s1 == 1L) {
            for (int p = 0; p < fullPanels; p++) {
                long dstBase = (long) p * 8 * kc * 8L;
                for (int r = 0; r < 8; r++) {
                    long srcRow = (long)(rowStart + p * 8 + r) * a_s0 + colStart;
                    for (int k = 0; k < kc; k++) {
                        double v = src.getAtIndex(VD, srcRow + k);
                        dst.set(VD, dstBase + (long) k * 8 * 8L + (long) r * 8L, v);
                    }
                }
            }
            if (tailRows > 0) {
                long dstBase = (long) fullPanels * 8 * kc * 8L;
                for (int r = 0; r < 8; r++) {
                    if (r < tailRows) {
                        long srcRow = (long)(rowStart + fullPanels * 8 + r) * a_s0 + colStart;
                        for (int k = 0; k < kc; k++) {
                            dst.set(VD,
                                dstBase + (long) k * 8 * 8L + (long) r * 8L,
                                src.getAtIndex(VD, srcRow + k));
                        }
                    } else {
                        for (int k = 0; k < kc; k++) {
                            dst.set(VD,
                                dstBase + (long) k * 8 * 8L + (long) r * 8L, 0.0d);
                        }
                    }
                }
            }
        } else {
            for (int p = 0; p < fullPanels; p++) {
                long dstBase = (long) p * 8 * kc * 8L;
                for (int r = 0; r < 8; r++) {
                    long srcRow = (long)(rowStart + p * 8 + r) * a_s0 + (long) colStart * a_s1;
                    for (int k = 0; k < kc; k++) {
                        double v = src.getAtIndex(VD, srcRow + (long) k * a_s1);
                        dst.set(VD, dstBase + (long) k * 8 * 8L + (long) r * 8L, v);
                    }
                }
            }
            if (tailRows > 0) {
                long dstBase = (long) fullPanels * 8 * kc * 8L;
                for (int r = 0; r < 8; r++) {
                    if (r < tailRows) {
                        long srcRow = (long)(rowStart + fullPanels * 8 + r) * a_s0 + (long) colStart * a_s1;
                        for (int k = 0; k < kc; k++) {
                            dst.set(VD,
                                dstBase + (long) k * 8 * 8L + (long) r * 8L,
                                src.getAtIndex(VD, srcRow + (long) k * a_s1));
                        }
                    } else {
                        for (int k = 0; k < kc; k++) {
                            dst.set(VD,
                                dstBase + (long) k * 8 * 8L + (long) r * 8L, 0.0d);
                        }
                    }
                }
            }
        }
    }

    public static void packB_panel_Aarch_Double(MemorySegment src, MemorySegment dst, int rowStart, int colStart, int kc, int nc, long b_s0, long b_s1) {
        int fullPanels = nc / 12;
        int tailCols = nc % 12;
        if (fullPanels >= 4 && AVAILABLE_CORES >= 2) {
            POOL.invoke(new PackBTask_Aarch_Double(src, dst, rowStart, colStart, kc, nc, b_s0, b_s1, 0, fullPanels));
        } else {
            packB_panels_range_Aarch_Double(src, dst, rowStart, colStart, kc, b_s0, b_s1, 0, fullPanels);
        }
        if (tailCols > 0) {
            packB_tail_Aarch_Double(src, dst, rowStart, colStart, kc, nc, b_s0, b_s1, fullPanels, tailCols);
        }
    }

    public static void packB_panels_range_Aarch_Double(MemorySegment src, MemorySegment dst, int rowStart, int colStart, int kc, long b_s0, long b_s1, int pStart, int pEnd) {
        if (b_s1 == 1L) {
            for (int p = pStart; p < pEnd; p++) {
                long dstBase = (long) p * 16 * kc * 8L;
                for (int k = 0; k < kc; k++) {
                    long srcOff = ((long)(rowStart + k) * b_s0 + colStart + (long) p * 12) * 8L;
                    long dstOff = dstBase + (long) k * 16 * 8L;
                    MemorySegment.copy(src, srcOff, dst, dstOff, 12L * 8L);
                    dst.set(VD, dstOff + 12 * 8L, 0.0d);
                    dst.set(VD, dstOff + 13 * 8L, 0.0d);
                    dst.set(VD, dstOff + 14 * 8L, 0.0d);
                    dst.set(VD, dstOff + 15 * 8L, 0.0d);
                }
            }
        } else {
            for (int p = pStart; p < pEnd; p++) {
                long dstBase = (long) p * 16 * kc * 8L;
                long colBase = colStart + (long) p * 12;
                for (int k = 0; k < kc; k++) {
                    long rowBase = (long)(rowStart + k) * b_s0;
                    long dstOff = dstBase + (long) k * 16 * 8L;
                    for (int c = 0; c < 12; c++) {
                        dst.set(VD, dstOff + (long) c * 8L,
                            src.get(VD, (rowBase + (colBase + c) * b_s1) * 8L));
                    }
                    dst.set(VD, dstOff + 12 * 8L, 0.0d);
                    dst.set(VD, dstOff + 13 * 8L, 0.0d);
                    dst.set(VD, dstOff + 14 * 8L, 0.0d);
                    dst.set(VD, dstOff + 15 * 8L, 0.0d);
                }
            }
        }
    }

    public static void packB_tail_Aarch_Double(MemorySegment src, MemorySegment dst, int rowStart, int colStart, int kc, int nc, long b_s0, long b_s1, int fullPanels, int tailCols) {
        long dstBase = (long) fullPanels * 16 * kc * 8L;
        if (b_s1 == 1L) {
            for (int k = 0; k < kc; k++) {
                long srcOff = ((long)(rowStart + k) * b_s0 + colStart + (long) fullPanels * 12) * 8L;
                long dstOff = dstBase + (long) k * 16 * 8L;
                MemorySegment.copy(src, srcOff, dst, dstOff, (long) tailCols * 8L);
                for (int c = tailCols; c < 16; c++) {
                    dst.set(VD, dstOff + (long) c * 8L, 0.0d);
                }
            }
        } else {
            long colBase = colStart + (long) fullPanels * 12;
            for (int k = 0; k < kc; k++) {
                long rowBase = (long)(rowStart + k) * b_s0;
                long dstOff = dstBase + (long) k * 16 * 8L;
                for (int c = 0; c < tailCols; c++) {
                    dst.set(VD, dstOff + (long) c * 8L,
                        src.get(VD, (rowBase + (colBase + c) * b_s1) * 8L));
                }
                for (int c = tailCols; c < 16; c++) {
                    dst.set(VD, dstOff + (long) c * 8L, 0.0d);
                }
            }
        }
    }

    public static final class PackBTask_Aarch_Double extends RecursiveAction {
        final MemorySegment src, dst;
        final int rowStart, colStart, kc, nc;
        final long b_s0, b_s1;
        final int pStart, pEnd;

        PackBTask_Aarch_Double(MemorySegment src, MemorySegment dst, int rowStart, int colStart,
                              int kc, int nc, long b_s0, long b_s1, int pStart, int pEnd) {
            this.src = src; this.dst = dst; this.rowStart = rowStart; this.colStart = colStart;
            this.kc = kc; this.nc = nc; this.b_s0 = b_s0; this.b_s1 = b_s1;
            this.pStart = pStart; this.pEnd = pEnd;
        }

        @Override
        protected void compute() {
            int panels = pEnd - pStart;
            if (panels <= 4) {
                packB_panels_range_Aarch_Double(src, dst, rowStart, colStart, kc, b_s0, b_s1, pStart, pEnd);
            } else {
                int mid = pStart + panels / 2;
                invokeAll(
                    new PackBTask_Aarch_Double(src, dst, rowStart, colStart, kc, nc, b_s0, b_s1, pStart, mid),
                    new PackBTask_Aarch_Double(src, dst, rowStart, colStart, kc, nc, b_s0, b_s1, mid, pEnd)
                );
            }
        }
    }
    public static void packA_panel_Arm_Int(MemorySegment src, MemorySegment dst, int rowStart, int mc, int colStart, int kc, long a_s0, long a_s1) {
        int fullPanels = mc / MR;
        int tailRows = mc % MR;

        if (a_s1 == 1L) {
            for (int p = 0; p < fullPanels; p++) {
                long dstBase = (long) p * STEP_A_ROW_BYTES_I32 * kc;
                long r0 = (long)(rowStart + p * MR + 0) * a_s0 + colStart;
                long r1 = (long)(rowStart + p * MR + 1) * a_s0 + colStart;
                long r2 = (long)(rowStart + p * MR + 2) * a_s0 + colStart;
                long r3 = (long)(rowStart + p * MR + 3) * a_s0 + colStart;
                long r4 = (long)(rowStart + p * MR + 4) * a_s0 + colStart;
                long r5 = (long)(rowStart + p * MR + 5) * a_s0 + colStart;

                int k = 0;
                for (; k <= kc - 4; k += 4) {
                    long dOff0 = dstBase + (long) k * STEP_A_ROW_BYTES_I32;
                    dst.set(VI, dOff0,                    src.getAtIndex(VI, r0 + k));
                    dst.set(VI, dOff0 + 1L * 4L, src.getAtIndex(VI, r1 + k));
                    dst.set(VI, dOff0 + 2L * 4L, src.getAtIndex(VI, r2 + k));
                    dst.set(VI, dOff0 + 3L * 4L, src.getAtIndex(VI, r3 + k));
                    dst.set(VI, dOff0 + 4L * 4L, src.getAtIndex(VI, r4 + k));
                    dst.set(VI, dOff0 + 5L * 4L, src.getAtIndex(VI, r5 + k));

                    long dOff1 = dOff0 + STEP_A_ROW_BYTES_I32;
                    dst.set(VI, dOff1,                    src.getAtIndex(VI, r0 + k + 1));
                    dst.set(VI, dOff1 + 1L * 4L, src.getAtIndex(VI, r1 + k + 1));
                    dst.set(VI, dOff1 + 2L * 4L, src.getAtIndex(VI, r2 + k + 1));
                    dst.set(VI, dOff1 + 3L * 4L, src.getAtIndex(VI, r3 + k + 1));
                    dst.set(VI, dOff1 + 4L * 4L, src.getAtIndex(VI, r4 + k + 1));
                    dst.set(VI, dOff1 + 5L * 4L, src.getAtIndex(VI, r5 + k + 1));

                    long dOff2 = dOff1 + STEP_A_ROW_BYTES_I32;
                    dst.set(VI, dOff2,                    src.getAtIndex(VI, r0 + k + 2));
                    dst.set(VI, dOff2 + 1L * 4L, src.getAtIndex(VI, r1 + k + 2));
                    dst.set(VI, dOff2 + 2L * 4L, src.getAtIndex(VI, r2 + k + 2));
                    dst.set(VI, dOff2 + 3L * 4L, src.getAtIndex(VI, r3 + k + 2));
                    dst.set(VI, dOff2 + 4L * 4L, src.getAtIndex(VI, r4 + k + 2));
                    dst.set(VI, dOff2 + 5L * 4L, src.getAtIndex(VI, r5 + k + 2));

                    long dOff3 = dOff2 + STEP_A_ROW_BYTES_I32;
                    dst.set(VI, dOff3,                    src.getAtIndex(VI, r0 + k + 3));
                    dst.set(VI, dOff3 + 1L * 4L, src.getAtIndex(VI, r1 + k + 3));
                    dst.set(VI, dOff3 + 2L * 4L, src.getAtIndex(VI, r2 + k + 3));
                    dst.set(VI, dOff3 + 3L * 4L, src.getAtIndex(VI, r3 + k + 3));
                    dst.set(VI, dOff3 + 4L * 4L, src.getAtIndex(VI, r4 + k + 3));
                    dst.set(VI, dOff3 + 5L * 4L, src.getAtIndex(VI, r5 + k + 3));
                }
                for (; k < kc; k++) {
                    long dOff = dstBase + (long) k * STEP_A_ROW_BYTES_I32;
                    dst.set(VI, dOff,                    src.getAtIndex(VI, r0 + k));
                    dst.set(VI, dOff + 1L * 4L, src.getAtIndex(VI, r1 + k));
                    dst.set(VI, dOff + 2L * 4L, src.getAtIndex(VI, r2 + k));
                    dst.set(VI, dOff + 3L * 4L, src.getAtIndex(VI, r3 + k));
                    dst.set(VI, dOff + 4L * 4L, src.getAtIndex(VI, r4 + k));
                    dst.set(VI, dOff + 5L * 4L, src.getAtIndex(VI, r5 + k));
                }
            }
            if (tailRows > 0) {
                long dstBase = (long) fullPanels * STEP_A_ROW_BYTES_I32 * kc;
                for (int r = 0; r < MR; r++) {
                    if (r < tailRows) {
                        long srcRow = (long)(rowStart + fullPanels * MR + r) * a_s0 + colStart;
                        for (int k = 0; k < kc; k++) {
                            dst.set(VI,
                                dstBase + (long) k * STEP_A_ROW_BYTES_I32 + (long) r * 4L,
                                src.getAtIndex(VI, srcRow + k));
                        }
                    } else {
                        for (int k = 0; k < kc; k++) {
                            dst.set(VI,
                                dstBase + (long) k * STEP_A_ROW_BYTES_I32 + (long) r * 4L, 0);
                        }
                    }
                }
            }
        } else {
            for (int p = 0; p < fullPanels; p++) {
                long dstBase = (long) p * STEP_A_ROW_BYTES_I32 * kc;
                long r0 = (long)(rowStart + p * MR + 0) * a_s0 + (long) colStart * a_s1;
                long r1 = (long)(rowStart + p * MR + 1) * a_s0 + (long) colStart * a_s1;
                long r2 = (long)(rowStart + p * MR + 2) * a_s0 + (long) colStart * a_s1;
                long r3 = (long)(rowStart + p * MR + 3) * a_s0 + (long) colStart * a_s1;
                long r4 = (long)(rowStart + p * MR + 4) * a_s0 + (long) colStart * a_s1;
                long r5 = (long)(rowStart + p * MR + 5) * a_s0 + (long) colStart * a_s1;

                for (int k = 0; k < kc; k++) {
                    long dOff = dstBase + (long) k * STEP_A_ROW_BYTES_I32;
                    long kOff = (long) k * a_s1;
                    dst.set(VI, dOff,                    src.getAtIndex(VI, r0 + kOff));
                    dst.set(VI, dOff + 1L * 4L, src.getAtIndex(VI, r1 + kOff));
                    dst.set(VI, dOff + 2L * 4L, src.getAtIndex(VI, r2 + kOff));
                    dst.set(VI, dOff + 3L * 4L, src.getAtIndex(VI, r3 + kOff));
                    dst.set(VI, dOff + 4L * 4L, src.getAtIndex(VI, r4 + kOff));
                    dst.set(VI, dOff + 5L * 4L, src.getAtIndex(VI, r5 + kOff));
                }
            }
            if (tailRows > 0) {
                long dstBase = (long) fullPanels * STEP_A_ROW_BYTES_I32 * kc;
                for (int r = 0; r < MR; r++) {
                    if (r < tailRows) {
                        long srcRow = (long)(rowStart + fullPanels * MR + r) * a_s0 + (long) colStart * a_s1;
                        for (int k = 0; k < kc; k++) {
                            dst.set(VI,
                                dstBase + (long) k * STEP_A_ROW_BYTES_I32 + (long) r * 4L,
                                src.getAtIndex(VI, srcRow + (long) k * a_s1));
                        }
                    } else {
                        for (int k = 0; k < kc; k++) {
                            dst.set(VI,
                                dstBase + (long) k * STEP_A_ROW_BYTES_I32 + (long) r * 4L, 0);
                        }
                    }
                }
            }
        }
    }

    public static void packB_panel_Arm_Int(MemorySegment src, MemorySegment dst, int rowStart, int colStart, int kc, int nc, long b_s0, long b_s1) {
        int fullPanels = nc / NR_INT;
        int tailCols = nc % NR_INT;
        if (fullPanels >= 4 && AVAILABLE_CORES >= 2) {
            POOL.invoke(new PackBTask_Arm_Int(src, dst, rowStart, colStart, kc, nc, b_s0, b_s1, 0, fullPanels));
        } else {
            packB_panels_range_Arm_Int(src, dst, rowStart, colStart, kc, b_s0, b_s1, 0, fullPanels);
        }
        if (tailCols > 0) {
            packB_tail_Arm_Int(src, dst, rowStart, colStart, kc, nc, b_s0, b_s1, fullPanels, tailCols);
        }
    }

    public static void packB_panels_range_Arm_Int(MemorySegment src, MemorySegment dst, int rowStart, int colStart, int kc, long b_s0, long b_s1, int pStart, int pEnd) {
        if (b_s1 == 1L) {
            for (int p = pStart; p < pEnd; p++) {
                long dstBase = (long) p * STEP_B_ROW_BYTES_I32 * kc;
                for (int k = 0; k < kc; k++) {
                    long srcOff = ((long)(rowStart + k) * b_s0 + colStart + (long) p * NR_INT) * 4L;
                    long dstOff = dstBase + (long) k * STEP_B_ROW_BYTES_I32;
                    IntVector.fromMemorySegment(SPECIESINT, src, srcOff, NATIVE).intoMemorySegment(dst, dstOff, NATIVE);
                    IntVector.fromMemorySegment(SPECIESINT, src, srcOff + STRIDE_VEC_BYTES_I32, NATIVE).intoMemorySegment(dst, dstOff + STRIDE_VEC_BYTES_I32, NATIVE);
                }
            }
        } else {
            for (int p = pStart; p < pEnd; p++) {
                long dstBase = (long) p * STEP_B_ROW_BYTES_I32 * kc;
                long colBase = colStart + (long) p * NR_INT;
                for (int k = 0; k < kc; k++) {
                    long rowBase = (long)(rowStart + k) * b_s0;
                    long dstOff = dstBase + (long) k * STEP_B_ROW_BYTES_I32;
                    for (int c = 0; c < NR_INT; c++) {
                        dst.set(VI, dstOff + (long) c * 4L,
                            src.get(VI, (rowBase + (colBase + c) * b_s1) * 4L));
                    }
                }
            }
        }
    }

    public static void packB_tail_Arm_Int(MemorySegment src, MemorySegment dst, int rowStart, int colStart, int kc, int nc, long b_s0, long b_s1, int fullPanels, int tailCols) {
        long dstBase = (long) fullPanels * STEP_B_ROW_BYTES_I32 * kc;
        if (b_s1 == 1L) {
            for (int k = 0; k < kc; k++) {
                long srcOff = ((long)(rowStart + k) * b_s0 + colStart + (long) fullPanels * NR_INT) * 4L;
                long dstOff = dstBase + (long) k * STEP_B_ROW_BYTES_I32;
                for (int c = 0; c < tailCols; c++) {
                    dst.set(VI, dstOff + (long) c * 4L,
                            src.get(VI, srcOff + (long) c * 4L));
                }
                for (int c = tailCols; c < NR_INT; c++) {
                    dst.set(VI, dstOff + (long) c * 4L, 0);
                }
            }
        } else {
            long colBase = colStart + (long) fullPanels * NR_INT;
            for (int k = 0; k < kc; k++) {
                long rowBase = (long)(rowStart + k) * b_s0;
                long dstOff = dstBase + (long) k * STEP_B_ROW_BYTES_I32;
                for (int c = 0; c < tailCols; c++) {
                    dst.set(VI, dstOff + (long) c * 4L,
                        src.get(VI, (rowBase + (colBase + c) * b_s1) * 4L));
                }
                for (int c = tailCols; c < NR_INT; c++) {
                    dst.set(VI, dstOff + (long) c * 4L, 0);
                }
            }
        }
    }

    public static final class PackBTask_Arm_Int extends RecursiveAction {
        final MemorySegment src, dst;
        final int rowStart, colStart, kc, nc;
        final long b_s0, b_s1;
        final int pStart, pEnd;

        PackBTask_Arm_Int(MemorySegment src, MemorySegment dst, int rowStart, int colStart,
                            int kc, int nc, long b_s0, long b_s1, int pStart, int pEnd) {
            this.src = src; this.dst = dst; this.rowStart = rowStart; this.colStart = colStart;
            this.kc = kc; this.nc = nc; this.b_s0 = b_s0; this.b_s1 = b_s1;
            this.pStart = pStart; this.pEnd = pEnd;
        }

        @Override
        protected void compute() {
            int panels = pEnd - pStart;
            if (panels <= 4) {
                packB_panels_range_Arm_Int(src, dst, rowStart, colStart, kc, b_s0, b_s1, pStart, pEnd);
            } else {
                int mid = pStart + panels / 2;
                invokeAll(
                    new PackBTask_Arm_Int(src, dst, rowStart, colStart, kc, nc, b_s0, b_s1, pStart, mid),
                    new PackBTask_Arm_Int(src, dst, rowStart, colStart, kc, nc, b_s0, b_s1, mid, pEnd)
                );
            }
        }
    }

    public static void packA_panel_Aarch_Int(MemorySegment src, MemorySegment dst, int rowStart, int mc, int colStart, int kc, long a_s0, long a_s1) {
        int fullPanels = mc / 8;
        int tailRows = mc % 8;

        if (a_s1 == 1L) {
            for (int p = 0; p < fullPanels; p++) {
                long dstBase = (long) p * 8 * kc * 4L;
                for (int r = 0; r < 8; r++) {
                    long srcRow = (long)(rowStart + p * 8 + r) * a_s0 + colStart;
                    for (int k = 0; k < kc; k++) {
                        int v = src.getAtIndex(VI, srcRow + k);
                        dst.set(VI, dstBase + (long) k * 8 * 4L + (long) r * 4L, v);
                    }
                }
            }
            if (tailRows > 0) {
                long dstBase = (long) fullPanels * 8 * kc * 4L;
                for (int r = 0; r < 8; r++) {
                    if (r < tailRows) {
                        long srcRow = (long)(rowStart + fullPanels * 8 + r) * a_s0 + colStart;
                        for (int k = 0; k < kc; k++) {
                            dst.set(VI,
                                dstBase + (long) k * 8 * 4L + (long) r * 4L,
                                src.getAtIndex(VI, srcRow + k));
                        }
                    } else {
                        for (int k = 0; k < kc; k++) {
                            dst.set(VI,
                                dstBase + (long) k * 8 * 4L + (long) r * 4L, 0);
                        }
                    }
                }
            }
        } else {
            for (int p = 0; p < fullPanels; p++) {
                long dstBase = (long) p * 8 * kc * 4L;
                for (int r = 0; r < 8; r++) {
                    long srcRow = (long)(rowStart + p * 8 + r) * a_s0 + (long) colStart * a_s1;
                    for (int k = 0; k < kc; k++) {
                        int v = src.getAtIndex(VI, srcRow + (long) k * a_s1);
                        dst.set(VI, dstBase + (long) k * 8 * 4L + (long) r * 4L, v);
                    }
                }
            }
            if (tailRows > 0) {
                long dstBase = (long) fullPanels * 8 * kc * 4L;
                for (int r = 0; r < 8; r++) {
                    if (r < tailRows) {
                        long srcRow = (long)(rowStart + fullPanels * 8 + r) * a_s0 + (long) colStart * a_s1;
                        for (int k = 0; k < kc; k++) {
                            dst.set(VI,
                                dstBase + (long) k * 8 * 4L + (long) r * 4L,
                                src.getAtIndex(VI, srcRow + (long) k * a_s1));
                        }
                    } else {
                        for (int k = 0; k < kc; k++) {
                            dst.set(VI,
                                dstBase + (long) k * 8 * 4L + (long) r * 4L, 0);
                        }
                    }
                }
            }
        }
    }

    public static void packB_panel_Aarch_Int(MemorySegment src, MemorySegment dst, int rowStart, int colStart, int kc, int nc, long b_s0, long b_s1) {
        int fullPanels = nc / 12;
        int tailCols = nc % 12;
        if (fullPanels >= 4 && AVAILABLE_CORES >= 2) {
            POOL.invoke(new PackBTask_Aarch_Int(src, dst, rowStart, colStart, kc, nc, b_s0, b_s1, 0, fullPanels));
        } else {
            packB_panels_range_Aarch_Int(src, dst, rowStart, colStart, kc, b_s0, b_s1, 0, fullPanels);
        }
        if (tailCols > 0) {
            packB_tail_Aarch_Int(src, dst, rowStart, colStart, kc, nc, b_s0, b_s1, fullPanels, tailCols);
        }
    }

    public static void packB_panels_range_Aarch_Int(MemorySegment src, MemorySegment dst, int rowStart, int colStart, int kc, long b_s0, long b_s1, int pStart, int pEnd) {
        if (b_s1 == 1L) {
            for (int p = pStart; p < pEnd; p++) {
                long dstBase = (long) p * 16 * kc * 4L;
                for (int k = 0; k < kc; k++) {
                    long srcOff = ((long)(rowStart + k) * b_s0 + colStart + (long) p * 12) * 4L;
                    long dstOff = dstBase + (long) k * 16 * 4L;
                    MemorySegment.copy(src, srcOff, dst, dstOff, 12L * 4L);
                    dst.set(VI, dstOff + 12 * 4L, 0);
                    dst.set(VI, dstOff + 13 * 4L, 0);
                    dst.set(VI, dstOff + 14 * 4L, 0);
                    dst.set(VI, dstOff + 15 * 4L, 0);
                }
            }
        } else {
            for (int p = pStart; p < pEnd; p++) {
                long dstBase = (long) p * 16 * kc * 4L;
                long colBase = colStart + (long) p * 12;
                for (int k = 0; k < kc; k++) {
                    long rowBase = (long)(rowStart + k) * b_s0;
                    long dstOff = dstBase + (long) k * 16 * 4L;
                    for (int c = 0; c < 12; c++) {
                        dst.set(VI, dstOff + (long) c * 4L,
                            src.get(VI, (rowBase + (colBase + c) * b_s1) * 4L));
                    }
                    dst.set(VI, dstOff + 12 * 4L, 0);
                    dst.set(VI, dstOff + 13 * 4L, 0);
                    dst.set(VI, dstOff + 14 * 4L, 0);
                    dst.set(VI, dstOff + 15 * 4L, 0);
                }
            }
        }
    }

    public static void packB_tail_Aarch_Int(MemorySegment src, MemorySegment dst, int rowStart, int colStart, int kc, int nc, long b_s0, long b_s1, int fullPanels, int tailCols) {
        long dstBase = (long) fullPanels * 16 * kc * 4L;
        if (b_s1 == 1L) {
            for (int k = 0; k < kc; k++) {
                long srcOff = ((long)(rowStart + k) * b_s0 + colStart + (long) fullPanels * 12) * 4L;
                long dstOff = dstBase + (long) k * 16 * 4L;
                MemorySegment.copy(src, srcOff, dst, dstOff, (long) tailCols * 4L);
                for (int c = tailCols; c < 16; c++) {
                    dst.set(VI, dstOff + (long) c * 4L, 0);
                }
            }
        } else {
            long colBase = colStart + (long) fullPanels * 12;
            for (int k = 0; k < kc; k++) {
                long rowBase = (long)(rowStart + k) * b_s0;
                long dstOff = dstBase + (long) k * 16 * 4L;
                for (int c = 0; c < tailCols; c++) {
                    dst.set(VI, dstOff + (long) c * 4L,
                        src.get(VI, (rowBase + (colBase + c) * b_s1) * 4L));
                }
                for (int c = tailCols; c < 16; c++) {
                    dst.set(VI, dstOff + (long) c * 4L, 0);
                }
            }
        }
    }

    public static final class PackBTask_Aarch_Int extends RecursiveAction {
        final MemorySegment src, dst;
        final int rowStart, colStart, kc, nc;
        final long b_s0, b_s1;
        final int pStart, pEnd;

        PackBTask_Aarch_Int(MemorySegment src, MemorySegment dst, int rowStart, int colStart,
                              int kc, int nc, long b_s0, long b_s1, int pStart, int pEnd) {
            this.src = src; this.dst = dst; this.rowStart = rowStart; this.colStart = colStart;
            this.kc = kc; this.nc = nc; this.b_s0 = b_s0; this.b_s1 = b_s1;
            this.pStart = pStart; this.pEnd = pEnd;
        }

        @Override
        protected void compute() {
            int panels = pEnd - pStart;
            if (panels <= 4) {
                packB_panels_range_Aarch_Int(src, dst, rowStart, colStart, kc, b_s0, b_s1, pStart, pEnd);
            } else {
                int mid = pStart + panels / 2;
                invokeAll(
                    new PackBTask_Aarch_Int(src, dst, rowStart, colStart, kc, nc, b_s0, b_s1, pStart, mid),
                    new PackBTask_Aarch_Int(src, dst, rowStart, colStart, kc, nc, b_s0, b_s1, mid, pEnd)
                );
            }
        }
    }


    // Backward-compatibility aliases for tests and external callers
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