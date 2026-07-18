package jnum.jnumops;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorSpecies;
import jdk.incubator.vector.VectorOperators;
import jnum.NDArray;

public class MatMulOps {

    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Integer> SPECIESINT = IntVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Double> SPECIESDB = DoubleVector.SPECIES_PREFERRED;
    
    private static final ForkJoinPool POOL = new ForkJoinPool(Runtime.getRuntime().availableProcessors());
    private static final int THRESHOLD = 64;
    private static final ByteOrder NATIVE= ByteOrder.nativeOrder();

    private static final int MC = 96;
    private static final int KC = 256;
    private static final int NC_ARM = 1024;
    private static final int NC_AARCH = 1008;

    // non instanciable utility class

    private MatMulOps() {
        throw new AssertionError();
    }


    public static NDArray matmulFloat(NDArray a, NDArray b, NDArray resArray) {
        int n = a.internalShapeUnsafe()[0]; 
        int m = a.internalShapeUnsafe()[1]; 
        int p = b.internalShapeUnsafe()[1];
        
        try (Arena arena = Arena.ofShared()) {
            MemorySegment memA = a.isContiguous() ? a.getData() : a.contiguous(arena).getData();
            MemorySegment memB = b.isContiguous() ? b.getData() : b.contiguous(arena).getData();
            MemorySegment memC = resArray.getData();

            if ((long) n * m * p >= 2_000_000_000L) {
                String arch = System.getProperty("os.arch").toLowerCase();
                if (arch.contains("aarch")) {
                    blisAarchMacro_Float(memA, memB, memC, n, m, p, arena);
                } else {
                    blisArmMacro_Float(memA, memB, memC, n, m, p, arena);
                }
            } else {
                MemorySegment memB_T = fastTranspose2D_Float(memB, arena, m, p);
                POOL.invoke(new AVX2_Float(memA, memB_T, memC, n, m, p, 0, n));
            }
        }
        return resArray;
    }

    private static final ThreadLocal<MemorySegment> tlPackedA_Arm_Float =
        ThreadLocal.withInitial(() -> Arena.ofAuto().allocate((long) MC * KC * 4L, 64));
    private static final ThreadLocal<MemorySegment> tlPackedA_Aarch_Float =
        ThreadLocal.withInitial(() -> Arena.ofAuto().allocate((long) MC * KC * 4L, 64));

    private static void blisArmMacro_Float(MemorySegment A, MemorySegment B, MemorySegment C, int n, int m, int p, Arena arena) {
        C.fill((byte) 0);
        MemorySegment pB = arena.allocate((long) KC * NC_ARM * 4L, 64);
        for (int jc = 0; jc < p; jc += NC_ARM) {
            int nc = Math.min(NC_ARM, p - jc);
            for (int pc = 0; pc < m; pc += KC) {
                int kc = Math.min(KC, m - pc);
                packB_panel_Arm_Float(B, pB, pc, jc, kc, nc, p);
                POOL.invoke(new GEBPTask_Arm_Float(A, pB, C, n, m, p, 0, n, pc, kc, jc, nc));
            }
        }
    }

    private static void blisAarchMacro_Float(MemorySegment A, MemorySegment B, MemorySegment C, int n, int m, int p, Arena arena) {
        C.fill((byte) 0);
        MemorySegment pB = arena.allocate((long) KC * NC_AARCH * 4L, 64);
        for (int jc = 0; jc < p; jc += NC_AARCH) {
            int nc = Math.min(NC_AARCH, p - jc);
            for (int pc = 0; pc < m; pc += KC) {
                int kc = Math.min(KC, m - pc);
                packB_panel_Aarch_Float(B, pB, pc, jc, kc, nc, p);
                POOL.invoke(new GEBPTask_Aarch_Float(A, pB, C, n, m, p, 0, n, pc, kc, jc, nc));
            }
        }
    }

    static void packA_panel_Arm_Float(MemorySegment src, MemorySegment dst, int rowStart, int mc, int colStart, int kc, int m) {
        int fullPanels = mc / 6; int tailRows = mc % 6;
        for (int panel = 0; panel < fullPanels; panel++) {
            long dstBase = (long) panel * 6 * kc * 4L;
            long r0 = (long)(rowStart + panel * 6 + 0) * m + colStart;
            long r1 = (long)(rowStart + panel * 6 + 1) * m + colStart;
            long r2 = (long)(rowStart + panel * 6 + 2) * m + colStart;
            long r3 = (long)(rowStart + panel * 6 + 3) * m + colStart;
            long r4 = (long)(rowStart + panel * 6 + 4) * m + colStart;
            long r5 = (long)(rowStart + panel * 6 + 5) * m + colStart;
            
            int k = 0;
            for (; k <= kc - 4; k += 4) {
                long dOff = dstBase + (long) k * 6 * 4L;
                dst.set(ValueLayout.JAVA_FLOAT, dOff, src.getAtIndex(ValueLayout.JAVA_FLOAT, r0 + k));
                dst.set(ValueLayout.JAVA_FLOAT, dOff + 4L, src.getAtIndex(ValueLayout.JAVA_FLOAT, r1 + k));
                dst.set(ValueLayout.JAVA_FLOAT, dOff + 2 * 4L, src.getAtIndex(ValueLayout.JAVA_FLOAT, r2 + k));
                dst.set(ValueLayout.JAVA_FLOAT, dOff + 3 * 4L, src.getAtIndex(ValueLayout.JAVA_FLOAT, r3 + k));
                dst.set(ValueLayout.JAVA_FLOAT, dOff + 4 * 4L, src.getAtIndex(ValueLayout.JAVA_FLOAT, r4 + k));
                dst.set(ValueLayout.JAVA_FLOAT, dOff + 5 * 4L, src.getAtIndex(ValueLayout.JAVA_FLOAT, r5 + k));
                
                long dOff1 = dOff + 6 * 4L;
                dst.set(ValueLayout.JAVA_FLOAT, dOff1, src.getAtIndex(ValueLayout.JAVA_FLOAT, r0 + k + 1));
                dst.set(ValueLayout.JAVA_FLOAT, dOff1 + 4L, src.getAtIndex(ValueLayout.JAVA_FLOAT, r1 + k + 1));
                dst.set(ValueLayout.JAVA_FLOAT, dOff1 + 2 * 4L, src.getAtIndex(ValueLayout.JAVA_FLOAT, r2 + k + 1));
                dst.set(ValueLayout.JAVA_FLOAT, dOff1 + 3 * 4L, src.getAtIndex(ValueLayout.JAVA_FLOAT, r3 + k + 1));
                dst.set(ValueLayout.JAVA_FLOAT, dOff1 + 4 * 4L, src.getAtIndex(ValueLayout.JAVA_FLOAT, r4 + k + 1));
                dst.set(ValueLayout.JAVA_FLOAT, dOff1 + 5 * 4L, src.getAtIndex(ValueLayout.JAVA_FLOAT, r5 + k + 1));

                long dOff2 = dOff1 + 6 * 4L;
                dst.set(ValueLayout.JAVA_FLOAT, dOff2, src.getAtIndex(ValueLayout.JAVA_FLOAT, r0 + k + 2));
                dst.set(ValueLayout.JAVA_FLOAT, dOff2 + 4L, src.getAtIndex(ValueLayout.JAVA_FLOAT, r1 + k + 2));
                dst.set(ValueLayout.JAVA_FLOAT, dOff2 + 2 * 4L, src.getAtIndex(ValueLayout.JAVA_FLOAT, r2 + k + 2));
                dst.set(ValueLayout.JAVA_FLOAT, dOff2 + 3 * 4L, src.getAtIndex(ValueLayout.JAVA_FLOAT, r3 + k + 2));
                dst.set(ValueLayout.JAVA_FLOAT, dOff2 + 4 * 4L, src.getAtIndex(ValueLayout.JAVA_FLOAT, r4 + k + 2));
                dst.set(ValueLayout.JAVA_FLOAT, dOff2 + 5 * 4L, src.getAtIndex(ValueLayout.JAVA_FLOAT, r5 + k + 2));

                long dOff3 = dOff2 + 6 * 4L;
                dst.set(ValueLayout.JAVA_FLOAT, dOff3, src.getAtIndex(ValueLayout.JAVA_FLOAT, r0 + k + 3));
                dst.set(ValueLayout.JAVA_FLOAT, dOff3 + 4L, src.getAtIndex(ValueLayout.JAVA_FLOAT, r1 + k + 3));
                dst.set(ValueLayout.JAVA_FLOAT, dOff3 + 2 * 4L, src.getAtIndex(ValueLayout.JAVA_FLOAT, r2 + k + 3));
                dst.set(ValueLayout.JAVA_FLOAT, dOff3 + 3 * 4L, src.getAtIndex(ValueLayout.JAVA_FLOAT, r3 + k + 3));
                dst.set(ValueLayout.JAVA_FLOAT, dOff3 + 4 * 4L, src.getAtIndex(ValueLayout.JAVA_FLOAT, r4 + k + 3));
                dst.set(ValueLayout.JAVA_FLOAT, dOff3 + 5 * 4L, src.getAtIndex(ValueLayout.JAVA_FLOAT, r5 + k + 3));
            }
            for (; k < kc; k++) {
                long dOff = dstBase + (long) k * 6 * 4L;
                dst.set(ValueLayout.JAVA_FLOAT, dOff, src.getAtIndex(ValueLayout.JAVA_FLOAT, r0 + k));
                dst.set(ValueLayout.JAVA_FLOAT, dOff + 4L, src.getAtIndex(ValueLayout.JAVA_FLOAT, r1 + k));
                dst.set(ValueLayout.JAVA_FLOAT, dOff + 2 * 4L, src.getAtIndex(ValueLayout.JAVA_FLOAT, r2 + k));
                dst.set(ValueLayout.JAVA_FLOAT, dOff + 3 * 4L, src.getAtIndex(ValueLayout.JAVA_FLOAT, r3 + k));
                dst.set(ValueLayout.JAVA_FLOAT, dOff + 4 * 4L, src.getAtIndex(ValueLayout.JAVA_FLOAT, r4 + k));
                dst.set(ValueLayout.JAVA_FLOAT, dOff + 5 * 4L, src.getAtIndex(ValueLayout.JAVA_FLOAT, r5 + k));
            }
        }
        if (tailRows > 0) {
            long dstBase = (long) fullPanels * 6 * kc * 4L;
            for (int r = 0; r < 6; r++) {
                if (r < tailRows) {
                    long srcRow = (long)(rowStart + fullPanels * 6 + r) * m + colStart;
                    for (int k = 0; k < kc; k++) {
                        dst.set(ValueLayout.JAVA_FLOAT, dstBase + (long) k * 6 * 4L + (long) r * 4L, src.getAtIndex(ValueLayout.JAVA_FLOAT, srcRow + k));
                    }
                } else {
                    for (int k = 0; k < kc; k++) {
                        dst.set(ValueLayout.JAVA_FLOAT, dstBase + (long) k * 6 * 4L + (long) r * 4L, 0f);
                    }
                }
            }
        }
    }

    static void packB_panel_Arm_Float(MemorySegment src, MemorySegment dst, int rowStart, int colStart, int kc, int nc, int p) {
        int fullPanels = nc / 16; int tailCols = nc % 16;
        for (int panel = 0; panel < fullPanels; panel++) {
            long dstBase = (long) panel * 16 * kc * 4L;
            for (int k = 0; k < kc; k++) {
                long srcOff = ((long)(rowStart + k) * p + colStart + (long) panel * 16) * 4L;
                MemorySegment.copy(src, srcOff, dst, dstBase + (long) k * 16 * 4L, 16L * 4L);
            }
        }
        if (tailCols > 0) {
            long dstBase = (long) fullPanels * 16 * kc * 4L;
            for (int k = 0; k < kc; k++) {
                long srcOff = ((long)(rowStart + k) * p + colStart + (long) fullPanels * 16) * 4L;
                long dstOff = dstBase + (long) k * 16 * 4L;
                MemorySegment.copy(src, srcOff, dst, dstOff, (long) tailCols * 4L);
                dst.asSlice(dstOff + (long) tailCols * 4L, (long)(16 - tailCols) * 4L).fill((byte) 0);
            }
        }
    }

    static void packA_panel_Aarch_Float(MemorySegment src, MemorySegment dst, int rowStart, int mc, int colStart, int kc, int m) {
        int fullPanels = mc / 8; int tailRows = mc % 8;
        for (int panel = 0; panel < fullPanels; panel++) {
            long dstBase = (long) panel * 8 * kc * 4L;
            long r0 = (long)(rowStart + panel * 8 + 0) * m + colStart;
            long r1 = (long)(rowStart + panel * 8 + 1) * m + colStart;
            long r2 = (long)(rowStart + panel * 8 + 2) * m + colStart;
            long r3 = (long)(rowStart + panel * 8 + 3) * m + colStart;
            long r4 = (long)(rowStart + panel * 8 + 4) * m + colStart;
            long r5 = (long)(rowStart + panel * 8 + 5) * m + colStart;
            long r6 = (long)(rowStart + panel * 8 + 6) * m + colStart;
            long r7 = (long)(rowStart + panel * 8 + 7) * m + colStart;

            int k = 0;
            for (; k <= kc - 4; k += 4) {
                long dOff = dstBase + (long) k * 8 * 4L;
                dst.set(ValueLayout.JAVA_FLOAT, dOff, src.getAtIndex(ValueLayout.JAVA_FLOAT, r0 + k));
                dst.set(ValueLayout.JAVA_FLOAT, dOff + 4L, src.getAtIndex(ValueLayout.JAVA_FLOAT, r1 + k));
                dst.set(ValueLayout.JAVA_FLOAT, dOff + 2 * 4L, src.getAtIndex(ValueLayout.JAVA_FLOAT, r2 + k));
                dst.set(ValueLayout.JAVA_FLOAT, dOff + 3 * 4L, src.getAtIndex(ValueLayout.JAVA_FLOAT, r3 + k));
                dst.set(ValueLayout.JAVA_FLOAT, dOff + 4 * 4L, src.getAtIndex(ValueLayout.JAVA_FLOAT, r4 + k));
                dst.set(ValueLayout.JAVA_FLOAT, dOff + 5 * 4L, src.getAtIndex(ValueLayout.JAVA_FLOAT, r5 + k));
                dst.set(ValueLayout.JAVA_FLOAT, dOff + 6 * 4L, src.getAtIndex(ValueLayout.JAVA_FLOAT, r6 + k));
                dst.set(ValueLayout.JAVA_FLOAT, dOff + 7 * 4L, src.getAtIndex(ValueLayout.JAVA_FLOAT, r7 + k));

                long dOff1 = dOff + 8 * 4L;
                dst.set(ValueLayout.JAVA_FLOAT, dOff1, src.getAtIndex(ValueLayout.JAVA_FLOAT, r0 + k + 1));
                dst.set(ValueLayout.JAVA_FLOAT, dOff1 + 4L, src.getAtIndex(ValueLayout.JAVA_FLOAT, r1 + k + 1));
                dst.set(ValueLayout.JAVA_FLOAT, dOff1 + 2 * 4L, src.getAtIndex(ValueLayout.JAVA_FLOAT, r2 + k + 1));
                dst.set(ValueLayout.JAVA_FLOAT, dOff1 + 3 * 4L, src.getAtIndex(ValueLayout.JAVA_FLOAT, r3 + k + 1));
                dst.set(ValueLayout.JAVA_FLOAT, dOff1 + 4 * 4L, src.getAtIndex(ValueLayout.JAVA_FLOAT, r4 + k + 1));
                dst.set(ValueLayout.JAVA_FLOAT, dOff1 + 5 * 4L, src.getAtIndex(ValueLayout.JAVA_FLOAT, r5 + k + 1));
                dst.set(ValueLayout.JAVA_FLOAT, dOff1 + 6 * 4L, src.getAtIndex(ValueLayout.JAVA_FLOAT, r6 + k + 1));
                dst.set(ValueLayout.JAVA_FLOAT, dOff1 + 7 * 4L, src.getAtIndex(ValueLayout.JAVA_FLOAT, r7 + k + 1));

                long dOff2 = dOff1 + 8 * 4L;
                dst.set(ValueLayout.JAVA_FLOAT, dOff2, src.getAtIndex(ValueLayout.JAVA_FLOAT, r0 + k + 2));
                dst.set(ValueLayout.JAVA_FLOAT, dOff2 + 4L, src.getAtIndex(ValueLayout.JAVA_FLOAT, r1 + k + 2));
                dst.set(ValueLayout.JAVA_FLOAT, dOff2 + 2 * 4L, src.getAtIndex(ValueLayout.JAVA_FLOAT, r2 + k + 2));
                dst.set(ValueLayout.JAVA_FLOAT, dOff2 + 3 * 4L, src.getAtIndex(ValueLayout.JAVA_FLOAT, r3 + k + 2));
                dst.set(ValueLayout.JAVA_FLOAT, dOff2 + 4 * 4L, src.getAtIndex(ValueLayout.JAVA_FLOAT, r4 + k + 2));
                dst.set(ValueLayout.JAVA_FLOAT, dOff2 + 5 * 4L, src.getAtIndex(ValueLayout.JAVA_FLOAT, r5 + k + 2));
                dst.set(ValueLayout.JAVA_FLOAT, dOff2 + 6 * 4L, src.getAtIndex(ValueLayout.JAVA_FLOAT, r6 + k + 2));
                dst.set(ValueLayout.JAVA_FLOAT, dOff2 + 7 * 4L, src.getAtIndex(ValueLayout.JAVA_FLOAT, r7 + k + 2));

                long dOff3 = dOff2 + 8 * 4L;
                dst.set(ValueLayout.JAVA_FLOAT, dOff3, src.getAtIndex(ValueLayout.JAVA_FLOAT, r0 + k + 3));
                dst.set(ValueLayout.JAVA_FLOAT, dOff3 + 4L, src.getAtIndex(ValueLayout.JAVA_FLOAT, r1 + k + 3));
                dst.set(ValueLayout.JAVA_FLOAT, dOff3 + 2 * 4L, src.getAtIndex(ValueLayout.JAVA_FLOAT, r2 + k + 3));
                dst.set(ValueLayout.JAVA_FLOAT, dOff3 + 3 * 4L, src.getAtIndex(ValueLayout.JAVA_FLOAT, r3 + k + 3));
                dst.set(ValueLayout.JAVA_FLOAT, dOff3 + 4 * 4L, src.getAtIndex(ValueLayout.JAVA_FLOAT, r4 + k + 3));
                dst.set(ValueLayout.JAVA_FLOAT, dOff3 + 5 * 4L, src.getAtIndex(ValueLayout.JAVA_FLOAT, r5 + k + 3));
                dst.set(ValueLayout.JAVA_FLOAT, dOff3 + 6 * 4L, src.getAtIndex(ValueLayout.JAVA_FLOAT, r6 + k + 3));
                dst.set(ValueLayout.JAVA_FLOAT, dOff3 + 7 * 4L, src.getAtIndex(ValueLayout.JAVA_FLOAT, r7 + k + 3));
            }
            for (; k < kc; k++) {
                long dOff = dstBase + (long) k * 8 * 4L;
                dst.set(ValueLayout.JAVA_FLOAT, dOff, src.getAtIndex(ValueLayout.JAVA_FLOAT, r0 + k));
                dst.set(ValueLayout.JAVA_FLOAT, dOff + 4L, src.getAtIndex(ValueLayout.JAVA_FLOAT, r1 + k));
                dst.set(ValueLayout.JAVA_FLOAT, dOff + 2 * 4L, src.getAtIndex(ValueLayout.JAVA_FLOAT, r2 + k));
                dst.set(ValueLayout.JAVA_FLOAT, dOff + 3 * 4L, src.getAtIndex(ValueLayout.JAVA_FLOAT, r3 + k));
                dst.set(ValueLayout.JAVA_FLOAT, dOff + 4 * 4L, src.getAtIndex(ValueLayout.JAVA_FLOAT, r4 + k));
                dst.set(ValueLayout.JAVA_FLOAT, dOff + 5 * 4L, src.getAtIndex(ValueLayout.JAVA_FLOAT, r5 + k));
                dst.set(ValueLayout.JAVA_FLOAT, dOff + 6 * 4L, src.getAtIndex(ValueLayout.JAVA_FLOAT, r6 + k));
                dst.set(ValueLayout.JAVA_FLOAT, dOff + 7 * 4L, src.getAtIndex(ValueLayout.JAVA_FLOAT, r7 + k));
            }
        }
        if (tailRows > 0) {
            long dstBase = (long) fullPanels * 8 * kc * 4L;
            for (int r = 0; r < 8; r++) {
                if (r < tailRows) {
                    long srcRow = (long)(rowStart + fullPanels * 8 + r) * m + colStart;
                    for (int k = 0; k < kc; k++) {
                        dst.set(ValueLayout.JAVA_FLOAT, dstBase + (long) k * 8 * 4L + (long) r * 4L, src.getAtIndex(ValueLayout.JAVA_FLOAT, srcRow + k));
                    }
                } else {
                    for (int k = 0; k < kc; k++) {
                        dst.set(ValueLayout.JAVA_FLOAT, dstBase + (long) k * 8 * 4L + (long) r * 4L, 0f);
                    }
                }
            }
        }
    }

    static void packB_panel_Aarch_Float(MemorySegment src, MemorySegment dst, int rowStart, int colStart, int kc, int nc, int p) {
        int fullPanels = nc / 12; int tailCols = nc % 12;
        for (int panel = 0; panel < fullPanels; panel++) {
            long dstBase = (long) panel * 16 * kc * 4L;
            for (int k = 0; k < kc; k++) {
                long srcOff = ((long)(rowStart + k) * p + colStart + (long) panel * 12) * 4L;
                long dstOff = dstBase + (long) k * 16 * 4L;
                MemorySegment.copy(src, srcOff, dst, dstOff, 12L * 4L);
                dst.asSlice(dstOff + 12L * 4L, 4L * 4L).fill((byte) 0);
            }
        }
        if (tailCols > 0) {
            long dstBase = (long) fullPanels * 16 * kc * 4L;
            for (int k = 0; k < kc; k++) {
                long srcOff = ((long)(rowStart + k) * p + colStart + (long) fullPanels * 12) * 4L;
                long dstOff = dstBase + (long) k * 16 * 4L;
                MemorySegment.copy(src, srcOff, dst, dstOff, (long) tailCols * 4L);
                dst.asSlice(dstOff + (long) tailCols * 4L, (long)(16 - tailCols) * 4L).fill((byte) 0);
            }
        }
    }
    static final class GEBPTask_Arm_Float extends RecursiveAction {
        final MemorySegment A, pB, C; int n, m, p_cols, rowStart, rowEnd, pc, kc, jc, nc;
        GEBPTask_Arm_Float(MemorySegment A, MemorySegment pB, MemorySegment C, int n, int m, int p_cols, int rowStart, int rowEnd, int pc, int kc, int jc, int nc) {
            this.A = A; this.pB = pB; this.C = C; this.n = n; this.m = m; this.p_cols = p_cols;
            this.rowStart = rowStart; this.rowEnd = rowEnd; this.pc = pc; this.kc = kc; this.jc = jc; this.nc = nc;
        }
        @Override
        protected void compute() {
            int mc = rowEnd - rowStart;
            if (mc <= MC) {
                MemorySegment pA = tlPackedA_Arm_Float.get();
                packA_panel_Arm_Float(A, pA, rowStart, mc, pc, kc, m);
                gebpMacroKernel_Arm_Float(pA, pB, C, rowStart, mc, jc, nc, kc, p_cols);
            } else {
                int mid = rowStart + (mc / 2) - ((mc / 2) % 6);
                if (mid == rowStart) mid += 6;
                invokeAll(new GEBPTask_Arm_Float(A, pB, C, n, m, p_cols, rowStart, mid, pc, kc, jc, nc), new GEBPTask_Arm_Float(A, pB, C, n, m, p_cols, mid, rowEnd, pc, kc, jc, nc));
            }
        }
    }

    static final class GEBPTask_Aarch_Float extends RecursiveAction {
        final MemorySegment A, pB, C; int n, m, p_cols, rowStart, rowEnd, pc, kc, jc, nc;
        GEBPTask_Aarch_Float(MemorySegment A, MemorySegment pB, MemorySegment C, int n, int m, int p_cols, int rowStart, int rowEnd, int pc, int kc, int jc, int nc) {
            this.A = A; this.pB = pB; this.C = C; this.n = n; this.m = m; this.p_cols = p_cols;
            this.rowStart = rowStart; this.rowEnd = rowEnd; this.pc = pc; this.kc = kc; this.jc = jc; this.nc = nc;
        }
        @Override
        protected void compute() {
            int mc = rowEnd - rowStart;
            if (mc <= MC) {
                MemorySegment pA = tlPackedA_Aarch_Float.get();
                packA_panel_Aarch_Float(A, pA, rowStart, mc, pc, kc, m);
                gebpMacroKernel_Aarch_Float(pA, pB, C, rowStart, mc, jc, nc, kc, p_cols);
            } else {
                int mid = rowStart + (mc / 2) - ((mc / 2) % 8);
                if (mid == rowStart) mid += 8;
                invokeAll(new GEBPTask_Aarch_Float(A, pB, C, n, m, p_cols, rowStart, mid, pc, kc, jc, nc), new GEBPTask_Aarch_Float(A, pB, C, n, m, p_cols, mid, rowEnd, pc, kc, jc, nc));
            }
        }
    }

    private static void gebpMacroKernel_Arm_Float(MemorySegment pA, MemorySegment pB, MemorySegment C, int rowStart, int mc, int jc, int nc, int kc, int p) {
        int nrPanels = (nc + 15) / 16; int fullIPanels = mc / 6;
        for (int jp = 0; jp < nrPanels; jp++) {
            int jr = jp * 16; int actualNR = Math.min(16, nc - jr);
            long bBase = (long) jp * 16 * kc * 4L;
            if (actualNR == 16) {
                for (int ip = 0; ip < fullIPanels; ip++) {
                    microKernel6x16_Float(pA, (long) ip * 6 * kc * 4L, pB, bBase, C, rowStart + ip * 6, jc + jr, kc, p);
                }
                // Issue 3 Resolved: Capture tail rows for full NR bounds
                int tailIRows = mc % 6;
                if (tailIRows > 0) {
                    microKernelScalar_Float(pA, (long) fullIPanels * 6 * kc * 4L, 0, pB, bBase, C, rowStart + fullIPanels * 6, jc + jr, kc, p, tailIRows, 16, 6, 16);
                }
            } else {
                microKernelScalar_Float(pA, (long) fullIPanels * 6 * kc * 4L, 0, pB, bBase, C, rowStart, jc + jr, kc, p, mc, actualNR, 6, 16);
            }
        }
    }

    private static void gebpMacroKernel_Aarch_Float(MemorySegment pA, MemorySegment pB, MemorySegment C, int rowStart, int mc, int jc, int nc, int kc, int p) {
        int nrPanels = (nc + 11) / 12; int fullIPanels = mc / 8;
        for (int jp = 0; jp < nrPanels; jp++) {
            int jr = jp * 12; int actualNR = Math.min(12, nc - jr);
            long bBase = (long) jp * 16 * kc * 4L;
            if (actualNR == 12) {
                for (int ip = 0; ip < fullIPanels; ip++) {
                    microKernel8x12_Float(pA, (long) ip * 8 * kc * 4L, pB, bBase, C, rowStart + ip * 8, jc + jr, kc, p);
                }
                // Issue 3 Resolved: Capture tail rows for full NR bounds
                int tailIRows = mc % 8;
                if (tailIRows > 0) {
                    microKernelScalar_Float(pA, (long) fullIPanels * 8 * kc * 4L, 0, pB, bBase, C, rowStart + fullIPanels * 8, jc + jr, kc, p, tailIRows, 12, 8, 16);
                }
            } else {
                microKernelScalar_Float(pA, (long) fullIPanels * 8 * kc * 4L, 0, pB, bBase, C, rowStart, jc + jr, kc, p, mc, actualNR, 8, 16);
            }
        }
    }

    private static void microKernelScalar_Float(MemorySegment pA, long aBase, int rOff, MemorySegment pB, long bBase, MemorySegment C, int ci, int cj, int kc, int N, int mr, int nr, int MR_dim, int NR_dim) {
        for (int k = 0; k < kc; k++) {
            long aOff = aBase + (long) k * MR_dim * 4L + (long) rOff * 4L;
            long bOff = bBase + (long) k * NR_dim * 4L;
            for (int r = 0; r < mr; r++) {
                float aVal = pA.get(ValueLayout.JAVA_FLOAT, aOff + (long) r * 4L);
                for (int c = 0; c < nr; c++) {
                    long cIdx = (long)(ci + r) * N + cj + c;
                    C.setAtIndex(ValueLayout.JAVA_FLOAT, cIdx, (float)(C.getAtIndex(ValueLayout.JAVA_FLOAT, cIdx) + (aVal * pB.get(ValueLayout.JAVA_FLOAT, bOff + (long) c * 4L))));
                }
            }
        }
    }

    private static void microKernel6x16_Float(MemorySegment pA, long aBase, MemorySegment pB, long bBase, MemorySegment C, int ci, int cj, int kc, int N) {
        var c00 = FloatVector.zero(SPECIES);
        var c01 = FloatVector.zero(SPECIES);
        var c10 = FloatVector.zero(SPECIES);
        var c11 = FloatVector.zero(SPECIES);
        var c20 = FloatVector.zero(SPECIES);
        var c21 = FloatVector.zero(SPECIES);
        var c30 = FloatVector.zero(SPECIES);
        var c31 = FloatVector.zero(SPECIES);
        var c40 = FloatVector.zero(SPECIES);
        var c41 = FloatVector.zero(SPECIES);
        var c50 = FloatVector.zero(SPECIES);
        var c51 = FloatVector.zero(SPECIES);
        for (int k = 0; k < kc; k++) {
            long aOff = aBase + (long) k * 6 * 4L;
            long bOff = bBase + (long) k * 16 * 4L;
            var b0 = FloatVector.fromMemorySegment(SPECIES, pB, bOff, NATIVE);
            var b1 = FloatVector.fromMemorySegment(SPECIES, pB, bOff + 8 * 4L, NATIVE);
            var a0 = FloatVector.broadcast(SPECIES, pA.get(ValueLayout.JAVA_FLOAT, aOff + 0 * 4L));
            c00 = a0.fma(b0, c00);
            c01 = a0.fma(b1, c01);
            var a1 = FloatVector.broadcast(SPECIES, pA.get(ValueLayout.JAVA_FLOAT, aOff + 1 * 4L));
            c10 = a1.fma(b0, c10);
            c11 = a1.fma(b1, c11);
            var a2 = FloatVector.broadcast(SPECIES, pA.get(ValueLayout.JAVA_FLOAT, aOff + 2 * 4L));
            c20 = a2.fma(b0, c20);
            c21 = a2.fma(b1, c21);
            var a3 = FloatVector.broadcast(SPECIES, pA.get(ValueLayout.JAVA_FLOAT, aOff + 3 * 4L));
            c30 = a3.fma(b0, c30);
            c31 = a3.fma(b1, c31);
            var a4 = FloatVector.broadcast(SPECIES, pA.get(ValueLayout.JAVA_FLOAT, aOff + 4 * 4L));
            c40 = a4.fma(b0, c40);
            c41 = a4.fma(b1, c41);
            var a5 = FloatVector.broadcast(SPECIES, pA.get(ValueLayout.JAVA_FLOAT, aOff + 5 * 4L));
            c50 = a5.fma(b0, c50);
            c51 = a5.fma(b1, c51);
        }
        long row0 = ((long)(ci + 0) * N + cj) * 4L;
        FloatVector.fromMemorySegment(SPECIES, C, row0, NATIVE).add(c00).intoMemorySegment(C, row0, NATIVE);
        FloatVector.fromMemorySegment(SPECIES, C, row0 + 8 * 4L, NATIVE).add(c01).intoMemorySegment(C, row0 + 8 * 4L, NATIVE);
        long row1 = ((long)(ci + 1) * N + cj) * 4L;
        FloatVector.fromMemorySegment(SPECIES, C, row1, NATIVE).add(c10).intoMemorySegment(C, row1, NATIVE);
        FloatVector.fromMemorySegment(SPECIES, C, row1 + 8 * 4L, NATIVE).add(c11).intoMemorySegment(C, row1 + 8 * 4L, NATIVE);
        long row2 = ((long)(ci + 2) * N + cj) * 4L;
        FloatVector.fromMemorySegment(SPECIES, C, row2, NATIVE).add(c20).intoMemorySegment(C, row2, NATIVE);
        FloatVector.fromMemorySegment(SPECIES, C, row2 + 8 * 4L, NATIVE).add(c21).intoMemorySegment(C, row2 + 8 * 4L, NATIVE);
        long row3 = ((long)(ci + 3) * N + cj) * 4L;
        FloatVector.fromMemorySegment(SPECIES, C, row3, NATIVE).add(c30).intoMemorySegment(C, row3, NATIVE);
        FloatVector.fromMemorySegment(SPECIES, C, row3 + 8 * 4L, NATIVE).add(c31).intoMemorySegment(C, row3 + 8 * 4L, NATIVE);
        long row4 = ((long)(ci + 4) * N + cj) * 4L;
        FloatVector.fromMemorySegment(SPECIES, C, row4, NATIVE).add(c40).intoMemorySegment(C, row4, NATIVE);
        FloatVector.fromMemorySegment(SPECIES, C, row4 + 8 * 4L, NATIVE).add(c41).intoMemorySegment(C, row4 + 8 * 4L, NATIVE);
        long row5 = ((long)(ci + 5) * N + cj) * 4L;
        FloatVector.fromMemorySegment(SPECIES, C, row5, NATIVE).add(c50).intoMemorySegment(C, row5, NATIVE);
        FloatVector.fromMemorySegment(SPECIES, C, row5 + 8 * 4L, NATIVE).add(c51).intoMemorySegment(C, row5 + 8 * 4L, NATIVE);
    }
    private static void microKernel8x12_Float(MemorySegment pA, long aBase, MemorySegment pB, long bBase, MemorySegment C, int ci, int cj, int kc, int N) {
        var c00 = FloatVector.zero(SPECIES);
        var c01 = FloatVector.zero(SPECIES);
        var c10 = FloatVector.zero(SPECIES);
        var c11 = FloatVector.zero(SPECIES);
        var c20 = FloatVector.zero(SPECIES);
        var c21 = FloatVector.zero(SPECIES);
        var c30 = FloatVector.zero(SPECIES);
        var c31 = FloatVector.zero(SPECIES);
        var c40 = FloatVector.zero(SPECIES);
        var c41 = FloatVector.zero(SPECIES);
        var c50 = FloatVector.zero(SPECIES);
        var c51 = FloatVector.zero(SPECIES);
        var c60 = FloatVector.zero(SPECIES);
        var c61 = FloatVector.zero(SPECIES);
        var c70 = FloatVector.zero(SPECIES);
        var c71 = FloatVector.zero(SPECIES);
        for (int k = 0; k < kc; k++) {
            long aOff = aBase + (long) k * 8 * 4L;
            long bOff = bBase + (long) k * 16 * 4L;
            var b0 = FloatVector.fromMemorySegment(SPECIES, pB, bOff, NATIVE);
            var b1 = FloatVector.fromMemorySegment(SPECIES, pB, bOff + 8 * 4L, NATIVE);
            var a0 = FloatVector.broadcast(SPECIES, pA.get(ValueLayout.JAVA_FLOAT, aOff + 0 * 4L));
            c00 = a0.fma(b0, c00);
            c01 = a0.fma(b1, c01);
            var a1 = FloatVector.broadcast(SPECIES, pA.get(ValueLayout.JAVA_FLOAT, aOff + 1 * 4L));
            c10 = a1.fma(b0, c10);
            c11 = a1.fma(b1, c11);
            var a2 = FloatVector.broadcast(SPECIES, pA.get(ValueLayout.JAVA_FLOAT, aOff + 2 * 4L));
            c20 = a2.fma(b0, c20);
            c21 = a2.fma(b1, c21);
            var a3 = FloatVector.broadcast(SPECIES, pA.get(ValueLayout.JAVA_FLOAT, aOff + 3 * 4L));
            c30 = a3.fma(b0, c30);
            c31 = a3.fma(b1, c31);
            var a4 = FloatVector.broadcast(SPECIES, pA.get(ValueLayout.JAVA_FLOAT, aOff + 4 * 4L));
            c40 = a4.fma(b0, c40);
            c41 = a4.fma(b1, c41);
            var a5 = FloatVector.broadcast(SPECIES, pA.get(ValueLayout.JAVA_FLOAT, aOff + 5 * 4L));
            c50 = a5.fma(b0, c50);
            c51 = a5.fma(b1, c51);
            var a6 = FloatVector.broadcast(SPECIES, pA.get(ValueLayout.JAVA_FLOAT, aOff + 6 * 4L));
            c60 = a6.fma(b0, c60);
            c61 = a6.fma(b1, c61);
            var a7 = FloatVector.broadcast(SPECIES, pA.get(ValueLayout.JAVA_FLOAT, aOff + 7 * 4L));
            c70 = a7.fma(b0, c70);
            c71 = a7.fma(b1, c71);
        }
        long row0 = ((long)(ci + 0) * N + cj) * 4L;
        FloatVector.fromMemorySegment(SPECIES, C, row0, NATIVE).add(c00).intoMemorySegment(C, row0, NATIVE);
        C.setAtIndex(ValueLayout.JAVA_FLOAT, (long)(ci + 0) * N + cj + 8 + 0, (float)(C.getAtIndex(ValueLayout.JAVA_FLOAT, (long)(ci + 0) * N + cj + 8 + 0) + c01.lane(0)));
        C.setAtIndex(ValueLayout.JAVA_FLOAT, (long)(ci + 0) * N + cj + 8 + 1, (float)(C.getAtIndex(ValueLayout.JAVA_FLOAT, (long)(ci + 0) * N + cj + 8 + 1) + c01.lane(1)));
        C.setAtIndex(ValueLayout.JAVA_FLOAT, (long)(ci + 0) * N + cj + 8 + 2, (float)(C.getAtIndex(ValueLayout.JAVA_FLOAT, (long)(ci + 0) * N + cj + 8 + 2) + c01.lane(2)));
        C.setAtIndex(ValueLayout.JAVA_FLOAT, (long)(ci + 0) * N + cj + 8 + 3, (float)(C.getAtIndex(ValueLayout.JAVA_FLOAT, (long)(ci + 0) * N + cj + 8 + 3) + c01.lane(3)));
        long row1 = ((long)(ci + 1) * N + cj) * 4L;
        FloatVector.fromMemorySegment(SPECIES, C, row1, NATIVE).add(c10).intoMemorySegment(C, row1, NATIVE);
        C.setAtIndex(ValueLayout.JAVA_FLOAT, (long)(ci + 1) * N + cj + 8 + 0, (float)(C.getAtIndex(ValueLayout.JAVA_FLOAT, (long)(ci + 1) * N + cj + 8 + 0) + c11.lane(0)));
        C.setAtIndex(ValueLayout.JAVA_FLOAT, (long)(ci + 1) * N + cj + 8 + 1, (float)(C.getAtIndex(ValueLayout.JAVA_FLOAT, (long)(ci + 1) * N + cj + 8 + 1) + c11.lane(1)));
        C.setAtIndex(ValueLayout.JAVA_FLOAT, (long)(ci + 1) * N + cj + 8 + 2, (float)(C.getAtIndex(ValueLayout.JAVA_FLOAT, (long)(ci + 1) * N + cj + 8 + 2) + c11.lane(2)));
        C.setAtIndex(ValueLayout.JAVA_FLOAT, (long)(ci + 1) * N + cj + 8 + 3, (float)(C.getAtIndex(ValueLayout.JAVA_FLOAT, (long)(ci + 1) * N + cj + 8 + 3) + c11.lane(3)));
        long row2 = ((long)(ci + 2) * N + cj) * 4L;
        FloatVector.fromMemorySegment(SPECIES, C, row2, NATIVE).add(c20).intoMemorySegment(C, row2, NATIVE);
        C.setAtIndex(ValueLayout.JAVA_FLOAT, (long)(ci + 2) * N + cj + 8 + 0, (float)(C.getAtIndex(ValueLayout.JAVA_FLOAT, (long)(ci + 2) * N + cj + 8 + 0) + c21.lane(0)));
        C.setAtIndex(ValueLayout.JAVA_FLOAT, (long)(ci + 2) * N + cj + 8 + 1, (float)(C.getAtIndex(ValueLayout.JAVA_FLOAT, (long)(ci + 2) * N + cj + 8 + 1) + c21.lane(1)));
        C.setAtIndex(ValueLayout.JAVA_FLOAT, (long)(ci + 2) * N + cj + 8 + 2, (float)(C.getAtIndex(ValueLayout.JAVA_FLOAT, (long)(ci + 2) * N + cj + 8 + 2) + c21.lane(2)));
        C.setAtIndex(ValueLayout.JAVA_FLOAT, (long)(ci + 2) * N + cj + 8 + 3, (float)(C.getAtIndex(ValueLayout.JAVA_FLOAT, (long)(ci + 2) * N + cj + 8 + 3) + c21.lane(3)));
        long row3 = ((long)(ci + 3) * N + cj) * 4L;
        FloatVector.fromMemorySegment(SPECIES, C, row3, NATIVE).add(c30).intoMemorySegment(C, row3, NATIVE);
        C.setAtIndex(ValueLayout.JAVA_FLOAT, (long)(ci + 3) * N + cj + 8 + 0, (float)(C.getAtIndex(ValueLayout.JAVA_FLOAT, (long)(ci + 3) * N + cj + 8 + 0) + c31.lane(0)));
        C.setAtIndex(ValueLayout.JAVA_FLOAT, (long)(ci + 3) * N + cj + 8 + 1, (float)(C.getAtIndex(ValueLayout.JAVA_FLOAT, (long)(ci + 3) * N + cj + 8 + 1) + c31.lane(1)));
        C.setAtIndex(ValueLayout.JAVA_FLOAT, (long)(ci + 3) * N + cj + 8 + 2, (float)(C.getAtIndex(ValueLayout.JAVA_FLOAT, (long)(ci + 3) * N + cj + 8 + 2) + c31.lane(2)));
        C.setAtIndex(ValueLayout.JAVA_FLOAT, (long)(ci + 3) * N + cj + 8 + 3, (float)(C.getAtIndex(ValueLayout.JAVA_FLOAT, (long)(ci + 3) * N + cj + 8 + 3) + c31.lane(3)));
        long row4 = ((long)(ci + 4) * N + cj) * 4L;
        FloatVector.fromMemorySegment(SPECIES, C, row4, NATIVE).add(c40).intoMemorySegment(C, row4, NATIVE);
        C.setAtIndex(ValueLayout.JAVA_FLOAT, (long)(ci + 4) * N + cj + 8 + 0, (float)(C.getAtIndex(ValueLayout.JAVA_FLOAT, (long)(ci + 4) * N + cj + 8 + 0) + c41.lane(0)));
        C.setAtIndex(ValueLayout.JAVA_FLOAT, (long)(ci + 4) * N + cj + 8 + 1, (float)(C.getAtIndex(ValueLayout.JAVA_FLOAT, (long)(ci + 4) * N + cj + 8 + 1) + c41.lane(1)));
        C.setAtIndex(ValueLayout.JAVA_FLOAT, (long)(ci + 4) * N + cj + 8 + 2, (float)(C.getAtIndex(ValueLayout.JAVA_FLOAT, (long)(ci + 4) * N + cj + 8 + 2) + c41.lane(2)));
        C.setAtIndex(ValueLayout.JAVA_FLOAT, (long)(ci + 4) * N + cj + 8 + 3, (float)(C.getAtIndex(ValueLayout.JAVA_FLOAT, (long)(ci + 4) * N + cj + 8 + 3) + c41.lane(3)));
        long row5 = ((long)(ci + 5) * N + cj) * 4L;
        FloatVector.fromMemorySegment(SPECIES, C, row5, NATIVE).add(c50).intoMemorySegment(C, row5, NATIVE);
        C.setAtIndex(ValueLayout.JAVA_FLOAT, (long)(ci + 5) * N + cj + 8 + 0, (float)(C.getAtIndex(ValueLayout.JAVA_FLOAT, (long)(ci + 5) * N + cj + 8 + 0) + c51.lane(0)));
        C.setAtIndex(ValueLayout.JAVA_FLOAT, (long)(ci + 5) * N + cj + 8 + 1, (float)(C.getAtIndex(ValueLayout.JAVA_FLOAT, (long)(ci + 5) * N + cj + 8 + 1) + c51.lane(1)));
        C.setAtIndex(ValueLayout.JAVA_FLOAT, (long)(ci + 5) * N + cj + 8 + 2, (float)(C.getAtIndex(ValueLayout.JAVA_FLOAT, (long)(ci + 5) * N + cj + 8 + 2) + c51.lane(2)));
        C.setAtIndex(ValueLayout.JAVA_FLOAT, (long)(ci + 5) * N + cj + 8 + 3, (float)(C.getAtIndex(ValueLayout.JAVA_FLOAT, (long)(ci + 5) * N + cj + 8 + 3) + c51.lane(3)));
        long row6 = ((long)(ci + 6) * N + cj) * 4L;
        FloatVector.fromMemorySegment(SPECIES, C, row6, NATIVE).add(c60).intoMemorySegment(C, row6, NATIVE);
        C.setAtIndex(ValueLayout.JAVA_FLOAT, (long)(ci + 6) * N + cj + 8 + 0, (float)(C.getAtIndex(ValueLayout.JAVA_FLOAT, (long)(ci + 6) * N + cj + 8 + 0) + c61.lane(0)));
        C.setAtIndex(ValueLayout.JAVA_FLOAT, (long)(ci + 6) * N + cj + 8 + 1, (float)(C.getAtIndex(ValueLayout.JAVA_FLOAT, (long)(ci + 6) * N + cj + 8 + 1) + c61.lane(1)));
        C.setAtIndex(ValueLayout.JAVA_FLOAT, (long)(ci + 6) * N + cj + 8 + 2, (float)(C.getAtIndex(ValueLayout.JAVA_FLOAT, (long)(ci + 6) * N + cj + 8 + 2) + c61.lane(2)));
        C.setAtIndex(ValueLayout.JAVA_FLOAT, (long)(ci + 6) * N + cj + 8 + 3, (float)(C.getAtIndex(ValueLayout.JAVA_FLOAT, (long)(ci + 6) * N + cj + 8 + 3) + c61.lane(3)));
        long row7 = ((long)(ci + 7) * N + cj) * 4L;
        FloatVector.fromMemorySegment(SPECIES, C, row7, NATIVE).add(c70).intoMemorySegment(C, row7, NATIVE);
        C.setAtIndex(ValueLayout.JAVA_FLOAT, (long)(ci + 7) * N + cj + 8 + 0, (float)(C.getAtIndex(ValueLayout.JAVA_FLOAT, (long)(ci + 7) * N + cj + 8 + 0) + c71.lane(0)));
        C.setAtIndex(ValueLayout.JAVA_FLOAT, (long)(ci + 7) * N + cj + 8 + 1, (float)(C.getAtIndex(ValueLayout.JAVA_FLOAT, (long)(ci + 7) * N + cj + 8 + 1) + c71.lane(1)));
        C.setAtIndex(ValueLayout.JAVA_FLOAT, (long)(ci + 7) * N + cj + 8 + 2, (float)(C.getAtIndex(ValueLayout.JAVA_FLOAT, (long)(ci + 7) * N + cj + 8 + 2) + c71.lane(2)));
        C.setAtIndex(ValueLayout.JAVA_FLOAT, (long)(ci + 7) * N + cj + 8 + 3, (float)(C.getAtIndex(ValueLayout.JAVA_FLOAT, (long)(ci + 7) * N + cj + 8 + 3) + c71.lane(3)));
    }

    static class AVX2_Float extends RecursiveAction {
        MemorySegment A, B_T, C; int n, m, p, startRow, endRow;
        AVX2_Float(MemorySegment A, MemorySegment B_T, MemorySegment C, int n, int m, int p, int startRow, int endRow) {
            this.A = A; this.B_T = B_T; this.C = C; this.n = n; this.m = m; this.p = p; this.startRow = startRow; this.endRow = endRow;
        }
        @Override
        protected void compute() {
            if (endRow - startRow <= THRESHOLD) {
                int safeRowEnd = endRow - ((endRow - startRow) % 2); int safeColEnd = p - (p % 2);
                for (int i = startRow; i < safeRowEnd; i += 2) {
                    for (int j = 0; j < safeColEnd; j += 2) {
                        hybridKernel2x2_Float(A, B_T, C, m, p, i, j);
                    }
                }
                if (safeRowEnd < endRow) {
                    for (int j = 0; j < safeColEnd; j++) scalarDotProduct_Float(A, B_T, C, m, p, safeRowEnd, j);
                }
                if (safeColEnd < p) {
                    for (int i = startRow; i < safeRowEnd; i++) scalarDotProduct_Float(A, B_T, C, m, p, i, safeColEnd);
                }
                if (safeRowEnd < endRow && safeColEnd < p) {
                    scalarDotProduct_Float(A, B_T, C, m, p, safeRowEnd, safeColEnd);
                }
            } else {
                int mid = startRow + (endRow - startRow) / 2;
                invokeAll(new AVX2_Float(A, B_T, C, n, m, p, startRow, mid), new AVX2_Float(A, B_T, C, n, m, p, mid, endRow));
            }
        }
    }

    private static void hybridKernel2x2_Float(MemorySegment A, MemorySegment B_T, MemorySegment C, int m, int p, int i, int j) {
        var vSum00 = FloatVector.zero(SPECIES); var vSum01 = FloatVector.zero(SPECIES);
        var vSum10 = FloatVector.zero(SPECIES); var vSum11 = FloatVector.zero(SPECIES);
        long k = 0; long loopBound = SPECIES.loopBound(m);
        for (; k < loopBound; k += SPECIES.length()) {
            var vA0 = FloatVector.fromMemorySegment(SPECIES, A, ((long) i * m + k) * 4L, ByteOrder.nativeOrder());
            var vA1 = FloatVector.fromMemorySegment(SPECIES, A, ((long) (i + 1) * m + k) * 4L, ByteOrder.nativeOrder());
            var vB0 = FloatVector.fromMemorySegment(SPECIES, B_T, ((long) j * m + k) * 4L, ByteOrder.nativeOrder());
            var vB1 = FloatVector.fromMemorySegment(SPECIES, B_T, ((long) (j + 1) * m + k) * 4L, ByteOrder.nativeOrder());
            vSum00 = vA0.fma(vB0, vSum00); vSum01 = vA0.fma(vB1, vSum01);
            vSum10 = vA1.fma(vB0, vSum10); vSum11 = vA1.fma(vB1, vSum11);
        }
        float sum00 = vSum00.reduceLanes(VectorOperators.ADD); float sum01 = vSum01.reduceLanes(VectorOperators.ADD);
        float sum10 = vSum10.reduceLanes(VectorOperators.ADD); float sum11 = vSum11.reduceLanes(VectorOperators.ADD);
        for (; k < m; k++) {
            float a0 = A.getAtIndex(ValueLayout.JAVA_FLOAT, ((long) i * m + k)); float a1 = A.getAtIndex(ValueLayout.JAVA_FLOAT, ((long) (i + 1) * m + k));
            float b0 = B_T.getAtIndex(ValueLayout.JAVA_FLOAT, ((long) j * m + k)); float b1 = B_T.getAtIndex(ValueLayout.JAVA_FLOAT, ((long) (j + 1) * m + k));
            sum00 += a0 * b0; sum01 += a0 * b1; sum10 += a1 * b0; sum11 += a1 * b1;
        }
        C.setAtIndex(ValueLayout.JAVA_FLOAT, ((long) i * p + j), sum00); C.setAtIndex(ValueLayout.JAVA_FLOAT, ((long) i * p + j + 1), sum01);
        C.setAtIndex(ValueLayout.JAVA_FLOAT, ((long) (i + 1) * p + j), sum10); C.setAtIndex(ValueLayout.JAVA_FLOAT, ((long) (i + 1) * p + j + 1), sum11);
    }

    private static void scalarDotProduct_Float(MemorySegment A, MemorySegment B_T, MemorySegment C, int m, int p, int i, int j) {
        float sum = 0f;
        for (int k = 0; k < m; k++) {
            sum += A.getAtIndex(ValueLayout.JAVA_FLOAT, ((long) i * m + k)) * B_T.getAtIndex(ValueLayout.JAVA_FLOAT, ((long) j * m + k));
        }
        C.setAtIndex(ValueLayout.JAVA_FLOAT, ((long) i * p + j), sum);
    }

    // High performance sequential tiled transpose loop to prevent ForkJoin pool thread contention
    private static MemorySegment fastTranspose2D_Float(MemorySegment src, Arena arena, int rows, int cols) {
        MemorySegment dst = arena.allocate((long) rows * cols * 4L);
        int TILE = 64;
        for (int rB = 0; rB < rows; rB += TILE) {
            int rMax = Math.min(rB + TILE, rows);
            for (int cB = 0; cB < cols; cB += TILE) {
                int cMax = Math.min(cB + TILE, cols);
                for (int i = rB; i < rMax; i++) {
                    long iStride = (long) i * cols;
                    for (int j = cB; j < cMax; j++) {
                        dst.setAtIndex(ValueLayout.JAVA_FLOAT, (long) j * rows + i, src.getAtIndex(ValueLayout.JAVA_FLOAT, iStride + j));
                    }
                }
            }
        }
        return dst;
    }


    public static NDArray matmulDouble(NDArray a, NDArray b, NDArray resArray) {
        int n = a.internalShapeUnsafe()[0]; 
        int m = a.internalShapeUnsafe()[1]; 
        int p = b.internalShapeUnsafe()[1];
        
        try (Arena arena = Arena.ofShared()) {
            MemorySegment memA = a.isContiguous() ? a.getData() : a.contiguous(arena).getData();
            MemorySegment memB = b.isContiguous() ? b.getData() : b.contiguous(arena).getData();
            MemorySegment memC = resArray.getData();

            if ((long) n * m * p >= 2_000_000_000L) {
                String arch = System.getProperty("os.arch").toLowerCase();
                if (arch.contains("aarch")) {
                    blisAarchMacro_Double(memA, memB, memC, n, m, p, arena);
                } else {
                    blisArmMacro_Double(memA, memB, memC, n, m, p, arena);
                }
            } else {
                MemorySegment memB_T = fastTranspose2D_Double(memB, arena, m, p);
                POOL.invoke(new AVX2_Double(memA, memB_T, memC, n, m, p, 0, n));
            }
        }
        return resArray;
    }

    private static final ThreadLocal<MemorySegment> tlPackedA_Arm_Double =
        ThreadLocal.withInitial(() -> Arena.ofAuto().allocate((long) MC * KC * 8L, 64));
    private static final ThreadLocal<MemorySegment> tlPackedA_Aarch_Double =
        ThreadLocal.withInitial(() -> Arena.ofAuto().allocate((long) MC * KC * 8L, 64));

    private static void blisArmMacro_Double(MemorySegment A, MemorySegment B, MemorySegment C, int n, int m, int p, Arena arena) {
        C.fill((byte) 0);
        MemorySegment pB = arena.allocate((long) KC * NC_ARM * 8L, 64);
        for (int jc = 0; jc < p; jc += NC_ARM) {
            int nc = Math.min(NC_ARM, p - jc);
            for (int pc = 0; pc < m; pc += KC) {
                int kc = Math.min(KC, m - pc);
                packB_panel_Arm_Double(B, pB, pc, jc, kc, nc, p);
                POOL.invoke(new GEBPTask_Arm_Double(A, pB, C, n, m, p, 0, n, pc, kc, jc, nc));
            }
        }
    }

    private static void blisAarchMacro_Double(MemorySegment A, MemorySegment B, MemorySegment C, int n, int m, int p, Arena arena) {
        C.fill((byte) 0);
        MemorySegment pB = arena.allocate((long) KC * NC_AARCH * 8L, 64);
        for (int jc = 0; jc < p; jc += NC_AARCH) {
            int nc = Math.min(NC_AARCH, p - jc);
            for (int pc = 0; pc < m; pc += KC) {
                int kc = Math.min(KC, m - pc);
                packB_panel_Aarch_Double(B, pB, pc, jc, kc, nc, p);
                POOL.invoke(new GEBPTask_Aarch_Double(A, pB, C, n, m, p, 0, n, pc, kc, jc, nc));
            }
        }
    }

    static void packA_panel_Arm_Double(MemorySegment src, MemorySegment dst, int rowStart, int mc, int colStart, int kc, int m) {
        int fullPanels = mc / 6; int tailRows = mc % 6;
        for (int panel = 0; panel < fullPanels; panel++) {
            long dstBase = (long) panel * 6 * kc * 8L;
            long r0 = (long)(rowStart + panel * 6 + 0) * m + colStart;
            long r1 = (long)(rowStart + panel * 6 + 1) * m + colStart;
            long r2 = (long)(rowStart + panel * 6 + 2) * m + colStart;
            long r3 = (long)(rowStart + panel * 6 + 3) * m + colStart;
            long r4 = (long)(rowStart + panel * 6 + 4) * m + colStart;
            long r5 = (long)(rowStart + panel * 6 + 5) * m + colStart;
            
            int k = 0;
            for (; k <= kc - 4; k += 4) {
                long dOff = dstBase + (long) k * 6 * 8L;
                dst.set(ValueLayout.JAVA_DOUBLE, dOff, src.getAtIndex(ValueLayout.JAVA_DOUBLE, r0 + k));
                dst.set(ValueLayout.JAVA_DOUBLE, dOff + 8L, src.getAtIndex(ValueLayout.JAVA_DOUBLE, r1 + k));
                dst.set(ValueLayout.JAVA_DOUBLE, dOff + 2 * 8L, src.getAtIndex(ValueLayout.JAVA_DOUBLE, r2 + k));
                dst.set(ValueLayout.JAVA_DOUBLE, dOff + 3 * 8L, src.getAtIndex(ValueLayout.JAVA_DOUBLE, r3 + k));
                dst.set(ValueLayout.JAVA_DOUBLE, dOff + 4 * 8L, src.getAtIndex(ValueLayout.JAVA_DOUBLE, r4 + k));
                dst.set(ValueLayout.JAVA_DOUBLE, dOff + 5 * 8L, src.getAtIndex(ValueLayout.JAVA_DOUBLE, r5 + k));
                
                long dOff1 = dOff + 6 * 8L;
                dst.set(ValueLayout.JAVA_DOUBLE, dOff1, src.getAtIndex(ValueLayout.JAVA_DOUBLE, r0 + k + 1));
                dst.set(ValueLayout.JAVA_DOUBLE, dOff1 + 8L, src.getAtIndex(ValueLayout.JAVA_DOUBLE, r1 + k + 1));
                dst.set(ValueLayout.JAVA_DOUBLE, dOff1 + 2 * 8L, src.getAtIndex(ValueLayout.JAVA_DOUBLE, r2 + k + 1));
                dst.set(ValueLayout.JAVA_DOUBLE, dOff1 + 3 * 8L, src.getAtIndex(ValueLayout.JAVA_DOUBLE, r3 + k + 1));
                dst.set(ValueLayout.JAVA_DOUBLE, dOff1 + 4 * 8L, src.getAtIndex(ValueLayout.JAVA_DOUBLE, r4 + k + 1));
                dst.set(ValueLayout.JAVA_DOUBLE, dOff1 + 5 * 8L, src.getAtIndex(ValueLayout.JAVA_DOUBLE, r5 + k + 1));

                long dOff2 = dOff1 + 6 * 8L;
                dst.set(ValueLayout.JAVA_DOUBLE, dOff2, src.getAtIndex(ValueLayout.JAVA_DOUBLE, r0 + k + 2));
                dst.set(ValueLayout.JAVA_DOUBLE, dOff2 + 8L, src.getAtIndex(ValueLayout.JAVA_DOUBLE, r1 + k + 2));
                dst.set(ValueLayout.JAVA_DOUBLE, dOff2 + 2 * 8L, src.getAtIndex(ValueLayout.JAVA_DOUBLE, r2 + k + 2));
                dst.set(ValueLayout.JAVA_DOUBLE, dOff2 + 3 * 8L, src.getAtIndex(ValueLayout.JAVA_DOUBLE, r3 + k + 2));
                dst.set(ValueLayout.JAVA_DOUBLE, dOff2 + 4 * 8L, src.getAtIndex(ValueLayout.JAVA_DOUBLE, r4 + k + 2));
                dst.set(ValueLayout.JAVA_DOUBLE, dOff2 + 5 * 8L, src.getAtIndex(ValueLayout.JAVA_DOUBLE, r5 + k + 2));

                long dOff3 = dOff2 + 6 * 8L;
                dst.set(ValueLayout.JAVA_DOUBLE, dOff3, src.getAtIndex(ValueLayout.JAVA_DOUBLE, r0 + k + 3));
                dst.set(ValueLayout.JAVA_DOUBLE, dOff3 + 8L, src.getAtIndex(ValueLayout.JAVA_DOUBLE, r1 + k + 3));
                dst.set(ValueLayout.JAVA_DOUBLE, dOff3 + 2 * 8L, src.getAtIndex(ValueLayout.JAVA_DOUBLE, r2 + k + 3));
                dst.set(ValueLayout.JAVA_DOUBLE, dOff3 + 3 * 8L, src.getAtIndex(ValueLayout.JAVA_DOUBLE, r3 + k + 3));
                dst.set(ValueLayout.JAVA_DOUBLE, dOff3 + 4 * 8L, src.getAtIndex(ValueLayout.JAVA_DOUBLE, r4 + k + 3));
                dst.set(ValueLayout.JAVA_DOUBLE, dOff3 + 5 * 8L, src.getAtIndex(ValueLayout.JAVA_DOUBLE, r5 + k + 3));
            }
            for (; k < kc; k++) {
                long dOff = dstBase + (long) k * 6 * 8L;
                dst.set(ValueLayout.JAVA_DOUBLE, dOff, src.getAtIndex(ValueLayout.JAVA_DOUBLE, r0 + k));
                dst.set(ValueLayout.JAVA_DOUBLE, dOff + 8L, src.getAtIndex(ValueLayout.JAVA_DOUBLE, r1 + k));
                dst.set(ValueLayout.JAVA_DOUBLE, dOff + 2 * 8L, src.getAtIndex(ValueLayout.JAVA_DOUBLE, r2 + k));
                dst.set(ValueLayout.JAVA_DOUBLE, dOff + 3 * 8L, src.getAtIndex(ValueLayout.JAVA_DOUBLE, r3 + k));
                dst.set(ValueLayout.JAVA_DOUBLE, dOff + 4 * 8L, src.getAtIndex(ValueLayout.JAVA_DOUBLE, r4 + k));
                dst.set(ValueLayout.JAVA_DOUBLE, dOff + 5 * 8L, src.getAtIndex(ValueLayout.JAVA_DOUBLE, r5 + k));
            }
        }
        if (tailRows > 0) {
            long dstBase = (long) fullPanels * 6 * kc * 8L;
            for (int r = 0; r < 6; r++) {
                if (r < tailRows) {
                    long srcRow = (long)(rowStart + fullPanels * 6 + r) * m + colStart;
                    for (int k = 0; k < kc; k++) {
                        dst.set(ValueLayout.JAVA_DOUBLE, dstBase + (long) k * 6 * 8L + (long) r * 8L, src.getAtIndex(ValueLayout.JAVA_DOUBLE, srcRow + k));
                    }
                } else {
                    for (int k = 0; k < kc; k++) {
                        dst.set(ValueLayout.JAVA_DOUBLE, dstBase + (long) k * 6 * 8L + (long) r * 8L, 0.0);
                    }
                }
            }
        }
    }

    static void packB_panel_Arm_Double(MemorySegment src, MemorySegment dst, int rowStart, int colStart, int kc, int nc, int p) {
        int fullPanels = nc / 16; int tailCols = nc % 16;
        for (int panel = 0; panel < fullPanels; panel++) {
            long dstBase = (long) panel * 16 * kc * 8L;
            for (int k = 0; k < kc; k++) {
                long srcOff = ((long)(rowStart + k) * p + colStart + (long) panel * 16) * 8L;
                MemorySegment.copy(src, srcOff, dst, dstBase + (long) k * 16 * 8L, 16L * 8L);
            }
        }
        if (tailCols > 0) {
            long dstBase = (long) fullPanels * 16 * kc * 8L;
            for (int k = 0; k < kc; k++) {
                long srcOff = ((long)(rowStart + k) * p + colStart + (long) fullPanels * 16) * 8L;
                long dstOff = dstBase + (long) k * 16 * 8L;
                MemorySegment.copy(src, srcOff, dst, dstOff, (long) tailCols * 8L);
                dst.asSlice(dstOff + (long) tailCols * 8L, (long)(16 - tailCols) * 8L).fill((byte) 0);
            }
        }
    }

    static void packA_panel_Aarch_Double(MemorySegment src, MemorySegment dst, int rowStart, int mc, int colStart, int kc, int m) {
        int fullPanels = mc / 8; int tailRows = mc % 8;
        for (int panel = 0; panel < fullPanels; panel++) {
            long dstBase = (long) panel * 8 * kc * 8L;
            long r0 = (long)(rowStart + panel * 8 + 0) * m + colStart;
            long r1 = (long)(rowStart + panel * 8 + 1) * m + colStart;
            long r2 = (long)(rowStart + panel * 8 + 2) * m + colStart;
            long r3 = (long)(rowStart + panel * 8 + 3) * m + colStart;
            long r4 = (long)(rowStart + panel * 8 + 4) * m + colStart;
            long r5 = (long)(rowStart + panel * 8 + 5) * m + colStart;
            long r6 = (long)(rowStart + panel * 8 + 6) * m + colStart;
            long r7 = (long)(rowStart + panel * 8 + 7) * m + colStart;

            int k = 0;
            for (; k <= kc - 4; k += 4) {
                long dOff = dstBase + (long) k * 8 * 8L;
                dst.set(ValueLayout.JAVA_DOUBLE, dOff, src.getAtIndex(ValueLayout.JAVA_DOUBLE, r0 + k));
                dst.set(ValueLayout.JAVA_DOUBLE, dOff + 8L, src.getAtIndex(ValueLayout.JAVA_DOUBLE, r1 + k));
                dst.set(ValueLayout.JAVA_DOUBLE, dOff + 2 * 8L, src.getAtIndex(ValueLayout.JAVA_DOUBLE, r2 + k));
                dst.set(ValueLayout.JAVA_DOUBLE, dOff + 3 * 8L, src.getAtIndex(ValueLayout.JAVA_DOUBLE, r3 + k));
                dst.set(ValueLayout.JAVA_DOUBLE, dOff + 4 * 8L, src.getAtIndex(ValueLayout.JAVA_DOUBLE, r4 + k));
                dst.set(ValueLayout.JAVA_DOUBLE, dOff + 5 * 8L, src.getAtIndex(ValueLayout.JAVA_DOUBLE, r5 + k));
                dst.set(ValueLayout.JAVA_DOUBLE, dOff + 6 * 8L, src.getAtIndex(ValueLayout.JAVA_DOUBLE, r6 + k));
                dst.set(ValueLayout.JAVA_DOUBLE, dOff + 7 * 8L, src.getAtIndex(ValueLayout.JAVA_DOUBLE, r7 + k));

                long dOff1 = dOff + 8 * 8L;
                dst.set(ValueLayout.JAVA_DOUBLE, dOff1, src.getAtIndex(ValueLayout.JAVA_DOUBLE, r0 + k + 1));
                dst.set(ValueLayout.JAVA_DOUBLE, dOff1 + 8L, src.getAtIndex(ValueLayout.JAVA_DOUBLE, r1 + k + 1));
                dst.set(ValueLayout.JAVA_DOUBLE, dOff1 + 2 * 8L, src.getAtIndex(ValueLayout.JAVA_DOUBLE, r2 + k + 1));
                dst.set(ValueLayout.JAVA_DOUBLE, dOff1 + 3 * 8L, src.getAtIndex(ValueLayout.JAVA_DOUBLE, r3 + k + 1));
                dst.set(ValueLayout.JAVA_DOUBLE, dOff1 + 4 * 8L, src.getAtIndex(ValueLayout.JAVA_DOUBLE, r4 + k + 1));
                dst.set(ValueLayout.JAVA_DOUBLE, dOff1 + 5 * 8L, src.getAtIndex(ValueLayout.JAVA_DOUBLE, r5 + k + 1));
                dst.set(ValueLayout.JAVA_DOUBLE, dOff1 + 6 * 8L, src.getAtIndex(ValueLayout.JAVA_DOUBLE, r6 + k + 1));
                dst.set(ValueLayout.JAVA_DOUBLE, dOff1 + 7 * 8L, src.getAtIndex(ValueLayout.JAVA_DOUBLE, r7 + k + 1));

                long dOff2 = dOff1 + 8 * 8L;
                dst.set(ValueLayout.JAVA_DOUBLE, dOff2, src.getAtIndex(ValueLayout.JAVA_DOUBLE, r0 + k + 2));
                dst.set(ValueLayout.JAVA_DOUBLE, dOff2 + 8L, src.getAtIndex(ValueLayout.JAVA_DOUBLE, r1 + k + 2));
                dst.set(ValueLayout.JAVA_DOUBLE, dOff2 + 2 * 8L, src.getAtIndex(ValueLayout.JAVA_DOUBLE, r2 + k + 2));
                dst.set(ValueLayout.JAVA_DOUBLE, dOff2 + 3 * 8L, src.getAtIndex(ValueLayout.JAVA_DOUBLE, r3 + k + 2));
                dst.set(ValueLayout.JAVA_DOUBLE, dOff2 + 4 * 8L, src.getAtIndex(ValueLayout.JAVA_DOUBLE, r4 + k + 2));
                dst.set(ValueLayout.JAVA_DOUBLE, dOff2 + 5 * 8L, src.getAtIndex(ValueLayout.JAVA_DOUBLE, r5 + k + 2));
                dst.set(ValueLayout.JAVA_DOUBLE, dOff2 + 6 * 8L, src.getAtIndex(ValueLayout.JAVA_DOUBLE, r6 + k + 2));
                dst.set(ValueLayout.JAVA_DOUBLE, dOff2 + 7 * 8L, src.getAtIndex(ValueLayout.JAVA_DOUBLE, r7 + k + 2));

                long dOff3 = dOff2 + 8 * 8L;
                dst.set(ValueLayout.JAVA_DOUBLE, dOff3, src.getAtIndex(ValueLayout.JAVA_DOUBLE, r0 + k + 3));
                dst.set(ValueLayout.JAVA_DOUBLE, dOff3 + 8L, src.getAtIndex(ValueLayout.JAVA_DOUBLE, r1 + k + 3));
                dst.set(ValueLayout.JAVA_DOUBLE, dOff3 + 2 * 8L, src.getAtIndex(ValueLayout.JAVA_DOUBLE, r2 + k + 3));
                dst.set(ValueLayout.JAVA_DOUBLE, dOff3 + 3 * 8L, src.getAtIndex(ValueLayout.JAVA_DOUBLE, r3 + k + 3));
                dst.set(ValueLayout.JAVA_DOUBLE, dOff3 + 4 * 8L, src.getAtIndex(ValueLayout.JAVA_DOUBLE, r4 + k + 3));
                dst.set(ValueLayout.JAVA_DOUBLE, dOff3 + 5 * 8L, src.getAtIndex(ValueLayout.JAVA_DOUBLE, r5 + k + 3));
                dst.set(ValueLayout.JAVA_DOUBLE, dOff3 + 6 * 8L, src.getAtIndex(ValueLayout.JAVA_DOUBLE, r6 + k + 3));
                dst.set(ValueLayout.JAVA_DOUBLE, dOff3 + 7 * 8L, src.getAtIndex(ValueLayout.JAVA_DOUBLE, r7 + k + 3));
            }
            for (; k < kc; k++) {
                long dOff = dstBase + (long) k * 8 * 8L;
                dst.set(ValueLayout.JAVA_DOUBLE, dOff, src.getAtIndex(ValueLayout.JAVA_DOUBLE, r0 + k));
                dst.set(ValueLayout.JAVA_DOUBLE, dOff + 8L, src.getAtIndex(ValueLayout.JAVA_DOUBLE, r1 + k));
                dst.set(ValueLayout.JAVA_DOUBLE, dOff + 2 * 8L, src.getAtIndex(ValueLayout.JAVA_DOUBLE, r2 + k));
                dst.set(ValueLayout.JAVA_DOUBLE, dOff + 3 * 8L, src.getAtIndex(ValueLayout.JAVA_DOUBLE, r3 + k));
                dst.set(ValueLayout.JAVA_DOUBLE, dOff + 4 * 8L, src.getAtIndex(ValueLayout.JAVA_DOUBLE, r4 + k));
                dst.set(ValueLayout.JAVA_DOUBLE, dOff + 5 * 8L, src.getAtIndex(ValueLayout.JAVA_DOUBLE, r5 + k));
                dst.set(ValueLayout.JAVA_DOUBLE, dOff + 6 * 8L, src.getAtIndex(ValueLayout.JAVA_DOUBLE, r6 + k));
                dst.set(ValueLayout.JAVA_DOUBLE, dOff + 7 * 8L, src.getAtIndex(ValueLayout.JAVA_DOUBLE, r7 + k));
            }
        }
        if (tailRows > 0) {
            long dstBase = (long) fullPanels * 8 * kc * 8L;
            for (int r = 0; r < 8; r++) {
                if (r < tailRows) {
                    long srcRow = (long)(rowStart + fullPanels * 8 + r) * m + colStart;
                    for (int k = 0; k < kc; k++) {
                        dst.set(ValueLayout.JAVA_DOUBLE, dstBase + (long) k * 8 * 8L + (long) r * 8L, src.getAtIndex(ValueLayout.JAVA_DOUBLE, srcRow + k));
                    }
                } else {
                    for (int k = 0; k < kc; k++) {
                        dst.set(ValueLayout.JAVA_DOUBLE, dstBase + (long) k * 8 * 8L + (long) r * 8L, 0.0);
                    }
                }
            }
        }
    }

    static void packB_panel_Aarch_Double(MemorySegment src, MemorySegment dst, int rowStart, int colStart, int kc, int nc, int p) {
        int fullPanels = nc / 12; int tailCols = nc % 12;
        for (int panel = 0; panel < fullPanels; panel++) {
            long dstBase = (long) panel * 16 * kc * 8L;
            for (int k = 0; k < kc; k++) {
                long srcOff = ((long)(rowStart + k) * p + colStart + (long) panel * 12) * 8L;
                long dstOff = dstBase + (long) k * 16 * 8L;
                MemorySegment.copy(src, srcOff, dst, dstOff, 12L * 8L);
                dst.asSlice(dstOff + 12L * 8L, 4L * 8L).fill((byte) 0);
            }
        }
        if (tailCols > 0) {
            long dstBase = (long) fullPanels * 16 * kc * 8L;
            for (int k = 0; k < kc; k++) {
                long srcOff = ((long)(rowStart + k) * p + colStart + (long) fullPanels * 12) * 8L;
                long dstOff = dstBase + (long) k * 16 * 8L;
                MemorySegment.copy(src, srcOff, dst, dstOff, (long) tailCols * 8L);
                dst.asSlice(dstOff + (long) tailCols * 8L, (long)(16 - tailCols) * 8L).fill((byte) 0);
            }
        }
    }
    static final class GEBPTask_Arm_Double extends RecursiveAction {
        final MemorySegment A, pB, C; int n, m, p_cols, rowStart, rowEnd, pc, kc, jc, nc;
        GEBPTask_Arm_Double(MemorySegment A, MemorySegment pB, MemorySegment C, int n, int m, int p_cols, int rowStart, int rowEnd, int pc, int kc, int jc, int nc) {
            this.A = A; this.pB = pB; this.C = C; this.n = n; this.m = m; this.p_cols = p_cols;
            this.rowStart = rowStart; this.rowEnd = rowEnd; this.pc = pc; this.kc = kc; this.jc = jc; this.nc = nc;
        }
        @Override
        protected void compute() {
            int mc = rowEnd - rowStart;
            if (mc <= MC) {
                MemorySegment pA = tlPackedA_Arm_Double.get();
                packA_panel_Arm_Double(A, pA, rowStart, mc, pc, kc, m);
                gebpMacroKernel_Arm_Double(pA, pB, C, rowStart, mc, jc, nc, kc, p_cols);
            } else {
                int mid = rowStart + (mc / 2) - ((mc / 2) % 6);
                if (mid == rowStart) mid += 6;
                invokeAll(new GEBPTask_Arm_Double(A, pB, C, n, m, p_cols, rowStart, mid, pc, kc, jc, nc), new GEBPTask_Arm_Double(A, pB, C, n, m, p_cols, mid, rowEnd, pc, kc, jc, nc));
            }
        }
    }

    static final class GEBPTask_Aarch_Double extends RecursiveAction {
        final MemorySegment A, pB, C; int n, m, p_cols, rowStart, rowEnd, pc, kc, jc, nc;
        GEBPTask_Aarch_Double(MemorySegment A, MemorySegment pB, MemorySegment C, int n, int m, int p_cols, int rowStart, int rowEnd, int pc, int kc, int jc, int nc) {
            this.A = A; this.pB = pB; this.C = C; this.n = n; this.m = m; this.p_cols = p_cols;
            this.rowStart = rowStart; this.rowEnd = rowEnd; this.pc = pc; this.kc = kc; this.jc = jc; this.nc = nc;
        }
        @Override
        protected void compute() {
            int mc = rowEnd - rowStart;
            if (mc <= MC) {
                MemorySegment pA = tlPackedA_Aarch_Double.get();
                packA_panel_Aarch_Double(A, pA, rowStart, mc, pc, kc, m);
                gebpMacroKernel_Aarch_Double(pA, pB, C, rowStart, mc, jc, nc, kc, p_cols);
            } else {
                int mid = rowStart + (mc / 2) - ((mc / 2) % 8);
                if (mid == rowStart) mid += 8;
                invokeAll(new GEBPTask_Aarch_Double(A, pB, C, n, m, p_cols, rowStart, mid, pc, kc, jc, nc), new GEBPTask_Aarch_Double(A, pB, C, n, m, p_cols, mid, rowEnd, pc, kc, jc, nc));
            }
        }
    }

    private static void gebpMacroKernel_Arm_Double(MemorySegment pA, MemorySegment pB, MemorySegment C, int rowStart, int mc, int jc, int nc, int kc, int p) {
        int nrPanels = (nc + 15) / 16; int fullIPanels = mc / 6;
        for (int jp = 0; jp < nrPanels; jp++) {
            int jr = jp * 16; int actualNR = Math.min(16, nc - jr);
            long bBase = (long) jp * 16 * kc * 8L;
            if (actualNR == 16) {
                for (int ip = 0; ip < fullIPanels; ip++) {
                    microKernel6x16_Double(pA, (long) ip * 6 * kc * 8L, pB, bBase, C, rowStart + ip * 6, jc + jr, kc, p);
                }
                // Issue 3 Resolved: Capture tail rows for full NR bounds
                int tailIRows = mc % 6;
                if (tailIRows > 0) {
                    microKernelScalar_Double(pA, (long) fullIPanels * 6 * kc * 8L, 0, pB, bBase, C, rowStart + fullIPanels * 6, jc + jr, kc, p, tailIRows, 16, 6, 16);
                }
            } else {
                microKernelScalar_Double(pA, (long) fullIPanels * 6 * kc * 8L, 0, pB, bBase, C, rowStart, jc + jr, kc, p, mc, actualNR, 6, 16);
            }
        }
    }

    private static void gebpMacroKernel_Aarch_Double(MemorySegment pA, MemorySegment pB, MemorySegment C, int rowStart, int mc, int jc, int nc, int kc, int p) {
        int nrPanels = (nc + 11) / 12; int fullIPanels = mc / 8;
        for (int jp = 0; jp < nrPanels; jp++) {
            int jr = jp * 12; int actualNR = Math.min(12, nc - jr);
            long bBase = (long) jp * 16 * kc * 8L;
            if (actualNR == 12) {
                for (int ip = 0; ip < fullIPanels; ip++) {
                    microKernel8x12_Double(pA, (long) ip * 8 * kc * 8L, pB, bBase, C, rowStart + ip * 8, jc + jr, kc, p);
                }
                // Issue 3 Resolved: Capture tail rows for full NR bounds
                int tailIRows = mc % 8;
                if (tailIRows > 0) {
                    microKernelScalar_Double(pA, (long) fullIPanels * 8 * kc * 8L, 0, pB, bBase, C, rowStart + fullIPanels * 8, jc + jr, kc, p, tailIRows, 12, 8, 16);
                }
            } else {
                microKernelScalar_Double(pA, (long) fullIPanels * 8 * kc * 8L, 0, pB, bBase, C, rowStart, jc + jr, kc, p, mc, actualNR, 8, 16);
            }
        }
    }

    private static void microKernelScalar_Double(MemorySegment pA, long aBase, int rOff, MemorySegment pB, long bBase, MemorySegment C, int ci, int cj, int kc, int N, int mr, int nr, int MR_dim, int NR_dim) {
        for (int k = 0; k < kc; k++) {
            long aOff = aBase + (long) k * MR_dim * 8L + (long) rOff * 8L;
            long bOff = bBase + (long) k * NR_dim * 8L;
            for (int r = 0; r < mr; r++) {
                double aVal = pA.get(ValueLayout.JAVA_DOUBLE, aOff + (long) r * 8L);
                for (int c = 0; c < nr; c++) {
                    long cIdx = (long)(ci + r) * N + cj + c;
                    C.setAtIndex(ValueLayout.JAVA_DOUBLE, cIdx, (double)(C.getAtIndex(ValueLayout.JAVA_DOUBLE, cIdx) + (aVal * pB.get(ValueLayout.JAVA_DOUBLE, bOff + (long) c * 8L))));
                }
            }
        }
    }

    private static void microKernel6x16_Double(MemorySegment pA, long aBase, MemorySegment pB, long bBase, MemorySegment C, int ci, int cj, int kc, int N) {
        var c00 = DoubleVector.zero(SPECIESDB);
        var c01 = DoubleVector.zero(SPECIESDB);
        var c10 = DoubleVector.zero(SPECIESDB);
        var c11 = DoubleVector.zero(SPECIESDB);
        var c20 = DoubleVector.zero(SPECIESDB);
        var c21 = DoubleVector.zero(SPECIESDB);
        var c30 = DoubleVector.zero(SPECIESDB);
        var c31 = DoubleVector.zero(SPECIESDB);
        var c40 = DoubleVector.zero(SPECIESDB);
        var c41 = DoubleVector.zero(SPECIESDB);
        var c50 = DoubleVector.zero(SPECIESDB);
        var c51 = DoubleVector.zero(SPECIESDB);
        for (int k = 0; k < kc; k++) {
            long aOff = aBase + (long) k * 6 * 8L;
            long bOff = bBase + (long) k * 16 * 8L;
            var b0 = DoubleVector.fromMemorySegment(SPECIESDB, pB, bOff, NATIVE);
            var b1 = DoubleVector.fromMemorySegment(SPECIESDB, pB, bOff + 8 * 8L, NATIVE);
            var a0 = DoubleVector.broadcast(SPECIESDB, pA.get(ValueLayout.JAVA_DOUBLE, aOff + 0 * 8L));
            c00 = a0.fma(b0, c00);
            c01 = a0.fma(b1, c01);
            var a1 = DoubleVector.broadcast(SPECIESDB, pA.get(ValueLayout.JAVA_DOUBLE, aOff + 1 * 8L));
            c10 = a1.fma(b0, c10);
            c11 = a1.fma(b1, c11);
            var a2 = DoubleVector.broadcast(SPECIESDB, pA.get(ValueLayout.JAVA_DOUBLE, aOff + 2 * 8L));
            c20 = a2.fma(b0, c20);
            c21 = a2.fma(b1, c21);
            var a3 = DoubleVector.broadcast(SPECIESDB, pA.get(ValueLayout.JAVA_DOUBLE, aOff + 3 * 8L));
            c30 = a3.fma(b0, c30);
            c31 = a3.fma(b1, c31);
            var a4 = DoubleVector.broadcast(SPECIESDB, pA.get(ValueLayout.JAVA_DOUBLE, aOff + 4 * 8L));
            c40 = a4.fma(b0, c40);
            c41 = a4.fma(b1, c41);
            var a5 = DoubleVector.broadcast(SPECIESDB, pA.get(ValueLayout.JAVA_DOUBLE, aOff + 5 * 8L));
            c50 = a5.fma(b0, c50);
            c51 = a5.fma(b1, c51);
        }
        long row0 = ((long)(ci + 0) * N + cj) * 8L;
        DoubleVector.fromMemorySegment(SPECIESDB, C, row0, NATIVE).add(c00).intoMemorySegment(C, row0, NATIVE);
        DoubleVector.fromMemorySegment(SPECIESDB, C, row0 + 8 * 8L, NATIVE).add(c01).intoMemorySegment(C, row0 + 8 * 8L, NATIVE);
        long row1 = ((long)(ci + 1) * N + cj) * 8L;
        DoubleVector.fromMemorySegment(SPECIESDB, C, row1, NATIVE).add(c10).intoMemorySegment(C, row1, NATIVE);
        DoubleVector.fromMemorySegment(SPECIESDB, C, row1 + 8 * 8L, NATIVE).add(c11).intoMemorySegment(C, row1 + 8 * 8L, NATIVE);
        long row2 = ((long)(ci + 2) * N + cj) * 8L;
        DoubleVector.fromMemorySegment(SPECIESDB, C, row2, NATIVE).add(c20).intoMemorySegment(C, row2, NATIVE);
        DoubleVector.fromMemorySegment(SPECIESDB, C, row2 + 8 * 8L, NATIVE).add(c21).intoMemorySegment(C, row2 + 8 * 8L, NATIVE);
        long row3 = ((long)(ci + 3) * N + cj) * 8L;
        DoubleVector.fromMemorySegment(SPECIESDB, C, row3, NATIVE).add(c30).intoMemorySegment(C, row3, NATIVE);
        DoubleVector.fromMemorySegment(SPECIESDB, C, row3 + 8 * 8L, NATIVE).add(c31).intoMemorySegment(C, row3 + 8 * 8L, NATIVE);
        long row4 = ((long)(ci + 4) * N + cj) * 8L;
        DoubleVector.fromMemorySegment(SPECIESDB, C, row4, NATIVE).add(c40).intoMemorySegment(C, row4, NATIVE);
        DoubleVector.fromMemorySegment(SPECIESDB, C, row4 + 8 * 8L, NATIVE).add(c41).intoMemorySegment(C, row4 + 8 * 8L, NATIVE);
        long row5 = ((long)(ci + 5) * N + cj) * 8L;
        DoubleVector.fromMemorySegment(SPECIESDB, C, row5, NATIVE).add(c50).intoMemorySegment(C, row5, NATIVE);
        DoubleVector.fromMemorySegment(SPECIESDB, C, row5 + 8 * 8L, NATIVE).add(c51).intoMemorySegment(C, row5 + 8 * 8L, NATIVE);
    }
    private static void microKernel8x12_Double(MemorySegment pA, long aBase, MemorySegment pB, long bBase, MemorySegment C, int ci, int cj, int kc, int N) {
        var c00 = DoubleVector.zero(SPECIESDB);
        var c01 = DoubleVector.zero(SPECIESDB);
        var c10 = DoubleVector.zero(SPECIESDB);
        var c11 = DoubleVector.zero(SPECIESDB);
        var c20 = DoubleVector.zero(SPECIESDB);
        var c21 = DoubleVector.zero(SPECIESDB);
        var c30 = DoubleVector.zero(SPECIESDB);
        var c31 = DoubleVector.zero(SPECIESDB);
        var c40 = DoubleVector.zero(SPECIESDB);
        var c41 = DoubleVector.zero(SPECIESDB);
        var c50 = DoubleVector.zero(SPECIESDB);
        var c51 = DoubleVector.zero(SPECIESDB);
        var c60 = DoubleVector.zero(SPECIESDB);
        var c61 = DoubleVector.zero(SPECIESDB);
        var c70 = DoubleVector.zero(SPECIESDB);
        var c71 = DoubleVector.zero(SPECIESDB);
        for (int k = 0; k < kc; k++) {
            long aOff = aBase + (long) k * 8 * 8L;
            long bOff = bBase + (long) k * 16 * 8L;
            var b0 = DoubleVector.fromMemorySegment(SPECIESDB, pB, bOff, NATIVE);
            var b1 = DoubleVector.fromMemorySegment(SPECIESDB, pB, bOff + 8 * 8L, NATIVE);
            var a0 = DoubleVector.broadcast(SPECIESDB, pA.get(ValueLayout.JAVA_DOUBLE, aOff + 0 * 8L));
            c00 = a0.fma(b0, c00);
            c01 = a0.fma(b1, c01);
            var a1 = DoubleVector.broadcast(SPECIESDB, pA.get(ValueLayout.JAVA_DOUBLE, aOff + 1 * 8L));
            c10 = a1.fma(b0, c10);
            c11 = a1.fma(b1, c11);
            var a2 = DoubleVector.broadcast(SPECIESDB, pA.get(ValueLayout.JAVA_DOUBLE, aOff + 2 * 8L));
            c20 = a2.fma(b0, c20);
            c21 = a2.fma(b1, c21);
            var a3 = DoubleVector.broadcast(SPECIESDB, pA.get(ValueLayout.JAVA_DOUBLE, aOff + 3 * 8L));
            c30 = a3.fma(b0, c30);
            c31 = a3.fma(b1, c31);
            var a4 = DoubleVector.broadcast(SPECIESDB, pA.get(ValueLayout.JAVA_DOUBLE, aOff + 4 * 8L));
            c40 = a4.fma(b0, c40);
            c41 = a4.fma(b1, c41);
            var a5 = DoubleVector.broadcast(SPECIESDB, pA.get(ValueLayout.JAVA_DOUBLE, aOff + 5 * 8L));
            c50 = a5.fma(b0, c50);
            c51 = a5.fma(b1, c51);
            var a6 = DoubleVector.broadcast(SPECIESDB, pA.get(ValueLayout.JAVA_DOUBLE, aOff + 6 * 8L));
            c60 = a6.fma(b0, c60);
            c61 = a6.fma(b1, c61);
            var a7 = DoubleVector.broadcast(SPECIESDB, pA.get(ValueLayout.JAVA_DOUBLE, aOff + 7 * 8L));
            c70 = a7.fma(b0, c70);
            c71 = a7.fma(b1, c71);
        }
        long row0 = ((long)(ci + 0) * N + cj) * 8L;
        DoubleVector.fromMemorySegment(SPECIESDB, C, row0, NATIVE).add(c00).intoMemorySegment(C, row0, NATIVE);
        C.setAtIndex(ValueLayout.JAVA_DOUBLE, (long)(ci + 0) * N + cj + 8 + 0, (double)(C.getAtIndex(ValueLayout.JAVA_DOUBLE, (long)(ci + 0) * N + cj + 8 + 0) + c01.lane(0)));
        C.setAtIndex(ValueLayout.JAVA_DOUBLE, (long)(ci + 0) * N + cj + 8 + 1, (double)(C.getAtIndex(ValueLayout.JAVA_DOUBLE, (long)(ci + 0) * N + cj + 8 + 1) + c01.lane(1)));
        C.setAtIndex(ValueLayout.JAVA_DOUBLE, (long)(ci + 0) * N + cj + 8 + 2, (double)(C.getAtIndex(ValueLayout.JAVA_DOUBLE, (long)(ci + 0) * N + cj + 8 + 2) + c01.lane(2)));
        C.setAtIndex(ValueLayout.JAVA_DOUBLE, (long)(ci + 0) * N + cj + 8 + 3, (double)(C.getAtIndex(ValueLayout.JAVA_DOUBLE, (long)(ci + 0) * N + cj + 8 + 3) + c01.lane(3)));
        long row1 = ((long)(ci + 1) * N + cj) * 8L;
        DoubleVector.fromMemorySegment(SPECIESDB, C, row1, NATIVE).add(c10).intoMemorySegment(C, row1, NATIVE);
        C.setAtIndex(ValueLayout.JAVA_DOUBLE, (long)(ci + 1) * N + cj + 8 + 0, (double)(C.getAtIndex(ValueLayout.JAVA_DOUBLE, (long)(ci + 1) * N + cj + 8 + 0) + c11.lane(0)));
        C.setAtIndex(ValueLayout.JAVA_DOUBLE, (long)(ci + 1) * N + cj + 8 + 1, (double)(C.getAtIndex(ValueLayout.JAVA_DOUBLE, (long)(ci + 1) * N + cj + 8 + 1) + c11.lane(1)));
        C.setAtIndex(ValueLayout.JAVA_DOUBLE, (long)(ci + 1) * N + cj + 8 + 2, (double)(C.getAtIndex(ValueLayout.JAVA_DOUBLE, (long)(ci + 1) * N + cj + 8 + 2) + c11.lane(2)));
        C.setAtIndex(ValueLayout.JAVA_DOUBLE, (long)(ci + 1) * N + cj + 8 + 3, (double)(C.getAtIndex(ValueLayout.JAVA_DOUBLE, (long)(ci + 1) * N + cj + 8 + 3) + c11.lane(3)));
        long row2 = ((long)(ci + 2) * N + cj) * 8L;
        DoubleVector.fromMemorySegment(SPECIESDB, C, row2, NATIVE).add(c20).intoMemorySegment(C, row2, NATIVE);
        C.setAtIndex(ValueLayout.JAVA_DOUBLE, (long)(ci + 2) * N + cj + 8 + 0, (double)(C.getAtIndex(ValueLayout.JAVA_DOUBLE, (long)(ci + 2) * N + cj + 8 + 0) + c21.lane(0)));
        C.setAtIndex(ValueLayout.JAVA_DOUBLE, (long)(ci + 2) * N + cj + 8 + 1, (double)(C.getAtIndex(ValueLayout.JAVA_DOUBLE, (long)(ci + 2) * N + cj + 8 + 1) + c21.lane(1)));
        C.setAtIndex(ValueLayout.JAVA_DOUBLE, (long)(ci + 2) * N + cj + 8 + 2, (double)(C.getAtIndex(ValueLayout.JAVA_DOUBLE, (long)(ci + 2) * N + cj + 8 + 2) + c21.lane(2)));
        C.setAtIndex(ValueLayout.JAVA_DOUBLE, (long)(ci + 2) * N + cj + 8 + 3, (double)(C.getAtIndex(ValueLayout.JAVA_DOUBLE, (long)(ci + 2) * N + cj + 8 + 3) + c21.lane(3)));
        long row3 = ((long)(ci + 3) * N + cj) * 8L;
        DoubleVector.fromMemorySegment(SPECIESDB, C, row3, NATIVE).add(c30).intoMemorySegment(C, row3, NATIVE);
        C.setAtIndex(ValueLayout.JAVA_DOUBLE, (long)(ci + 3) * N + cj + 8 + 0, (double)(C.getAtIndex(ValueLayout.JAVA_DOUBLE, (long)(ci + 3) * N + cj + 8 + 0) + c31.lane(0)));
        C.setAtIndex(ValueLayout.JAVA_DOUBLE, (long)(ci + 3) * N + cj + 8 + 1, (double)(C.getAtIndex(ValueLayout.JAVA_DOUBLE, (long)(ci + 3) * N + cj + 8 + 1) + c31.lane(1)));
        C.setAtIndex(ValueLayout.JAVA_DOUBLE, (long)(ci + 3) * N + cj + 8 + 2, (double)(C.getAtIndex(ValueLayout.JAVA_DOUBLE, (long)(ci + 3) * N + cj + 8 + 2) + c31.lane(2)));
        C.setAtIndex(ValueLayout.JAVA_DOUBLE, (long)(ci + 3) * N + cj + 8 + 3, (double)(C.getAtIndex(ValueLayout.JAVA_DOUBLE, (long)(ci + 3) * N + cj + 8 + 3) + c31.lane(3)));
        long row4 = ((long)(ci + 4) * N + cj) * 8L;
        DoubleVector.fromMemorySegment(SPECIESDB, C, row4, NATIVE).add(c40).intoMemorySegment(C, row4, NATIVE);
        C.setAtIndex(ValueLayout.JAVA_DOUBLE, (long)(ci + 4) * N + cj + 8 + 0, (double)(C.getAtIndex(ValueLayout.JAVA_DOUBLE, (long)(ci + 4) * N + cj + 8 + 0) + c41.lane(0)));
        C.setAtIndex(ValueLayout.JAVA_DOUBLE, (long)(ci + 4) * N + cj + 8 + 1, (double)(C.getAtIndex(ValueLayout.JAVA_DOUBLE, (long)(ci + 4) * N + cj + 8 + 1) + c41.lane(1)));
        C.setAtIndex(ValueLayout.JAVA_DOUBLE, (long)(ci + 4) * N + cj + 8 + 2, (double)(C.getAtIndex(ValueLayout.JAVA_DOUBLE, (long)(ci + 4) * N + cj + 8 + 2) + c41.lane(2)));
        C.setAtIndex(ValueLayout.JAVA_DOUBLE, (long)(ci + 4) * N + cj + 8 + 3, (double)(C.getAtIndex(ValueLayout.JAVA_DOUBLE, (long)(ci + 4) * N + cj + 8 + 3) + c41.lane(3)));
        long row5 = ((long)(ci + 5) * N + cj) * 8L;
        DoubleVector.fromMemorySegment(SPECIESDB, C, row5, NATIVE).add(c50).intoMemorySegment(C, row5, NATIVE);
        C.setAtIndex(ValueLayout.JAVA_DOUBLE, (long)(ci + 5) * N + cj + 8 + 0, (double)(C.getAtIndex(ValueLayout.JAVA_DOUBLE, (long)(ci + 5) * N + cj + 8 + 0) + c51.lane(0)));
        C.setAtIndex(ValueLayout.JAVA_DOUBLE, (long)(ci + 5) * N + cj + 8 + 1, (double)(C.getAtIndex(ValueLayout.JAVA_DOUBLE, (long)(ci + 5) * N + cj + 8 + 1) + c51.lane(1)));
        C.setAtIndex(ValueLayout.JAVA_DOUBLE, (long)(ci + 5) * N + cj + 8 + 2, (double)(C.getAtIndex(ValueLayout.JAVA_DOUBLE, (long)(ci + 5) * N + cj + 8 + 2) + c51.lane(2)));
        C.setAtIndex(ValueLayout.JAVA_DOUBLE, (long)(ci + 5) * N + cj + 8 + 3, (double)(C.getAtIndex(ValueLayout.JAVA_DOUBLE, (long)(ci + 5) * N + cj + 8 + 3) + c51.lane(3)));
        long row6 = ((long)(ci + 6) * N + cj) * 8L;
        DoubleVector.fromMemorySegment(SPECIESDB, C, row6, NATIVE).add(c60).intoMemorySegment(C, row6, NATIVE);
        C.setAtIndex(ValueLayout.JAVA_DOUBLE, (long)(ci + 6) * N + cj + 8 + 0, (double)(C.getAtIndex(ValueLayout.JAVA_DOUBLE, (long)(ci + 6) * N + cj + 8 + 0) + c61.lane(0)));
        C.setAtIndex(ValueLayout.JAVA_DOUBLE, (long)(ci + 6) * N + cj + 8 + 1, (double)(C.getAtIndex(ValueLayout.JAVA_DOUBLE, (long)(ci + 6) * N + cj + 8 + 1) + c61.lane(1)));
        C.setAtIndex(ValueLayout.JAVA_DOUBLE, (long)(ci + 6) * N + cj + 8 + 2, (double)(C.getAtIndex(ValueLayout.JAVA_DOUBLE, (long)(ci + 6) * N + cj + 8 + 2) + c61.lane(2)));
        C.setAtIndex(ValueLayout.JAVA_DOUBLE, (long)(ci + 6) * N + cj + 8 + 3, (double)(C.getAtIndex(ValueLayout.JAVA_DOUBLE, (long)(ci + 6) * N + cj + 8 + 3) + c61.lane(3)));
        long row7 = ((long)(ci + 7) * N + cj) * 8L;
        DoubleVector.fromMemorySegment(SPECIESDB, C, row7, NATIVE).add(c70).intoMemorySegment(C, row7, NATIVE);
        C.setAtIndex(ValueLayout.JAVA_DOUBLE, (long)(ci + 7) * N + cj + 8 + 0, (double)(C.getAtIndex(ValueLayout.JAVA_DOUBLE, (long)(ci + 7) * N + cj + 8 + 0) + c71.lane(0)));
        C.setAtIndex(ValueLayout.JAVA_DOUBLE, (long)(ci + 7) * N + cj + 8 + 1, (double)(C.getAtIndex(ValueLayout.JAVA_DOUBLE, (long)(ci + 7) * N + cj + 8 + 1) + c71.lane(1)));
        C.setAtIndex(ValueLayout.JAVA_DOUBLE, (long)(ci + 7) * N + cj + 8 + 2, (double)(C.getAtIndex(ValueLayout.JAVA_DOUBLE, (long)(ci + 7) * N + cj + 8 + 2) + c71.lane(2)));
        C.setAtIndex(ValueLayout.JAVA_DOUBLE, (long)(ci + 7) * N + cj + 8 + 3, (double)(C.getAtIndex(ValueLayout.JAVA_DOUBLE, (long)(ci + 7) * N + cj + 8 + 3) + c71.lane(3)));
    }

    static class AVX2_Double extends RecursiveAction {
        MemorySegment A, B_T, C; int n, m, p, startRow, endRow;
        AVX2_Double(MemorySegment A, MemorySegment B_T, MemorySegment C, int n, int m, int p, int startRow, int endRow) {
            this.A = A; this.B_T = B_T; this.C = C; this.n = n; this.m = m; this.p = p; this.startRow = startRow; this.endRow = endRow;
        }
        @Override
        protected void compute() {
            if (endRow - startRow <= THRESHOLD) {
                int safeRowEnd = endRow - ((endRow - startRow) % 2); int safeColEnd = p - (p % 2);
                for (int i = startRow; i < safeRowEnd; i += 2) {
                    for (int j = 0; j < safeColEnd; j += 2) {
                        hybridKernel2x2_Double(A, B_T, C, m, p, i, j);
                    }
                }
                if (safeRowEnd < endRow) {
                    for (int j = 0; j < safeColEnd; j++) scalarDotProduct_Double(A, B_T, C, m, p, safeRowEnd, j);
                }
                if (safeColEnd < p) {
                    for (int i = startRow; i < safeRowEnd; i++) scalarDotProduct_Double(A, B_T, C, m, p, i, safeColEnd);
                }
                if (safeRowEnd < endRow && safeColEnd < p) {
                    scalarDotProduct_Double(A, B_T, C, m, p, safeRowEnd, safeColEnd);
                }
            } else {
                int mid = startRow + (endRow - startRow) / 2;
                invokeAll(new AVX2_Double(A, B_T, C, n, m, p, startRow, mid), new AVX2_Double(A, B_T, C, n, m, p, mid, endRow));
            }
        }
    }

    private static void hybridKernel2x2_Double(MemorySegment A, MemorySegment B_T, MemorySegment C, int m, int p, int i, int j) {
        var vSum00 = DoubleVector.zero(SPECIESDB); var vSum01 = DoubleVector.zero(SPECIESDB);
        var vSum10 = DoubleVector.zero(SPECIESDB); var vSum11 = DoubleVector.zero(SPECIESDB);
        long k = 0; long loopBound = SPECIESDB.loopBound(m);
        for (; k < loopBound; k += SPECIESDB.length()) {
            var vA0 = DoubleVector.fromMemorySegment(SPECIESDB, A, ((long) i * m + k) * 8L, ByteOrder.nativeOrder());
            var vA1 = DoubleVector.fromMemorySegment(SPECIESDB, A, ((long) (i + 1) * m + k) * 8L, ByteOrder.nativeOrder());
            var vB0 = DoubleVector.fromMemorySegment(SPECIESDB, B_T, ((long) j * m + k) * 8L, ByteOrder.nativeOrder());
            var vB1 = DoubleVector.fromMemorySegment(SPECIESDB, B_T, ((long) (j + 1) * m + k) * 8L, ByteOrder.nativeOrder());
            vSum00 = vA0.fma(vB0, vSum00); vSum01 = vA0.fma(vB1, vSum01);
            vSum10 = vA1.fma(vB0, vSum10); vSum11 = vA1.fma(vB1, vSum11);
        }
        double sum00 = vSum00.reduceLanes(VectorOperators.ADD); double sum01 = vSum01.reduceLanes(VectorOperators.ADD);
        double sum10 = vSum10.reduceLanes(VectorOperators.ADD); double sum11 = vSum11.reduceLanes(VectorOperators.ADD);
        for (; k < m; k++) {
            double a0 = A.getAtIndex(ValueLayout.JAVA_DOUBLE, ((long) i * m + k)); double a1 = A.getAtIndex(ValueLayout.JAVA_DOUBLE, ((long) (i + 1) * m + k));
            double b0 = B_T.getAtIndex(ValueLayout.JAVA_DOUBLE, ((long) j * m + k)); double b1 = B_T.getAtIndex(ValueLayout.JAVA_DOUBLE, ((long) (j + 1) * m + k));
            sum00 += a0 * b0; sum01 += a0 * b1; sum10 += a1 * b0; sum11 += a1 * b1;
        }
        C.setAtIndex(ValueLayout.JAVA_DOUBLE, ((long) i * p + j), sum00); C.setAtIndex(ValueLayout.JAVA_DOUBLE, ((long) i * p + j + 1), sum01);
        C.setAtIndex(ValueLayout.JAVA_DOUBLE, ((long) (i + 1) * p + j), sum10); C.setAtIndex(ValueLayout.JAVA_DOUBLE, ((long) (i + 1) * p + j + 1), sum11);
    }

    private static void scalarDotProduct_Double(MemorySegment A, MemorySegment B_T, MemorySegment C, int m, int p, int i, int j) {
        double sum = 0.0;
        for (int k = 0; k < m; k++) {
            sum += A.getAtIndex(ValueLayout.JAVA_DOUBLE, ((long) i * m + k)) * B_T.getAtIndex(ValueLayout.JAVA_DOUBLE, ((long) j * m + k));
        }
        C.setAtIndex(ValueLayout.JAVA_DOUBLE, ((long) i * p + j), sum);
    }

    // High performance sequential tiled transpose loop to prevent ForkJoin pool thread contention
    private static MemorySegment fastTranspose2D_Double(MemorySegment src, Arena arena, int rows, int cols) {
        MemorySegment dst = arena.allocate((long) rows * cols * 8L);
        int TILE = 64;
        for (int rB = 0; rB < rows; rB += TILE) {
            int rMax = Math.min(rB + TILE, rows);
            for (int cB = 0; cB < cols; cB += TILE) {
                int cMax = Math.min(cB + TILE, cols);
                for (int i = rB; i < rMax; i++) {
                    long iStride = (long) i * cols;
                    for (int j = cB; j < cMax; j++) {
                        dst.setAtIndex(ValueLayout.JAVA_DOUBLE, (long) j * rows + i, src.getAtIndex(ValueLayout.JAVA_DOUBLE, iStride + j));
                    }
                }
            }
        }
        return dst;
    }


    public static NDArray matmulInt(NDArray a, NDArray b, NDArray resArray) {
        int n = a.internalShapeUnsafe()[0]; 
        int m = a.internalShapeUnsafe()[1]; 
        int p = b.internalShapeUnsafe()[1];
        
        try (Arena arena = Arena.ofShared()) {
            MemorySegment memA = a.isContiguous() ? a.getData() : a.contiguous(arena).getData();
            MemorySegment memB = b.isContiguous() ? b.getData() : b.contiguous(arena).getData();
            MemorySegment memC = resArray.getData();

            if ((long) n * m * p >= 2_000_000_000L) {
                String arch = System.getProperty("os.arch").toLowerCase();
                if (arch.contains("aarch")) {
                    blisAarchMacro_Int(memA, memB, memC, n, m, p, arena);
                } else {
                    blisArmMacro_Int(memA, memB, memC, n, m, p, arena);
                }
            } else {
                MemorySegment memB_T = fastTranspose2D_Int(memB, arena, m, p);
                POOL.invoke(new AVX2_Int(memA, memB_T, memC, n, m, p, 0, n));
            }
        }
        return resArray;
    }

    private static final ThreadLocal<MemorySegment> tlPackedA_Arm_Int =
        ThreadLocal.withInitial(() -> Arena.ofAuto().allocate((long) MC * KC * 4L, 64));
    private static final ThreadLocal<MemorySegment> tlPackedA_Aarch_Int =
        ThreadLocal.withInitial(() -> Arena.ofAuto().allocate((long) MC * KC * 4L, 64));

    private static void blisArmMacro_Int(MemorySegment A, MemorySegment B, MemorySegment C, int n, int m, int p, Arena arena) {
        C.fill((byte) 0);
        MemorySegment pB = arena.allocate((long) KC * NC_ARM * 4L, 64);
        for (int jc = 0; jc < p; jc += NC_ARM) {
            int nc = Math.min(NC_ARM, p - jc);
            for (int pc = 0; pc < m; pc += KC) {
                int kc = Math.min(KC, m - pc);
                packB_panel_Arm_Int(B, pB, pc, jc, kc, nc, p);
                POOL.invoke(new GEBPTask_Arm_Int(A, pB, C, n, m, p, 0, n, pc, kc, jc, nc));
            }
        }
    }

    private static void blisAarchMacro_Int(MemorySegment A, MemorySegment B, MemorySegment C, int n, int m, int p, Arena arena) {
        C.fill((byte) 0);
        MemorySegment pB = arena.allocate((long) KC * NC_AARCH * 4L, 64);
        for (int jc = 0; jc < p; jc += NC_AARCH) {
            int nc = Math.min(NC_AARCH, p - jc);
            for (int pc = 0; pc < m; pc += KC) {
                int kc = Math.min(KC, m - pc);
                packB_panel_Aarch_Int(B, pB, pc, jc, kc, nc, p);
                POOL.invoke(new GEBPTask_Aarch_Int(A, pB, C, n, m, p, 0, n, pc, kc, jc, nc));
            }
        }
    }

    static void packA_panel_Arm_Int(MemorySegment src, MemorySegment dst, int rowStart, int mc, int colStart, int kc, int m) {
        int fullPanels = mc / 6; int tailRows = mc % 6;
        for (int panel = 0; panel < fullPanels; panel++) {
            long dstBase = (long) panel * 6 * kc * 4L;
            long r0 = (long)(rowStart + panel * 6 + 0) * m + colStart;
            long r1 = (long)(rowStart + panel * 6 + 1) * m + colStart;
            long r2 = (long)(rowStart + panel * 6 + 2) * m + colStart;
            long r3 = (long)(rowStart + panel * 6 + 3) * m + colStart;
            long r4 = (long)(rowStart + panel * 6 + 4) * m + colStart;
            long r5 = (long)(rowStart + panel * 6 + 5) * m + colStart;
            
            int k = 0;
            for (; k <= kc - 4; k += 4) {
                long dOff = dstBase + (long) k * 6 * 4L;
                dst.set(ValueLayout.JAVA_INT, dOff, src.getAtIndex(ValueLayout.JAVA_INT, r0 + k));
                dst.set(ValueLayout.JAVA_INT, dOff + 4L, src.getAtIndex(ValueLayout.JAVA_INT, r1 + k));
                dst.set(ValueLayout.JAVA_INT, dOff + 2 * 4L, src.getAtIndex(ValueLayout.JAVA_INT, r2 + k));
                dst.set(ValueLayout.JAVA_INT, dOff + 3 * 4L, src.getAtIndex(ValueLayout.JAVA_INT, r3 + k));
                dst.set(ValueLayout.JAVA_INT, dOff + 4 * 4L, src.getAtIndex(ValueLayout.JAVA_INT, r4 + k));
                dst.set(ValueLayout.JAVA_INT, dOff + 5 * 4L, src.getAtIndex(ValueLayout.JAVA_INT, r5 + k));
                
                long dOff1 = dOff + 6 * 4L;
                dst.set(ValueLayout.JAVA_INT, dOff1, src.getAtIndex(ValueLayout.JAVA_INT, r0 + k + 1));
                dst.set(ValueLayout.JAVA_INT, dOff1 + 4L, src.getAtIndex(ValueLayout.JAVA_INT, r1 + k + 1));
                dst.set(ValueLayout.JAVA_INT, dOff1 + 2 * 4L, src.getAtIndex(ValueLayout.JAVA_INT, r2 + k + 1));
                dst.set(ValueLayout.JAVA_INT, dOff1 + 3 * 4L, src.getAtIndex(ValueLayout.JAVA_INT, r3 + k + 1));
                dst.set(ValueLayout.JAVA_INT, dOff1 + 4 * 4L, src.getAtIndex(ValueLayout.JAVA_INT, r4 + k + 1));
                dst.set(ValueLayout.JAVA_INT, dOff1 + 5 * 4L, src.getAtIndex(ValueLayout.JAVA_INT, r5 + k + 1));

                long dOff2 = dOff1 + 6 * 4L;
                dst.set(ValueLayout.JAVA_INT, dOff2, src.getAtIndex(ValueLayout.JAVA_INT, r0 + k + 2));
                dst.set(ValueLayout.JAVA_INT, dOff2 + 4L, src.getAtIndex(ValueLayout.JAVA_INT, r1 + k + 2));
                dst.set(ValueLayout.JAVA_INT, dOff2 + 2 * 4L, src.getAtIndex(ValueLayout.JAVA_INT, r2 + k + 2));
                dst.set(ValueLayout.JAVA_INT, dOff2 + 3 * 4L, src.getAtIndex(ValueLayout.JAVA_INT, r3 + k + 2));
                dst.set(ValueLayout.JAVA_INT, dOff2 + 4 * 4L, src.getAtIndex(ValueLayout.JAVA_INT, r4 + k + 2));
                dst.set(ValueLayout.JAVA_INT, dOff2 + 5 * 4L, src.getAtIndex(ValueLayout.JAVA_INT, r5 + k + 2));

                long dOff3 = dOff2 + 6 * 4L;
                dst.set(ValueLayout.JAVA_INT, dOff3, src.getAtIndex(ValueLayout.JAVA_INT, r0 + k + 3));
                dst.set(ValueLayout.JAVA_INT, dOff3 + 4L, src.getAtIndex(ValueLayout.JAVA_INT, r1 + k + 3));
                dst.set(ValueLayout.JAVA_INT, dOff3 + 2 * 4L, src.getAtIndex(ValueLayout.JAVA_INT, r2 + k + 3));
                dst.set(ValueLayout.JAVA_INT, dOff3 + 3 * 4L, src.getAtIndex(ValueLayout.JAVA_INT, r3 + k + 3));
                dst.set(ValueLayout.JAVA_INT, dOff3 + 4 * 4L, src.getAtIndex(ValueLayout.JAVA_INT, r4 + k + 3));
                dst.set(ValueLayout.JAVA_INT, dOff3 + 5 * 4L, src.getAtIndex(ValueLayout.JAVA_INT, r5 + k + 3));
            }
            for (; k < kc; k++) {
                long dOff = dstBase + (long) k * 6 * 4L;
                dst.set(ValueLayout.JAVA_INT, dOff, src.getAtIndex(ValueLayout.JAVA_INT, r0 + k));
                dst.set(ValueLayout.JAVA_INT, dOff + 4L, src.getAtIndex(ValueLayout.JAVA_INT, r1 + k));
                dst.set(ValueLayout.JAVA_INT, dOff + 2 * 4L, src.getAtIndex(ValueLayout.JAVA_INT, r2 + k));
                dst.set(ValueLayout.JAVA_INT, dOff + 3 * 4L, src.getAtIndex(ValueLayout.JAVA_INT, r3 + k));
                dst.set(ValueLayout.JAVA_INT, dOff + 4 * 4L, src.getAtIndex(ValueLayout.JAVA_INT, r4 + k));
                dst.set(ValueLayout.JAVA_INT, dOff + 5 * 4L, src.getAtIndex(ValueLayout.JAVA_INT, r5 + k));
            }
        }
        if (tailRows > 0) {
            long dstBase = (long) fullPanels * 6 * kc * 4L;
            for (int r = 0; r < 6; r++) {
                if (r < tailRows) {
                    long srcRow = (long)(rowStart + fullPanels * 6 + r) * m + colStart;
                    for (int k = 0; k < kc; k++) {
                        dst.set(ValueLayout.JAVA_INT, dstBase + (long) k * 6 * 4L + (long) r * 4L, src.getAtIndex(ValueLayout.JAVA_INT, srcRow + k));
                    }
                } else {
                    for (int k = 0; k < kc; k++) {
                        dst.set(ValueLayout.JAVA_INT, dstBase + (long) k * 6 * 4L + (long) r * 4L, 0);
                    }
                }
            }
        }
    }

    static void packB_panel_Arm_Int(MemorySegment src, MemorySegment dst, int rowStart, int colStart, int kc, int nc, int p) {
        int fullPanels = nc / 16; int tailCols = nc % 16;
        for (int panel = 0; panel < fullPanels; panel++) {
            long dstBase = (long) panel * 16 * kc * 4L;
            for (int k = 0; k < kc; k++) {
                long srcOff = ((long)(rowStart + k) * p + colStart + (long) panel * 16) * 4L;
                MemorySegment.copy(src, srcOff, dst, dstBase + (long) k * 16 * 4L, 16L * 4L);
            }
        }
        if (tailCols > 0) {
            long dstBase = (long) fullPanels * 16 * kc * 4L;
            for (int k = 0; k < kc; k++) {
                long srcOff = ((long)(rowStart + k) * p + colStart + (long) fullPanels * 16) * 4L;
                long dstOff = dstBase + (long) k * 16 * 4L;
                MemorySegment.copy(src, srcOff, dst, dstOff, (long) tailCols * 4L);
                dst.asSlice(dstOff + (long) tailCols * 4L, (long)(16 - tailCols) * 4L).fill((byte) 0);
            }
        }
    }

    static void packA_panel_Aarch_Int(MemorySegment src, MemorySegment dst, int rowStart, int mc, int colStart, int kc, int m) {
        int fullPanels = mc / 8; int tailRows = mc % 8;
        for (int panel = 0; panel < fullPanels; panel++) {
            long dstBase = (long) panel * 8 * kc * 4L;
            long r0 = (long)(rowStart + panel * 8 + 0) * m + colStart;
            long r1 = (long)(rowStart + panel * 8 + 1) * m + colStart;
            long r2 = (long)(rowStart + panel * 8 + 2) * m + colStart;
            long r3 = (long)(rowStart + panel * 8 + 3) * m + colStart;
            long r4 = (long)(rowStart + panel * 8 + 4) * m + colStart;
            long r5 = (long)(rowStart + panel * 8 + 5) * m + colStart;
            long r6 = (long)(rowStart + panel * 8 + 6) * m + colStart;
            long r7 = (long)(rowStart + panel * 8 + 7) * m + colStart;

            int k = 0;
            for (; k <= kc - 4; k += 4) {
                long dOff = dstBase + (long) k * 8 * 4L;
                dst.set(ValueLayout.JAVA_INT, dOff, src.getAtIndex(ValueLayout.JAVA_INT, r0 + k));
                dst.set(ValueLayout.JAVA_INT, dOff + 4L, src.getAtIndex(ValueLayout.JAVA_INT, r1 + k));
                dst.set(ValueLayout.JAVA_INT, dOff + 2 * 4L, src.getAtIndex(ValueLayout.JAVA_INT, r2 + k));
                dst.set(ValueLayout.JAVA_INT, dOff + 3 * 4L, src.getAtIndex(ValueLayout.JAVA_INT, r3 + k));
                dst.set(ValueLayout.JAVA_INT, dOff + 4 * 4L, src.getAtIndex(ValueLayout.JAVA_INT, r4 + k));
                dst.set(ValueLayout.JAVA_INT, dOff + 5 * 4L, src.getAtIndex(ValueLayout.JAVA_INT, r5 + k));
                dst.set(ValueLayout.JAVA_INT, dOff + 6 * 4L, src.getAtIndex(ValueLayout.JAVA_INT, r6 + k));
                dst.set(ValueLayout.JAVA_INT, dOff + 7 * 4L, src.getAtIndex(ValueLayout.JAVA_INT, r7 + k));

                long dOff1 = dOff + 8 * 4L;
                dst.set(ValueLayout.JAVA_INT, dOff1, src.getAtIndex(ValueLayout.JAVA_INT, r0 + k + 1));
                dst.set(ValueLayout.JAVA_INT, dOff1 + 4L, src.getAtIndex(ValueLayout.JAVA_INT, r1 + k + 1));
                dst.set(ValueLayout.JAVA_INT, dOff1 + 2 * 4L, src.getAtIndex(ValueLayout.JAVA_INT, r2 + k + 1));
                dst.set(ValueLayout.JAVA_INT, dOff1 + 3 * 4L, src.getAtIndex(ValueLayout.JAVA_INT, r3 + k + 1));
                dst.set(ValueLayout.JAVA_INT, dOff1 + 4 * 4L, src.getAtIndex(ValueLayout.JAVA_INT, r4 + k + 1));
                dst.set(ValueLayout.JAVA_INT, dOff1 + 5 * 4L, src.getAtIndex(ValueLayout.JAVA_INT, r5 + k + 1));
                dst.set(ValueLayout.JAVA_INT, dOff1 + 6 * 4L, src.getAtIndex(ValueLayout.JAVA_INT, r6 + k + 1));
                dst.set(ValueLayout.JAVA_INT, dOff1 + 7 * 4L, src.getAtIndex(ValueLayout.JAVA_INT, r7 + k + 1));

                long dOff2 = dOff1 + 8 * 4L;
                dst.set(ValueLayout.JAVA_INT, dOff2, src.getAtIndex(ValueLayout.JAVA_INT, r0 + k + 2));
                dst.set(ValueLayout.JAVA_INT, dOff2 + 4L, src.getAtIndex(ValueLayout.JAVA_INT, r1 + k + 2));
                dst.set(ValueLayout.JAVA_INT, dOff2 + 2 * 4L, src.getAtIndex(ValueLayout.JAVA_INT, r2 + k + 2));
                dst.set(ValueLayout.JAVA_INT, dOff2 + 3 * 4L, src.getAtIndex(ValueLayout.JAVA_INT, r3 + k + 2));
                dst.set(ValueLayout.JAVA_INT, dOff2 + 4 * 4L, src.getAtIndex(ValueLayout.JAVA_INT, r4 + k + 2));
                dst.set(ValueLayout.JAVA_INT, dOff2 + 5 * 4L, src.getAtIndex(ValueLayout.JAVA_INT, r5 + k + 2));
                dst.set(ValueLayout.JAVA_INT, dOff2 + 6 * 4L, src.getAtIndex(ValueLayout.JAVA_INT, r6 + k + 2));
                dst.set(ValueLayout.JAVA_INT, dOff2 + 7 * 4L, src.getAtIndex(ValueLayout.JAVA_INT, r7 + k + 2));

                long dOff3 = dOff2 + 8 * 4L;
                dst.set(ValueLayout.JAVA_INT, dOff3, src.getAtIndex(ValueLayout.JAVA_INT, r0 + k + 3));
                dst.set(ValueLayout.JAVA_INT, dOff3 + 4L, src.getAtIndex(ValueLayout.JAVA_INT, r1 + k + 3));
                dst.set(ValueLayout.JAVA_INT, dOff3 + 2 * 4L, src.getAtIndex(ValueLayout.JAVA_INT, r2 + k + 3));
                dst.set(ValueLayout.JAVA_INT, dOff3 + 3 * 4L, src.getAtIndex(ValueLayout.JAVA_INT, r3 + k + 3));
                dst.set(ValueLayout.JAVA_INT, dOff3 + 4 * 4L, src.getAtIndex(ValueLayout.JAVA_INT, r4 + k + 3));
                dst.set(ValueLayout.JAVA_INT, dOff3 + 5 * 4L, src.getAtIndex(ValueLayout.JAVA_INT, r5 + k + 3));
                dst.set(ValueLayout.JAVA_INT, dOff3 + 6 * 4L, src.getAtIndex(ValueLayout.JAVA_INT, r6 + k + 3));
                dst.set(ValueLayout.JAVA_INT, dOff3 + 7 * 4L, src.getAtIndex(ValueLayout.JAVA_INT, r7 + k + 3));
            }
            for (; k < kc; k++) {
                long dOff = dstBase + (long) k * 8 * 4L;
                dst.set(ValueLayout.JAVA_INT, dOff, src.getAtIndex(ValueLayout.JAVA_INT, r0 + k));
                dst.set(ValueLayout.JAVA_INT, dOff + 4L, src.getAtIndex(ValueLayout.JAVA_INT, r1 + k));
                dst.set(ValueLayout.JAVA_INT, dOff + 2 * 4L, src.getAtIndex(ValueLayout.JAVA_INT, r2 + k));
                dst.set(ValueLayout.JAVA_INT, dOff + 3 * 4L, src.getAtIndex(ValueLayout.JAVA_INT, r3 + k));
                dst.set(ValueLayout.JAVA_INT, dOff + 4 * 4L, src.getAtIndex(ValueLayout.JAVA_INT, r4 + k));
                dst.set(ValueLayout.JAVA_INT, dOff + 5 * 4L, src.getAtIndex(ValueLayout.JAVA_INT, r5 + k));
                dst.set(ValueLayout.JAVA_INT, dOff + 6 * 4L, src.getAtIndex(ValueLayout.JAVA_INT, r6 + k));
                dst.set(ValueLayout.JAVA_INT, dOff + 7 * 4L, src.getAtIndex(ValueLayout.JAVA_INT, r7 + k));
            }
        }
        if (tailRows > 0) {
            long dstBase = (long) fullPanels * 8 * kc * 4L;
            for (int r = 0; r < 8; r++) {
                if (r < tailRows) {
                    long srcRow = (long)(rowStart + fullPanels * 8 + r) * m + colStart;
                    for (int k = 0; k < kc; k++) {
                        dst.set(ValueLayout.JAVA_INT, dstBase + (long) k * 8 * 4L + (long) r * 4L, src.getAtIndex(ValueLayout.JAVA_INT, srcRow + k));
                    }
                } else {
                    for (int k = 0; k < kc; k++) {
                        dst.set(ValueLayout.JAVA_INT, dstBase + (long) k * 8 * 4L + (long) r * 4L, 0);
                    }
                }
            }
        }
    }

    static void packB_panel_Aarch_Int(MemorySegment src, MemorySegment dst, int rowStart, int colStart, int kc, int nc, int p) {
        int fullPanels = nc / 12; int tailCols = nc % 12;
        for (int panel = 0; panel < fullPanels; panel++) {
            long dstBase = (long) panel * 16 * kc * 4L;
            for (int k = 0; k < kc; k++) {
                long srcOff = ((long)(rowStart + k) * p + colStart + (long) panel * 12) * 4L;
                long dstOff = dstBase + (long) k * 16 * 4L;
                MemorySegment.copy(src, srcOff, dst, dstOff, 12L * 4L);
                dst.asSlice(dstOff + 12L * 4L, 4L * 4L).fill((byte) 0);
            }
        }
        if (tailCols > 0) {
            long dstBase = (long) fullPanels * 16 * kc * 4L;
            for (int k = 0; k < kc; k++) {
                long srcOff = ((long)(rowStart + k) * p + colStart + (long) fullPanels * 12) * 4L;
                long dstOff = dstBase + (long) k * 16 * 4L;
                MemorySegment.copy(src, srcOff, dst, dstOff, (long) tailCols * 4L);
                dst.asSlice(dstOff + (long) tailCols * 4L, (long)(16 - tailCols) * 4L).fill((byte) 0);
            }
        }
    }
    static final class GEBPTask_Arm_Int extends RecursiveAction {
        final MemorySegment A, pB, C; int n, m, p_cols, rowStart, rowEnd, pc, kc, jc, nc;
        GEBPTask_Arm_Int(MemorySegment A, MemorySegment pB, MemorySegment C, int n, int m, int p_cols, int rowStart, int rowEnd, int pc, int kc, int jc, int nc) {
            this.A = A; this.pB = pB; this.C = C; this.n = n; this.m = m; this.p_cols = p_cols;
            this.rowStart = rowStart; this.rowEnd = rowEnd; this.pc = pc; this.kc = kc; this.jc = jc; this.nc = nc;
        }
        @Override
        protected void compute() {
            int mc = rowEnd - rowStart;
            if (mc <= MC) {
                MemorySegment pA = tlPackedA_Arm_Int.get();
                packA_panel_Arm_Int(A, pA, rowStart, mc, pc, kc, m);
                gebpMacroKernel_Arm_Int(pA, pB, C, rowStart, mc, jc, nc, kc, p_cols);
            } else {
                int mid = rowStart + (mc / 2) - ((mc / 2) % 6);
                if (mid == rowStart) mid += 6;
                invokeAll(new GEBPTask_Arm_Int(A, pB, C, n, m, p_cols, rowStart, mid, pc, kc, jc, nc), new GEBPTask_Arm_Int(A, pB, C, n, m, p_cols, mid, rowEnd, pc, kc, jc, nc));
            }
        }
    }

    static final class GEBPTask_Aarch_Int extends RecursiveAction {
        final MemorySegment A, pB, C; int n, m, p_cols, rowStart, rowEnd, pc, kc, jc, nc;
        GEBPTask_Aarch_Int(MemorySegment A, MemorySegment pB, MemorySegment C, int n, int m, int p_cols, int rowStart, int rowEnd, int pc, int kc, int jc, int nc) {
            this.A = A; this.pB = pB; this.C = C; this.n = n; this.m = m; this.p_cols = p_cols;
            this.rowStart = rowStart; this.rowEnd = rowEnd; this.pc = pc; this.kc = kc; this.jc = jc; this.nc = nc;
        }
        @Override
        protected void compute() {
            int mc = rowEnd - rowStart;
            if (mc <= MC) {
                MemorySegment pA = tlPackedA_Aarch_Int.get();
                packA_panel_Aarch_Int(A, pA, rowStart, mc, pc, kc, m);
                gebpMacroKernel_Aarch_Int(pA, pB, C, rowStart, mc, jc, nc, kc, p_cols);
            } else {
                int mid = rowStart + (mc / 2) - ((mc / 2) % 8);
                if (mid == rowStart) mid += 8;
                invokeAll(new GEBPTask_Aarch_Int(A, pB, C, n, m, p_cols, rowStart, mid, pc, kc, jc, nc), new GEBPTask_Aarch_Int(A, pB, C, n, m, p_cols, mid, rowEnd, pc, kc, jc, nc));
            }
        }
    }

    private static void gebpMacroKernel_Arm_Int(MemorySegment pA, MemorySegment pB, MemorySegment C, int rowStart, int mc, int jc, int nc, int kc, int p) {
        int nrPanels = (nc + 15) / 16; int fullIPanels = mc / 6;
        for (int jp = 0; jp < nrPanels; jp++) {
            int jr = jp * 16; int actualNR = Math.min(16, nc - jr);
            long bBase = (long) jp * 16 * kc * 4L;
            if (actualNR == 16) {
                for (int ip = 0; ip < fullIPanels; ip++) {
                    microKernel6x16_Int(pA, (long) ip * 6 * kc * 4L, pB, bBase, C, rowStart + ip * 6, jc + jr, kc, p);
                }
                // Issue 3 Resolved: Capture tail rows for full NR bounds
                int tailIRows = mc % 6;
                if (tailIRows > 0) {
                    microKernelScalar_Int(pA, (long) fullIPanels * 6 * kc * 4L, 0, pB, bBase, C, rowStart + fullIPanels * 6, jc + jr, kc, p, tailIRows, 16, 6, 16);
                }
            } else {
                microKernelScalar_Int(pA, (long) fullIPanels * 6 * kc * 4L, 0, pB, bBase, C, rowStart, jc + jr, kc, p, mc, actualNR, 6, 16);
            }
        }
    }

    private static void gebpMacroKernel_Aarch_Int(MemorySegment pA, MemorySegment pB, MemorySegment C, int rowStart, int mc, int jc, int nc, int kc, int p) {
        int nrPanels = (nc + 11) / 12; int fullIPanels = mc / 8;
        for (int jp = 0; jp < nrPanels; jp++) {
            int jr = jp * 12; int actualNR = Math.min(12, nc - jr);
            long bBase = (long) jp * 16 * kc * 4L;
            if (actualNR == 12) {
                for (int ip = 0; ip < fullIPanels; ip++) {
                    microKernel8x12_Int(pA, (long) ip * 8 * kc * 4L, pB, bBase, C, rowStart + ip * 8, jc + jr, kc, p);
                }
                // Issue 3 Resolved: Capture tail rows for full NR bounds
                int tailIRows = mc % 8;
                if (tailIRows > 0) {
                    microKernelScalar_Int(pA, (long) fullIPanels * 8 * kc * 4L, 0, pB, bBase, C, rowStart + fullIPanels * 8, jc + jr, kc, p, tailIRows, 12, 8, 16);
                }
            } else {
                microKernelScalar_Int(pA, (long) fullIPanels * 8 * kc * 4L, 0, pB, bBase, C, rowStart, jc + jr, kc, p, mc, actualNR, 8, 16);
            }
        }
    }

    private static void microKernelScalar_Int(MemorySegment pA, long aBase, int rOff, MemorySegment pB, long bBase, MemorySegment C, int ci, int cj, int kc, int N, int mr, int nr, int MR_dim, int NR_dim) {
        for (int k = 0; k < kc; k++) {
            long aOff = aBase + (long) k * MR_dim * 4L + (long) rOff * 4L;
            long bOff = bBase + (long) k * NR_dim * 4L;
            for (int r = 0; r < mr; r++) {
                int aVal = pA.get(ValueLayout.JAVA_INT, aOff + (long) r * 4L);
                for (int c = 0; c < nr; c++) {
                    long cIdx = (long)(ci + r) * N + cj + c;
                    C.setAtIndex(ValueLayout.JAVA_INT, cIdx, (int)(C.getAtIndex(ValueLayout.JAVA_INT, cIdx) + (aVal * pB.get(ValueLayout.JAVA_INT, bOff + (long) c * 4L))));
                }
            }
        }
    }

    private static void microKernel6x16_Int(MemorySegment pA, long aBase, MemorySegment pB, long bBase, MemorySegment C, int ci, int cj, int kc, int N) {
        var c00 = IntVector.zero(SPECIESINT);
        var c01 = IntVector.zero(SPECIESINT);
        var c10 = IntVector.zero(SPECIESINT);
        var c11 = IntVector.zero(SPECIESINT);
        var c20 = IntVector.zero(SPECIESINT);
        var c21 = IntVector.zero(SPECIESINT);
        var c30 = IntVector.zero(SPECIESINT);
        var c31 = IntVector.zero(SPECIESINT);
        var c40 = IntVector.zero(SPECIESINT);
        var c41 = IntVector.zero(SPECIESINT);
        var c50 = IntVector.zero(SPECIESINT);
        var c51 = IntVector.zero(SPECIESINT);
        for (int k = 0; k < kc; k++) {
            long aOff = aBase + (long) k * 6 * 4L;
            long bOff = bBase + (long) k * 16 * 4L;
            var b0 = IntVector.fromMemorySegment(SPECIESINT, pB, bOff, NATIVE);
            var b1 = IntVector.fromMemorySegment(SPECIESINT, pB, bOff + 8 * 4L, NATIVE);
            var a0 = IntVector.broadcast(SPECIESINT, pA.get(ValueLayout.JAVA_INT, aOff + 0 * 4L));
            c00 = c00.add(a0.mul(b0));
            c01 = c01.add(a0.mul(b1));
            var a1 = IntVector.broadcast(SPECIESINT, pA.get(ValueLayout.JAVA_INT, aOff + 1 * 4L));
            c10 = c10.add(a1.mul(b0));
            c11 = c11.add(a1.mul(b1));
            var a2 = IntVector.broadcast(SPECIESINT, pA.get(ValueLayout.JAVA_INT, aOff + 2 * 4L));
            c20 = c20.add(a2.mul(b0));
            c21 = c21.add(a2.mul(b1));
            var a3 = IntVector.broadcast(SPECIESINT, pA.get(ValueLayout.JAVA_INT, aOff + 3 * 4L));
            c30 = c30.add(a3.mul(b0));
            c31 = c31.add(a3.mul(b1));
            var a4 = IntVector.broadcast(SPECIESINT, pA.get(ValueLayout.JAVA_INT, aOff + 4 * 4L));
            c40 = c40.add(a4.mul(b0));
            c41 = c41.add(a4.mul(b1));
            var a5 = IntVector.broadcast(SPECIESINT, pA.get(ValueLayout.JAVA_INT, aOff + 5 * 4L));
            c50 = c50.add(a5.mul(b0));
            c51 = c51.add(a5.mul(b1));
        }
        long row0 = ((long)(ci + 0) * N + cj) * 4L;
        IntVector.fromMemorySegment(SPECIESINT, C, row0, NATIVE).add(c00).intoMemorySegment(C, row0, NATIVE);
        IntVector.fromMemorySegment(SPECIESINT, C, row0 + 8 * 4L, NATIVE).add(c01).intoMemorySegment(C, row0 + 8 * 4L, NATIVE);
        long row1 = ((long)(ci + 1) * N + cj) * 4L;
        IntVector.fromMemorySegment(SPECIESINT, C, row1, NATIVE).add(c10).intoMemorySegment(C, row1, NATIVE);
        IntVector.fromMemorySegment(SPECIESINT, C, row1 + 8 * 4L, NATIVE).add(c11).intoMemorySegment(C, row1 + 8 * 4L, NATIVE);
        long row2 = ((long)(ci + 2) * N + cj) * 4L;
        IntVector.fromMemorySegment(SPECIESINT, C, row2, NATIVE).add(c20).intoMemorySegment(C, row2, NATIVE);
        IntVector.fromMemorySegment(SPECIESINT, C, row2 + 8 * 4L, NATIVE).add(c21).intoMemorySegment(C, row2 + 8 * 4L, NATIVE);
        long row3 = ((long)(ci + 3) * N + cj) * 4L;
        IntVector.fromMemorySegment(SPECIESINT, C, row3, NATIVE).add(c30).intoMemorySegment(C, row3, NATIVE);
        IntVector.fromMemorySegment(SPECIESINT, C, row3 + 8 * 4L, NATIVE).add(c31).intoMemorySegment(C, row3 + 8 * 4L, NATIVE);
        long row4 = ((long)(ci + 4) * N + cj) * 4L;
        IntVector.fromMemorySegment(SPECIESINT, C, row4, NATIVE).add(c40).intoMemorySegment(C, row4, NATIVE);
        IntVector.fromMemorySegment(SPECIESINT, C, row4 + 8 * 4L, NATIVE).add(c41).intoMemorySegment(C, row4 + 8 * 4L, NATIVE);
        long row5 = ((long)(ci + 5) * N + cj) * 4L;
        IntVector.fromMemorySegment(SPECIESINT, C, row5, NATIVE).add(c50).intoMemorySegment(C, row5, NATIVE);
        IntVector.fromMemorySegment(SPECIESINT, C, row5 + 8 * 4L, NATIVE).add(c51).intoMemorySegment(C, row5 + 8 * 4L, NATIVE);
    }
    private static void microKernel8x12_Int(MemorySegment pA, long aBase, MemorySegment pB, long bBase, MemorySegment C, int ci, int cj, int kc, int N) {
        var c00 = IntVector.zero(SPECIESINT);
        var c01 = IntVector.zero(SPECIESINT);
        var c10 = IntVector.zero(SPECIESINT);
        var c11 = IntVector.zero(SPECIESINT);
        var c20 = IntVector.zero(SPECIESINT);
        var c21 = IntVector.zero(SPECIESINT);
        var c30 = IntVector.zero(SPECIESINT);
        var c31 = IntVector.zero(SPECIESINT);
        var c40 = IntVector.zero(SPECIESINT);
        var c41 = IntVector.zero(SPECIESINT);
        var c50 = IntVector.zero(SPECIESINT);
        var c51 = IntVector.zero(SPECIESINT);
        var c60 = IntVector.zero(SPECIESINT);
        var c61 = IntVector.zero(SPECIESINT);
        var c70 = IntVector.zero(SPECIESINT);
        var c71 = IntVector.zero(SPECIESINT);
        for (int k = 0; k < kc; k++) {
            long aOff = aBase + (long) k * 8 * 4L;
            long bOff = bBase + (long) k * 16 * 4L;
            var b0 = IntVector.fromMemorySegment(SPECIESINT, pB, bOff, NATIVE);
            var b1 = IntVector.fromMemorySegment(SPECIESINT, pB, bOff + 8 * 4L, NATIVE);
            var a0 = IntVector.broadcast(SPECIESINT, pA.get(ValueLayout.JAVA_INT, aOff + 0 * 4L));
            c00 = c00.add(a0.mul(b0));
            c01 = c01.add(a0.mul(b1));
            var a1 = IntVector.broadcast(SPECIESINT, pA.get(ValueLayout.JAVA_INT, aOff + 1 * 4L));
            c10 = c10.add(a1.mul(b0));
            c11 = c11.add(a1.mul(b1));
            var a2 = IntVector.broadcast(SPECIESINT, pA.get(ValueLayout.JAVA_INT, aOff + 2 * 4L));
            c20 = c20.add(a2.mul(b0));
            c21 = c21.add(a2.mul(b1));
            var a3 = IntVector.broadcast(SPECIESINT, pA.get(ValueLayout.JAVA_INT, aOff + 3 * 4L));
            c30 = c30.add(a3.mul(b0));
            c31 = c31.add(a3.mul(b1));
            var a4 = IntVector.broadcast(SPECIESINT, pA.get(ValueLayout.JAVA_INT, aOff + 4 * 4L));
            c40 = c40.add(a4.mul(b0));
            c41 = c41.add(a4.mul(b1));
            var a5 = IntVector.broadcast(SPECIESINT, pA.get(ValueLayout.JAVA_INT, aOff + 5 * 4L));
            c50 = c50.add(a5.mul(b0));
            c51 = c51.add(a5.mul(b1));
            var a6 = IntVector.broadcast(SPECIESINT, pA.get(ValueLayout.JAVA_INT, aOff + 6 * 4L));
            c60 = c60.add(a6.mul(b0));
            c61 = c61.add(a6.mul(b1));
            var a7 = IntVector.broadcast(SPECIESINT, pA.get(ValueLayout.JAVA_INT, aOff + 7 * 4L));
            c70 = c70.add(a7.mul(b0));
            c71 = c71.add(a7.mul(b1));
        }
        long row0 = ((long)(ci + 0) * N + cj) * 4L;
        IntVector.fromMemorySegment(SPECIESINT, C, row0, NATIVE).add(c00).intoMemorySegment(C, row0, NATIVE);
        C.setAtIndex(ValueLayout.JAVA_INT, (long)(ci + 0) * N + cj + 8 + 0, (int)(C.getAtIndex(ValueLayout.JAVA_INT, (long)(ci + 0) * N + cj + 8 + 0) + c01.lane(0)));
        C.setAtIndex(ValueLayout.JAVA_INT, (long)(ci + 0) * N + cj + 8 + 1, (int)(C.getAtIndex(ValueLayout.JAVA_INT, (long)(ci + 0) * N + cj + 8 + 1) + c01.lane(1)));
        C.setAtIndex(ValueLayout.JAVA_INT, (long)(ci + 0) * N + cj + 8 + 2, (int)(C.getAtIndex(ValueLayout.JAVA_INT, (long)(ci + 0) * N + cj + 8 + 2) + c01.lane(2)));
        C.setAtIndex(ValueLayout.JAVA_INT, (long)(ci + 0) * N + cj + 8 + 3, (int)(C.getAtIndex(ValueLayout.JAVA_INT, (long)(ci + 0) * N + cj + 8 + 3) + c01.lane(3)));
        long row1 = ((long)(ci + 1) * N + cj) * 4L;
        IntVector.fromMemorySegment(SPECIESINT, C, row1, NATIVE).add(c10).intoMemorySegment(C, row1, NATIVE);
        C.setAtIndex(ValueLayout.JAVA_INT, (long)(ci + 1) * N + cj + 8 + 0, (int)(C.getAtIndex(ValueLayout.JAVA_INT, (long)(ci + 1) * N + cj + 8 + 0) + c11.lane(0)));
        C.setAtIndex(ValueLayout.JAVA_INT, (long)(ci + 1) * N + cj + 8 + 1, (int)(C.getAtIndex(ValueLayout.JAVA_INT, (long)(ci + 1) * N + cj + 8 + 1) + c11.lane(1)));
        C.setAtIndex(ValueLayout.JAVA_INT, (long)(ci + 1) * N + cj + 8 + 2, (int)(C.getAtIndex(ValueLayout.JAVA_INT, (long)(ci + 1) * N + cj + 8 + 2) + c11.lane(2)));
        C.setAtIndex(ValueLayout.JAVA_INT, (long)(ci + 1) * N + cj + 8 + 3, (int)(C.getAtIndex(ValueLayout.JAVA_INT, (long)(ci + 1) * N + cj + 8 + 3) + c11.lane(3)));
        long row2 = ((long)(ci + 2) * N + cj) * 4L;
        IntVector.fromMemorySegment(SPECIESINT, C, row2, NATIVE).add(c20).intoMemorySegment(C, row2, NATIVE);
        C.setAtIndex(ValueLayout.JAVA_INT, (long)(ci + 2) * N + cj + 8 + 0, (int)(C.getAtIndex(ValueLayout.JAVA_INT, (long)(ci + 2) * N + cj + 8 + 0) + c21.lane(0)));
        C.setAtIndex(ValueLayout.JAVA_INT, (long)(ci + 2) * N + cj + 8 + 1, (int)(C.getAtIndex(ValueLayout.JAVA_INT, (long)(ci + 2) * N + cj + 8 + 1) + c21.lane(1)));
        C.setAtIndex(ValueLayout.JAVA_INT, (long)(ci + 2) * N + cj + 8 + 2, (int)(C.getAtIndex(ValueLayout.JAVA_INT, (long)(ci + 2) * N + cj + 8 + 2) + c21.lane(2)));
        C.setAtIndex(ValueLayout.JAVA_INT, (long)(ci + 2) * N + cj + 8 + 3, (int)(C.getAtIndex(ValueLayout.JAVA_INT, (long)(ci + 2) * N + cj + 8 + 3) + c21.lane(3)));
        long row3 = ((long)(ci + 3) * N + cj) * 4L;
        IntVector.fromMemorySegment(SPECIESINT, C, row3, NATIVE).add(c30).intoMemorySegment(C, row3, NATIVE);
        C.setAtIndex(ValueLayout.JAVA_INT, (long)(ci + 3) * N + cj + 8 + 0, (int)(C.getAtIndex(ValueLayout.JAVA_INT, (long)(ci + 3) * N + cj + 8 + 0) + c31.lane(0)));
        C.setAtIndex(ValueLayout.JAVA_INT, (long)(ci + 3) * N + cj + 8 + 1, (int)(C.getAtIndex(ValueLayout.JAVA_INT, (long)(ci + 3) * N + cj + 8 + 1) + c31.lane(1)));
        C.setAtIndex(ValueLayout.JAVA_INT, (long)(ci + 3) * N + cj + 8 + 2, (int)(C.getAtIndex(ValueLayout.JAVA_INT, (long)(ci + 3) * N + cj + 8 + 2) + c31.lane(2)));
        C.setAtIndex(ValueLayout.JAVA_INT, (long)(ci + 3) * N + cj + 8 + 3, (int)(C.getAtIndex(ValueLayout.JAVA_INT, (long)(ci + 3) * N + cj + 8 + 3) + c31.lane(3)));
        long row4 = ((long)(ci + 4) * N + cj) * 4L;
        IntVector.fromMemorySegment(SPECIESINT, C, row4, NATIVE).add(c40).intoMemorySegment(C, row4, NATIVE);
        C.setAtIndex(ValueLayout.JAVA_INT, (long)(ci + 4) * N + cj + 8 + 0, (int)(C.getAtIndex(ValueLayout.JAVA_INT, (long)(ci + 4) * N + cj + 8 + 0) + c41.lane(0)));
        C.setAtIndex(ValueLayout.JAVA_INT, (long)(ci + 4) * N + cj + 8 + 1, (int)(C.getAtIndex(ValueLayout.JAVA_INT, (long)(ci + 4) * N + cj + 8 + 1) + c41.lane(1)));
        C.setAtIndex(ValueLayout.JAVA_INT, (long)(ci + 4) * N + cj + 8 + 2, (int)(C.getAtIndex(ValueLayout.JAVA_INT, (long)(ci + 4) * N + cj + 8 + 2) + c41.lane(2)));
        C.setAtIndex(ValueLayout.JAVA_INT, (long)(ci + 4) * N + cj + 8 + 3, (int)(C.getAtIndex(ValueLayout.JAVA_INT, (long)(ci + 4) * N + cj + 8 + 3) + c41.lane(3)));
        long row5 = ((long)(ci + 5) * N + cj) * 4L;
        IntVector.fromMemorySegment(SPECIESINT, C, row5, NATIVE).add(c50).intoMemorySegment(C, row5, NATIVE);
        C.setAtIndex(ValueLayout.JAVA_INT, (long)(ci + 5) * N + cj + 8 + 0, (int)(C.getAtIndex(ValueLayout.JAVA_INT, (long)(ci + 5) * N + cj + 8 + 0) + c51.lane(0)));
        C.setAtIndex(ValueLayout.JAVA_INT, (long)(ci + 5) * N + cj + 8 + 1, (int)(C.getAtIndex(ValueLayout.JAVA_INT, (long)(ci + 5) * N + cj + 8 + 1) + c51.lane(1)));
        C.setAtIndex(ValueLayout.JAVA_INT, (long)(ci + 5) * N + cj + 8 + 2, (int)(C.getAtIndex(ValueLayout.JAVA_INT, (long)(ci + 5) * N + cj + 8 + 2) + c51.lane(2)));
        C.setAtIndex(ValueLayout.JAVA_INT, (long)(ci + 5) * N + cj + 8 + 3, (int)(C.getAtIndex(ValueLayout.JAVA_INT, (long)(ci + 5) * N + cj + 8 + 3) + c51.lane(3)));
        long row6 = ((long)(ci + 6) * N + cj) * 4L;
        IntVector.fromMemorySegment(SPECIESINT, C, row6, NATIVE).add(c60).intoMemorySegment(C, row6, NATIVE);
        C.setAtIndex(ValueLayout.JAVA_INT, (long)(ci + 6) * N + cj + 8 + 0, (int)(C.getAtIndex(ValueLayout.JAVA_INT, (long)(ci + 6) * N + cj + 8 + 0) + c61.lane(0)));
        C.setAtIndex(ValueLayout.JAVA_INT, (long)(ci + 6) * N + cj + 8 + 1, (int)(C.getAtIndex(ValueLayout.JAVA_INT, (long)(ci + 6) * N + cj + 8 + 1) + c61.lane(1)));
        C.setAtIndex(ValueLayout.JAVA_INT, (long)(ci + 6) * N + cj + 8 + 2, (int)(C.getAtIndex(ValueLayout.JAVA_INT, (long)(ci + 6) * N + cj + 8 + 2) + c61.lane(2)));
        C.setAtIndex(ValueLayout.JAVA_INT, (long)(ci + 6) * N + cj + 8 + 3, (int)(C.getAtIndex(ValueLayout.JAVA_INT, (long)(ci + 6) * N + cj + 8 + 3) + c61.lane(3)));
        long row7 = ((long)(ci + 7) * N + cj) * 4L;
        IntVector.fromMemorySegment(SPECIESINT, C, row7, NATIVE).add(c70).intoMemorySegment(C, row7, NATIVE);
        C.setAtIndex(ValueLayout.JAVA_INT, (long)(ci + 7) * N + cj + 8 + 0, (int)(C.getAtIndex(ValueLayout.JAVA_INT, (long)(ci + 7) * N + cj + 8 + 0) + c71.lane(0)));
        C.setAtIndex(ValueLayout.JAVA_INT, (long)(ci + 7) * N + cj + 8 + 1, (int)(C.getAtIndex(ValueLayout.JAVA_INT, (long)(ci + 7) * N + cj + 8 + 1) + c71.lane(1)));
        C.setAtIndex(ValueLayout.JAVA_INT, (long)(ci + 7) * N + cj + 8 + 2, (int)(C.getAtIndex(ValueLayout.JAVA_INT, (long)(ci + 7) * N + cj + 8 + 2) + c71.lane(2)));
        C.setAtIndex(ValueLayout.JAVA_INT, (long)(ci + 7) * N + cj + 8 + 3, (int)(C.getAtIndex(ValueLayout.JAVA_INT, (long)(ci + 7) * N + cj + 8 + 3) + c71.lane(3)));
    }

    static class AVX2_Int extends RecursiveAction {
        MemorySegment A, B_T, C; int n, m, p, startRow, endRow;
        AVX2_Int(MemorySegment A, MemorySegment B_T, MemorySegment C, int n, int m, int p, int startRow, int endRow) {
            this.A = A; this.B_T = B_T; this.C = C; this.n = n; this.m = m; this.p = p; this.startRow = startRow; this.endRow = endRow;
        }
        @Override
        protected void compute() {
            if (endRow - startRow <= THRESHOLD) {
                int safeRowEnd = endRow - ((endRow - startRow) % 2); int safeColEnd = p - (p % 2);
                for (int i = startRow; i < safeRowEnd; i += 2) {
                    for (int j = 0; j < safeColEnd; j += 2) {
                        hybridKernel2x2_Int(A, B_T, C, m, p, i, j);
                    }
                }
                if (safeRowEnd < endRow) {
                    for (int j = 0; j < safeColEnd; j++) scalarDotProduct_Int(A, B_T, C, m, p, safeRowEnd, j);
                }
                if (safeColEnd < p) {
                    for (int i = startRow; i < safeRowEnd; i++) scalarDotProduct_Int(A, B_T, C, m, p, i, safeColEnd);
                }
                if (safeRowEnd < endRow && safeColEnd < p) {
                    scalarDotProduct_Int(A, B_T, C, m, p, safeRowEnd, safeColEnd);
                }
            } else {
                int mid = startRow + (endRow - startRow) / 2;
                invokeAll(new AVX2_Int(A, B_T, C, n, m, p, startRow, mid), new AVX2_Int(A, B_T, C, n, m, p, mid, endRow));
            }
        }
    }

    private static void hybridKernel2x2_Int(MemorySegment A, MemorySegment B_T, MemorySegment C, int m, int p, int i, int j) {
        var vSum00 = IntVector.zero(SPECIESINT); var vSum01 = IntVector.zero(SPECIESINT);
        var vSum10 = IntVector.zero(SPECIESINT); var vSum11 = IntVector.zero(SPECIESINT);
        long k = 0; long loopBound = SPECIESINT.loopBound(m);
        for (; k < loopBound; k += SPECIESINT.length()) {
            var vA0 = IntVector.fromMemorySegment(SPECIESINT, A, ((long) i * m + k) * 4L, ByteOrder.nativeOrder());
            var vA1 = IntVector.fromMemorySegment(SPECIESINT, A, ((long) (i + 1) * m + k) * 4L, ByteOrder.nativeOrder());
            var vB0 = IntVector.fromMemorySegment(SPECIESINT, B_T, ((long) j * m + k) * 4L, ByteOrder.nativeOrder());
            var vB1 = IntVector.fromMemorySegment(SPECIESINT, B_T, ((long) (j + 1) * m + k) * 4L, ByteOrder.nativeOrder());
            vSum00 = vSum00.add(vA0.mul(vB0)); vSum01 = vSum01.add(vA0.mul(vB1));
            vSum10 = vSum10.add(vA1.mul(vB0)); vSum11 = vSum11.add(vA1.mul(vB1));
        }
        int sum00 = vSum00.reduceLanes(VectorOperators.ADD); int sum01 = vSum01.reduceLanes(VectorOperators.ADD);
        int sum10 = vSum10.reduceLanes(VectorOperators.ADD); int sum11 = vSum11.reduceLanes(VectorOperators.ADD);
        for (; k < m; k++) {
            int a0 = A.getAtIndex(ValueLayout.JAVA_INT, ((long) i * m + k)); int a1 = A.getAtIndex(ValueLayout.JAVA_INT, ((long) (i + 1) * m + k));
            int b0 = B_T.getAtIndex(ValueLayout.JAVA_INT, ((long) j * m + k)); int b1 = B_T.getAtIndex(ValueLayout.JAVA_INT, ((long) (j + 1) * m + k));
            sum00 += a0 * b0; sum01 += a0 * b1; sum10 += a1 * b0; sum11 += a1 * b1;
        }
        C.setAtIndex(ValueLayout.JAVA_INT, ((long) i * p + j), sum00); C.setAtIndex(ValueLayout.JAVA_INT, ((long) i * p + j + 1), sum01);
        C.setAtIndex(ValueLayout.JAVA_INT, ((long) (i + 1) * p + j), sum10); C.setAtIndex(ValueLayout.JAVA_INT, ((long) (i + 1) * p + j + 1), sum11);
    }

    private static void scalarDotProduct_Int(MemorySegment A, MemorySegment B_T, MemorySegment C, int m, int p, int i, int j) {
        int sum = 0;
        for (int k = 0; k < m; k++) {
            sum += A.getAtIndex(ValueLayout.JAVA_INT, ((long) i * m + k)) * B_T.getAtIndex(ValueLayout.JAVA_INT, ((long) j * m + k));
        }
        C.setAtIndex(ValueLayout.JAVA_INT, ((long) i * p + j), sum);
    }

    // High performance sequential tiled transpose loop to prevent ForkJoin pool thread contention
    private static MemorySegment fastTranspose2D_Int(MemorySegment src, Arena arena, int rows, int cols) {
        MemorySegment dst = arena.allocate((long) rows * cols * 4L);
        int TILE = 64;
        for (int rB = 0; rB < rows; rB += TILE) {
            int rMax = Math.min(rB + TILE, rows);
            for (int cB = 0; cB < cols; cB += TILE) {
                int cMax = Math.min(cB + TILE, cols);
                for (int i = rB; i < rMax; i++) {
                    long iStride = (long) i * cols;
                    for (int j = cB; j < cMax; j++) {
                        dst.setAtIndex(ValueLayout.JAVA_INT, (long) j * rows + i, src.getAtIndex(ValueLayout.JAVA_INT, iStride + j));
                    }
                }
            }
        }
        return dst;
    }


}