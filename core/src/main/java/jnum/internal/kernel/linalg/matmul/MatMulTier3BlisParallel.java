package jnum.internal.kernel.linalg.matmul;

import static jnum.internal.Constants.*;
import static jnum.internal.kernel.linalg.matmul.MatMulPacking.*;
import static jnum.internal.kernel.linalg.matmul.MatMulMicroKernels.*;

import java.lang.foreign.MemorySegment;
import java.util.concurrent.RecursiveAction;
import jnum.NDArray;

/**
 * Tier 3: Parallel BLIS Macro-Kernel (maxDim > 256).
 */
public final class MatMulTier3BlisParallel {

    private MatMulTier3BlisParallel() {
        throw new AssertionError("Utility class");
    }

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
        if (IS_AARCH64) {
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
        if (IS_AARCH64) {
            blisAarchMacro_Int(A, a_s0, a_s1, B, b_s0, b_s1, C, n, m, p);
        } else {
            blisArmMacro_Int(A, a_s0, a_s1, B, b_s0, b_s1, C, n, m, p);
        }
    }

    public static void blisArmMacro_Float(MemorySegment A, long a_s0, long a_s1,
                                            MemorySegment B, long b_s0, long b_s1,
                                            MemorySegment C, int n, int m, int p) {
        if (p >= 2 * NC_ARM && AVAILABLE_CORES >= 2) {
            POOL.invoke(new JcTask_Arm_Float(A, a_s0, a_s1, B, b_s0, b_s1, C, n, m, p, 0, p));
        } else {
            MemorySegment pB = tlPackedB_Arm_Float.get();
            for (int jc = 0; jc < p; jc += NC_ARM) {
                int nc = Math.min(NC_ARM, p - jc);
                for (int pc = 0; pc < m; pc += KC) {
                    int kc = Math.min(KC, m - pc);
                    boolean isFirstKBlock = (pc == 0);
                    packB_panel_Arm_Float(B, pB, pc, jc, kc, nc, b_s0, b_s1);
                    POOL.invoke(new GEBPTask_Arm_Float(A, a_s0, a_s1, pB, C, n, m, p, 0, n, pc, kc, jc, nc, isFirstKBlock));
                }
            }
        }
    }

    public static void blisAarchMacro_Float(MemorySegment A, long a_s0, long a_s1,
                                              MemorySegment B, long b_s0, long b_s1,
                                              MemorySegment C, int n, int m, int p) {
        if (p >= 2 * NC_AARCH && AVAILABLE_CORES >= 2) {
            POOL.invoke(new JcTask_Aarch_Float(A, a_s0, a_s1, B, b_s0, b_s1, C, n, m, p, 0, p));
        } else {
            MemorySegment pB = tlPackedB_Aarch_Float.get();
            for (int jc = 0; jc < p; jc += NC_AARCH) {
                int nc = Math.min(NC_AARCH, p - jc);
                for (int pc = 0; pc < m; pc += KC) {
                    int kc = Math.min(KC, m - pc);
                    boolean isFirstKBlock = (pc == 0);
                    packB_panel_Aarch_Float(B, pB, pc, jc, kc, nc, b_s0, b_s1);
                    POOL.invoke(new GEBPTask_Aarch_Float(A, a_s0, a_s1, pB, C, n, m, p, 0, n, pc, kc, jc, nc, isFirstKBlock));
                }
            }
        }
    }
    public static final class GEBPTask_Arm_Float extends RecursiveAction {
        final MemorySegment A, pB, C;
        final long a_s0, a_s1;
        final int n, m, p_cols, rowStart, rowEnd, pc, kc, jc, nc;
        final boolean isFirstKBlock;

        GEBPTask_Arm_Float(MemorySegment A, long a_s0, long a_s1, MemorySegment pB, MemorySegment C,
                           int n, int m, int p_cols, int rowStart, int rowEnd,
                           int pc, int kc, int jc, int nc, boolean isFirstKBlock) {
            this.A = A; this.a_s0 = a_s0; this.a_s1 = a_s1;
            this.pB = pB; this.C = C; this.n = n; this.m = m; this.p_cols = p_cols;
            this.rowStart = rowStart; this.rowEnd = rowEnd; this.pc = pc; this.kc = kc; this.jc = jc; this.nc = nc;
            this.isFirstKBlock = isFirstKBlock;
        }

        @Override
        protected void compute() {
            int mc = rowEnd - rowStart;
            if (mc <= MC) {
                MemorySegment pA = tlPackedA_Arm_Float.get();
                packA_panel_Arm_Float(A, pA, rowStart, mc, pc, kc, a_s0, a_s1);
                gebpMacroKernel_Arm_Float(pA, pB, C, rowStart, mc, jc, nc, kc, p_cols, isFirstKBlock);
            } else {
                int half = mc / 2;
                half -= half % MR;
                if (half == 0) half = MR;
                int mid = rowStart + half;
                invokeAll(
                    new GEBPTask_Arm_Float(A, a_s0, a_s1, pB, C, n, m, p_cols, rowStart, mid, pc, kc, jc, nc, isFirstKBlock),
                    new GEBPTask_Arm_Float(A, a_s0, a_s1, pB, C, n, m, p_cols, mid, rowEnd, pc, kc, jc, nc, isFirstKBlock)
                );
            }
        }
    }

    public static final class GEBPTask_Aarch_Float extends RecursiveAction {
        final MemorySegment A, pB, C;
        final long a_s0, a_s1;
        final int n, m, p_cols, rowStart, rowEnd, pc, kc, jc, nc;
        final boolean isFirstKBlock;

        GEBPTask_Aarch_Float(MemorySegment A, long a_s0, long a_s1, MemorySegment pB, MemorySegment C,
                             int n, int m, int p_cols, int rowStart, int rowEnd,
                             int pc, int kc, int jc, int nc, boolean isFirstKBlock) {
            this.A = A; this.a_s0 = a_s0; this.a_s1 = a_s1;
            this.pB = pB; this.C = C; this.n = n; this.m = m; this.p_cols = p_cols;
            this.rowStart = rowStart; this.rowEnd = rowEnd; this.pc = pc; this.kc = kc; this.jc = jc; this.nc = nc;
            this.isFirstKBlock = isFirstKBlock;
        }

        @Override
        protected void compute() {
            int mc = rowEnd - rowStart;
            if (mc <= MC) {
                MemorySegment pA = tlPackedA_Aarch_Float.get();
                packA_panel_Aarch_Float(A, pA, rowStart, mc, pc, kc, a_s0, a_s1);
                gebpMacroKernel_Aarch_Float(pA, pB, C, rowStart, mc, jc, nc, kc, p_cols, isFirstKBlock);
            } else {
                int half = mc / 2;
                half -= half % 8;
                if (half == 0) half = 8;
                int mid = rowStart + half;
                invokeAll(
                    new GEBPTask_Aarch_Float(A, a_s0, a_s1, pB, C, n, m, p_cols, rowStart, mid, pc, kc, jc, nc, isFirstKBlock),
                    new GEBPTask_Aarch_Float(A, a_s0, a_s1, pB, C, n, m, p_cols, mid, rowEnd, pc, kc, jc, nc, isFirstKBlock)
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
            int width = colEnd - colStart;
            if (width <= NC_ARM) {
                MemorySegment pB = tlPackedB_Arm_Float.get();
                for (int pc = 0; pc < m; pc += KC) {
                    int kc = Math.min(KC, m - pc);
                    boolean isFirstKBlock = (pc == 0);
                    packB_panel_Arm_Float(B, pB, pc, colStart, kc, width, b_s0, b_s1);
                    POOL.invoke(new GEBPTask_Arm_Float(A, a_s0, a_s1, pB, C, n, m, p, 0, n, pc, kc, colStart, width, isFirstKBlock));
                }
            } else {
                int mid = colStart + width / 2;
                mid -= mid % NR;
                if (mid <= colStart) mid = colStart + NR;
                invokeAll(
                    new JcTask_Arm_Float(A, a_s0, a_s1, B, b_s0, b_s1, C, n, m, p, colStart, mid),
                    new JcTask_Arm_Float(A, a_s0, a_s1, B, b_s0, b_s1, C, n, m, p, mid, colEnd)
                );
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
            int width = colEnd - colStart;
            if (width <= NC_AARCH) {
                MemorySegment pB = tlPackedB_Aarch_Float.get();
                for (int pc = 0; pc < m; pc += KC) {
                    int kc = Math.min(KC, m - pc);
                    boolean isFirstKBlock = (pc == 0);
                    packB_panel_Aarch_Float(B, pB, pc, colStart, kc, width, b_s0, b_s1);
                    POOL.invoke(new GEBPTask_Aarch_Float(A, a_s0, a_s1, pB, C, n, m, p, 0, n, pc, kc, colStart, width, isFirstKBlock));
                }
            } else {
                int mid = colStart + width / 2;
                mid -= mid % 12;
                if (mid <= colStart) mid = colStart + 12;
                invokeAll(
                    new JcTask_Aarch_Float(A, a_s0, a_s1, B, b_s0, b_s1, C, n, m, p, colStart, mid),
                    new JcTask_Aarch_Float(A, a_s0, a_s1, B, b_s0, b_s1, C, n, m, p, mid, colEnd)
                );
            }
        }
    }
    public static void blisArmMacro_Double(MemorySegment A, long a_s0, long a_s1,
                                            MemorySegment B, long b_s0, long b_s1,
                                            MemorySegment C, int n, int m, int p) {
        if (p >= 2 * NC_ARM && AVAILABLE_CORES >= 2) {
            POOL.invoke(new JcTask_Arm_Double(A, a_s0, a_s1, B, b_s0, b_s1, C, n, m, p, 0, p));
        } else {
            MemorySegment pB = tlPackedB_Arm_Double.get();
            for (int jc = 0; jc < p; jc += NC_ARM) {
                int nc = Math.min(NC_ARM, p - jc);
                for (int pc = 0; pc < m; pc += KC) {
                    int kc = Math.min(KC, m - pc);
                    boolean isFirstKBlock = (pc == 0);
                    packB_panel_Arm_Double(B, pB, pc, jc, kc, nc, b_s0, b_s1);
                    POOL.invoke(new GEBPTask_Arm_Double(A, a_s0, a_s1, pB, C, n, m, p, 0, n, pc, kc, jc, nc, isFirstKBlock));
                }
            }
        }
    }

    public static void blisAarchMacro_Double(MemorySegment A, long a_s0, long a_s1,
                                              MemorySegment B, long b_s0, long b_s1,
                                              MemorySegment C, int n, int m, int p) {
        if (p >= 2 * NC_AARCH && AVAILABLE_CORES >= 2) {
            POOL.invoke(new JcTask_Aarch_Double(A, a_s0, a_s1, B, b_s0, b_s1, C, n, m, p, 0, p));
        } else {
            MemorySegment pB = tlPackedB_Aarch_Double.get();
            for (int jc = 0; jc < p; jc += NC_AARCH) {
                int nc = Math.min(NC_AARCH, p - jc);
                for (int pc = 0; pc < m; pc += KC) {
                    int kc = Math.min(KC, m - pc);
                    boolean isFirstKBlock = (pc == 0);
                    packB_panel_Aarch_Double(B, pB, pc, jc, kc, nc, b_s0, b_s1);
                    POOL.invoke(new GEBPTask_Aarch_Double(A, a_s0, a_s1, pB, C, n, m, p, 0, n, pc, kc, jc, nc, isFirstKBlock));
                }
            }
        }
    }
    public static final class GEBPTask_Arm_Double extends RecursiveAction {
        final MemorySegment A, pB, C;
        final long a_s0, a_s1;
        final int n, m, p_cols, rowStart, rowEnd, pc, kc, jc, nc;
        final boolean isFirstKBlock;

        GEBPTask_Arm_Double(MemorySegment A, long a_s0, long a_s1, MemorySegment pB, MemorySegment C,
                           int n, int m, int p_cols, int rowStart, int rowEnd,
                           int pc, int kc, int jc, int nc, boolean isFirstKBlock) {
            this.A = A; this.a_s0 = a_s0; this.a_s1 = a_s1;
            this.pB = pB; this.C = C; this.n = n; this.m = m; this.p_cols = p_cols;
            this.rowStart = rowStart; this.rowEnd = rowEnd; this.pc = pc; this.kc = kc; this.jc = jc; this.nc = nc;
            this.isFirstKBlock = isFirstKBlock;
        }

        @Override
        protected void compute() {
            int mc = rowEnd - rowStart;
            if (mc <= MC) {
                MemorySegment pA = tlPackedA_Arm_Double.get();
                packA_panel_Arm_Double(A, pA, rowStart, mc, pc, kc, a_s0, a_s1);
                gebpMacroKernel_Arm_Double(pA, pB, C, rowStart, mc, jc, nc, kc, p_cols, isFirstKBlock);
            } else {
                int half = mc / 2;
                half -= half % MR;
                if (half == 0) half = MR;
                int mid = rowStart + half;
                invokeAll(
                    new GEBPTask_Arm_Double(A, a_s0, a_s1, pB, C, n, m, p_cols, rowStart, mid, pc, kc, jc, nc, isFirstKBlock),
                    new GEBPTask_Arm_Double(A, a_s0, a_s1, pB, C, n, m, p_cols, mid, rowEnd, pc, kc, jc, nc, isFirstKBlock)
                );
            }
        }
    }

    public static final class GEBPTask_Aarch_Double extends RecursiveAction {
        final MemorySegment A, pB, C;
        final long a_s0, a_s1;
        final int n, m, p_cols, rowStart, rowEnd, pc, kc, jc, nc;
        final boolean isFirstKBlock;

        GEBPTask_Aarch_Double(MemorySegment A, long a_s0, long a_s1, MemorySegment pB, MemorySegment C,
                             int n, int m, int p_cols, int rowStart, int rowEnd,
                             int pc, int kc, int jc, int nc, boolean isFirstKBlock) {
            this.A = A; this.a_s0 = a_s0; this.a_s1 = a_s1;
            this.pB = pB; this.C = C; this.n = n; this.m = m; this.p_cols = p_cols;
            this.rowStart = rowStart; this.rowEnd = rowEnd; this.pc = pc; this.kc = kc; this.jc = jc; this.nc = nc;
            this.isFirstKBlock = isFirstKBlock;
        }

        @Override
        protected void compute() {
            int mc = rowEnd - rowStart;
            if (mc <= MC) {
                MemorySegment pA = tlPackedA_Aarch_Double.get();
                packA_panel_Aarch_Double(A, pA, rowStart, mc, pc, kc, a_s0, a_s1);
                gebpMacroKernel_Aarch_Double(pA, pB, C, rowStart, mc, jc, nc, kc, p_cols, isFirstKBlock);
            } else {
                int half = mc / 2;
                half -= half % 8;
                if (half == 0) half = 8;
                int mid = rowStart + half;
                invokeAll(
                    new GEBPTask_Aarch_Double(A, a_s0, a_s1, pB, C, n, m, p_cols, rowStart, mid, pc, kc, jc, nc, isFirstKBlock),
                    new GEBPTask_Aarch_Double(A, a_s0, a_s1, pB, C, n, m, p_cols, mid, rowEnd, pc, kc, jc, nc, isFirstKBlock)
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
            int width = colEnd - colStart;
            if (width <= NC_ARM) {
                MemorySegment pB = tlPackedB_Arm_Double.get();
                for (int pc = 0; pc < m; pc += KC) {
                    int kc = Math.min(KC, m - pc);
                    boolean isFirstKBlock = (pc == 0);
                    packB_panel_Arm_Double(B, pB, pc, colStart, kc, width, b_s0, b_s1);
                    POOL.invoke(new GEBPTask_Arm_Double(A, a_s0, a_s1, pB, C, n, m, p, 0, n, pc, kc, colStart, width, isFirstKBlock));
                }
            } else {
                int mid = colStart + width / 2;
                mid -= mid % NR_DB;
                if (mid <= colStart) mid = colStart + NR_DB;
                invokeAll(
                    new JcTask_Arm_Double(A, a_s0, a_s1, B, b_s0, b_s1, C, n, m, p, colStart, mid),
                    new JcTask_Arm_Double(A, a_s0, a_s1, B, b_s0, b_s1, C, n, m, p, mid, colEnd)
                );
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
            int width = colEnd - colStart;
            if (width <= NC_AARCH) {
                MemorySegment pB = tlPackedB_Aarch_Double.get();
                for (int pc = 0; pc < m; pc += KC) {
                    int kc = Math.min(KC, m - pc);
                    boolean isFirstKBlock = (pc == 0);
                    packB_panel_Aarch_Double(B, pB, pc, colStart, kc, width, b_s0, b_s1);
                    POOL.invoke(new GEBPTask_Aarch_Double(A, a_s0, a_s1, pB, C, n, m, p, 0, n, pc, kc, colStart, width, isFirstKBlock));
                }
            } else {
                int mid = colStart + width / 2;
                mid -= mid % 12;
                if (mid <= colStart) mid = colStart + 12;
                invokeAll(
                    new JcTask_Aarch_Double(A, a_s0, a_s1, B, b_s0, b_s1, C, n, m, p, colStart, mid),
                    new JcTask_Aarch_Double(A, a_s0, a_s1, B, b_s0, b_s1, C, n, m, p, mid, colEnd)
                );
            }
        }
    }
    public static void blisArmMacro_Int(MemorySegment A, long a_s0, long a_s1,
                                            MemorySegment B, long b_s0, long b_s1,
                                            MemorySegment C, int n, int m, int p) {
        if (p >= 2 * NC_ARM && AVAILABLE_CORES >= 2) {
            POOL.invoke(new JcTask_Arm_Int(A, a_s0, a_s1, B, b_s0, b_s1, C, n, m, p, 0, p));
        } else {
            MemorySegment pB = tlPackedB_Arm_Int.get();
            for (int jc = 0; jc < p; jc += NC_ARM) {
                int nc = Math.min(NC_ARM, p - jc);
                for (int pc = 0; pc < m; pc += KC) {
                    int kc = Math.min(KC, m - pc);
                    boolean isFirstKBlock = (pc == 0);
                    packB_panel_Arm_Int(B, pB, pc, jc, kc, nc, b_s0, b_s1);
                    POOL.invoke(new GEBPTask_Arm_Int(A, a_s0, a_s1, pB, C, n, m, p, 0, n, pc, kc, jc, nc, isFirstKBlock));
                }
            }
        }
    }

    public static void blisAarchMacro_Int(MemorySegment A, long a_s0, long a_s1,
                                              MemorySegment B, long b_s0, long b_s1,
                                              MemorySegment C, int n, int m, int p) {
        if (p >= 2 * NC_AARCH && AVAILABLE_CORES >= 2) {
            POOL.invoke(new JcTask_Aarch_Int(A, a_s0, a_s1, B, b_s0, b_s1, C, n, m, p, 0, p));
        } else {
            MemorySegment pB = tlPackedB_Aarch_Int.get();
            for (int jc = 0; jc < p; jc += NC_AARCH) {
                int nc = Math.min(NC_AARCH, p - jc);
                for (int pc = 0; pc < m; pc += KC) {
                    int kc = Math.min(KC, m - pc);
                    boolean isFirstKBlock = (pc == 0);
                    packB_panel_Aarch_Int(B, pB, pc, jc, kc, nc, b_s0, b_s1);
                    POOL.invoke(new GEBPTask_Aarch_Int(A, a_s0, a_s1, pB, C, n, m, p, 0, n, pc, kc, jc, nc, isFirstKBlock));
                }
            }
        }
    }
    public static final class GEBPTask_Arm_Int extends RecursiveAction {
        final MemorySegment A, pB, C;
        final long a_s0, a_s1;
        final int n, m, p_cols, rowStart, rowEnd, pc, kc, jc, nc;
        final boolean isFirstKBlock;

        GEBPTask_Arm_Int(MemorySegment A, long a_s0, long a_s1, MemorySegment pB, MemorySegment C,
                           int n, int m, int p_cols, int rowStart, int rowEnd,
                           int pc, int kc, int jc, int nc, boolean isFirstKBlock) {
            this.A = A; this.a_s0 = a_s0; this.a_s1 = a_s1;
            this.pB = pB; this.C = C; this.n = n; this.m = m; this.p_cols = p_cols;
            this.rowStart = rowStart; this.rowEnd = rowEnd; this.pc = pc; this.kc = kc; this.jc = jc; this.nc = nc;
            this.isFirstKBlock = isFirstKBlock;
        }

        @Override
        protected void compute() {
            int mc = rowEnd - rowStart;
            if (mc <= MC) {
                MemorySegment pA = tlPackedA_Arm_Int.get();
                packA_panel_Arm_Int(A, pA, rowStart, mc, pc, kc, a_s0, a_s1);
                gebpMacroKernel_Arm_Int(pA, pB, C, rowStart, mc, jc, nc, kc, p_cols, isFirstKBlock);
            } else {
                int half = mc / 2;
                half -= half % MR;
                if (half == 0) half = MR;
                int mid = rowStart + half;
                invokeAll(
                    new GEBPTask_Arm_Int(A, a_s0, a_s1, pB, C, n, m, p_cols, rowStart, mid, pc, kc, jc, nc, isFirstKBlock),
                    new GEBPTask_Arm_Int(A, a_s0, a_s1, pB, C, n, m, p_cols, mid, rowEnd, pc, kc, jc, nc, isFirstKBlock)
                );
            }
        }
    }

    public static final class GEBPTask_Aarch_Int extends RecursiveAction {
        final MemorySegment A, pB, C;
        final long a_s0, a_s1;
        final int n, m, p_cols, rowStart, rowEnd, pc, kc, jc, nc;
        final boolean isFirstKBlock;

        GEBPTask_Aarch_Int(MemorySegment A, long a_s0, long a_s1, MemorySegment pB, MemorySegment C,
                             int n, int m, int p_cols, int rowStart, int rowEnd,
                             int pc, int kc, int jc, int nc, boolean isFirstKBlock) {
            this.A = A; this.a_s0 = a_s0; this.a_s1 = a_s1;
            this.pB = pB; this.C = C; this.n = n; this.m = m; this.p_cols = p_cols;
            this.rowStart = rowStart; this.rowEnd = rowEnd; this.pc = pc; this.kc = kc; this.jc = jc; this.nc = nc;
            this.isFirstKBlock = isFirstKBlock;
        }

        @Override
        protected void compute() {
            int mc = rowEnd - rowStart;
            if (mc <= MC) {
                MemorySegment pA = tlPackedA_Aarch_Int.get();
                packA_panel_Aarch_Int(A, pA, rowStart, mc, pc, kc, a_s0, a_s1);
                gebpMacroKernel_Aarch_Int(pA, pB, C, rowStart, mc, jc, nc, kc, p_cols, isFirstKBlock);
            } else {
                int half = mc / 2;
                half -= half % 8;
                if (half == 0) half = 8;
                int mid = rowStart + half;
                invokeAll(
                    new GEBPTask_Aarch_Int(A, a_s0, a_s1, pB, C, n, m, p_cols, rowStart, mid, pc, kc, jc, nc, isFirstKBlock),
                    new GEBPTask_Aarch_Int(A, a_s0, a_s1, pB, C, n, m, p_cols, mid, rowEnd, pc, kc, jc, nc, isFirstKBlock)
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
            int width = colEnd - colStart;
            if (width <= NC_ARM) {
                MemorySegment pB = tlPackedB_Arm_Int.get();
                for (int pc = 0; pc < m; pc += KC) {
                    int kc = Math.min(KC, m - pc);
                    boolean isFirstKBlock = (pc == 0);
                    packB_panel_Arm_Int(B, pB, pc, colStart, kc, width, b_s0, b_s1);
                    POOL.invoke(new GEBPTask_Arm_Int(A, a_s0, a_s1, pB, C, n, m, p, 0, n, pc, kc, colStart, width, isFirstKBlock));
                }
            } else {
                int mid = colStart + width / 2;
                mid -= mid % NR_INT;
                if (mid <= colStart) mid = colStart + NR_INT;
                invokeAll(
                    new JcTask_Arm_Int(A, a_s0, a_s1, B, b_s0, b_s1, C, n, m, p, colStart, mid),
                    new JcTask_Arm_Int(A, a_s0, a_s1, B, b_s0, b_s1, C, n, m, p, mid, colEnd)
                );
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
            int width = colEnd - colStart;
            if (width <= NC_AARCH) {
                MemorySegment pB = tlPackedB_Aarch_Int.get();
                for (int pc = 0; pc < m; pc += KC) {
                    int kc = Math.min(KC, m - pc);
                    boolean isFirstKBlock = (pc == 0);
                    packB_panel_Aarch_Int(B, pB, pc, colStart, kc, width, b_s0, b_s1);
                    POOL.invoke(new GEBPTask_Aarch_Int(A, a_s0, a_s1, pB, C, n, m, p, 0, n, pc, kc, colStart, width, isFirstKBlock));
                }
            } else {
                int mid = colStart + width / 2;
                mid -= mid % 12;
                if (mid <= colStart) mid = colStart + 12;
                invokeAll(
                    new JcTask_Aarch_Int(A, a_s0, a_s1, B, b_s0, b_s1, C, n, m, p, colStart, mid),
                    new JcTask_Aarch_Int(A, a_s0, a_s1, B, b_s0, b_s1, C, n, m, p, mid, colEnd)
                );
            }
        }
    }
}