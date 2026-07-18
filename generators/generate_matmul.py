import os

TYPE_MAPPINGS = [
    {
        "Title": "Float", "primitive": "float", "VectorClass": "FloatVector",
        "Species": "SPECIES", "Layout": "ValueLayout.JAVA_FLOAT",
        "Bytes": "4L", "Zero": "0f",
        "FMA": "{c} = {a}.fma({b}, {c})",
        "FMA00": "vSum00 = vA0.fma(vB0, vSum00)", "FMA01": "vSum01 = vA0.fma(vB1, vSum01)",
        "FMA10": "vSum10 = vA1.fma(vB0, vSum10)", "FMA11": "vSum11 = vA1.fma(vB1, vSum11)"
    },
    {
        "Title": "Double", "primitive": "double", "VectorClass": "DoubleVector",
        "Species": "SPECIESDB", "Layout": "ValueLayout.JAVA_DOUBLE",
        "Bytes": "8L", "Zero": "0.0",
        "FMA": "{c} = {a}.fma({b}, {c})",
        "FMA00": "vSum00 = vA0.fma(vB0, vSum00)", "FMA01": "vSum01 = vA0.fma(vB1, vSum01)",
        "FMA10": "vSum10 = vA1.fma(vB0, vSum10)", "FMA11": "vSum11 = vA1.fma(vB1, vSum11)"
    },
    {
        "Title": "Int", "primitive": "int", "VectorClass": "IntVector",
        "Species": "SPECIESINT", "Layout": "ValueLayout.JAVA_INT",
        "Bytes": "4L", "Zero": "0",
        "FMA": "{c} = {c}.add({a}.mul({b}))",
        "FMA00": "vSum00 = vSum00.add(vA0.mul(vB0))", "FMA01": "vSum01 = vSum01.add(vA0.mul(vB1))",
        "FMA10": "vSum10 = vSum10.add(vA1.mul(vB0))", "FMA11": "vSum11 = vSum11.add(vA1.mul(vB1))"
    }
]

ROUTER_TEMPLATE = """
    public static NDArray matmul<Title>(NDArray a, NDArray b, NDArray resArray) {
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
                    blisAarchMacro_<Title>(memA, memB, memC, n, m, p, arena);
                } else {
                    blisArmMacro_<Title>(memA, memB, memC, n, m, p, arena);
                }
            } else {
                MemorySegment memB_T = fastTranspose2D_<Title>(memB, arena, m, p);
                POOL.invoke(new AVX2_<Title>(memA, memB_T, memC, n, m, p, 0, n));
            }
        }
        return resArray;
    }
"""

TILING_PIPELINE_TEMPLATE = """
    private static final ThreadLocal<MemorySegment> tlPackedA_Arm_<Title> =
        ThreadLocal.withInitial(() -> Arena.ofAuto().allocate((long) MC * KC * <Bytes>, 64));
    private static final ThreadLocal<MemorySegment> tlPackedA_Aarch_<Title> =
        ThreadLocal.withInitial(() -> Arena.ofAuto().allocate((long) MC * KC * <Bytes>, 64));

    private static void blisArmMacro_<Title>(MemorySegment A, MemorySegment B, MemorySegment C, int n, int m, int p, Arena arena) {
        C.fill((byte) 0);
        MemorySegment pB = arena.allocate((long) KC * NC_ARM * <Bytes>, 64);
        for (int jc = 0; jc < p; jc += NC_ARM) {
            int nc = Math.min(NC_ARM, p - jc);
            for (int pc = 0; pc < m; pc += KC) {
                int kc = Math.min(KC, m - pc);
                packB_panel_Arm_<Title>(B, pB, pc, jc, kc, nc, p);
                POOL.invoke(new GEBPTask_Arm_<Title>(A, pB, C, n, m, p, 0, n, pc, kc, jc, nc));
            }
        }
    }

    private static void blisAarchMacro_<Title>(MemorySegment A, MemorySegment B, MemorySegment C, int n, int m, int p, Arena arena) {
        C.fill((byte) 0);
        MemorySegment pB = arena.allocate((long) KC * NC_AARCH * <Bytes>, 64);
        for (int jc = 0; jc < p; jc += NC_AARCH) {
            int nc = Math.min(NC_AARCH, p - jc);
            for (int pc = 0; pc < m; pc += KC) {
                int kc = Math.min(KC, m - pc);
                packB_panel_Aarch_<Title>(B, pB, pc, jc, kc, nc, p);
                POOL.invoke(new GEBPTask_Aarch_<Title>(A, pB, C, n, m, p, 0, n, pc, kc, jc, nc));
            }
        }
    }

    static void packA_panel_Arm_<Title>(MemorySegment src, MemorySegment dst, int rowStart, int mc, int colStart, int kc, int m) {
        int fullPanels = mc / 6; int tailRows = mc % 6;
        for (int panel = 0; panel < fullPanels; panel++) {
            long dstBase = (long) panel * 6 * kc * <Bytes>;
            long r0 = (long)(rowStart + panel * 6 + 0) * m + colStart;
            long r1 = (long)(rowStart + panel * 6 + 1) * m + colStart;
            long r2 = (long)(rowStart + panel * 6 + 2) * m + colStart;
            long r3 = (long)(rowStart + panel * 6 + 3) * m + colStart;
            long r4 = (long)(rowStart + panel * 6 + 4) * m + colStart;
            long r5 = (long)(rowStart + panel * 6 + 5) * m + colStart;
            
            int k = 0;
            for (; k <= kc - 4; k += 4) {
                long dOff = dstBase + (long) k * 6 * <Bytes>;
                dst.set(<Layout>, dOff, src.getAtIndex(<Layout>, r0 + k));
                dst.set(<Layout>, dOff + <Bytes>, src.getAtIndex(<Layout>, r1 + k));
                dst.set(<Layout>, dOff + 2 * <Bytes>, src.getAtIndex(<Layout>, r2 + k));
                dst.set(<Layout>, dOff + 3 * <Bytes>, src.getAtIndex(<Layout>, r3 + k));
                dst.set(<Layout>, dOff + 4 * <Bytes>, src.getAtIndex(<Layout>, r4 + k));
                dst.set(<Layout>, dOff + 5 * <Bytes>, src.getAtIndex(<Layout>, r5 + k));
                
                long dOff1 = dOff + 6 * <Bytes>;
                dst.set(<Layout>, dOff1, src.getAtIndex(<Layout>, r0 + k + 1));
                dst.set(<Layout>, dOff1 + <Bytes>, src.getAtIndex(<Layout>, r1 + k + 1));
                dst.set(<Layout>, dOff1 + 2 * <Bytes>, src.getAtIndex(<Layout>, r2 + k + 1));
                dst.set(<Layout>, dOff1 + 3 * <Bytes>, src.getAtIndex(<Layout>, r3 + k + 1));
                dst.set(<Layout>, dOff1 + 4 * <Bytes>, src.getAtIndex(<Layout>, r4 + k + 1));
                dst.set(<Layout>, dOff1 + 5 * <Bytes>, src.getAtIndex(<Layout>, r5 + k + 1));

                long dOff2 = dOff1 + 6 * <Bytes>;
                dst.set(<Layout>, dOff2, src.getAtIndex(<Layout>, r0 + k + 2));
                dst.set(<Layout>, dOff2 + <Bytes>, src.getAtIndex(<Layout>, r1 + k + 2));
                dst.set(<Layout>, dOff2 + 2 * <Bytes>, src.getAtIndex(<Layout>, r2 + k + 2));
                dst.set(<Layout>, dOff2 + 3 * <Bytes>, src.getAtIndex(<Layout>, r3 + k + 2));
                dst.set(<Layout>, dOff2 + 4 * <Bytes>, src.getAtIndex(<Layout>, r4 + k + 2));
                dst.set(<Layout>, dOff2 + 5 * <Bytes>, src.getAtIndex(<Layout>, r5 + k + 2));

                long dOff3 = dOff2 + 6 * <Bytes>;
                dst.set(<Layout>, dOff3, src.getAtIndex(<Layout>, r0 + k + 3));
                dst.set(<Layout>, dOff3 + <Bytes>, src.getAtIndex(<Layout>, r1 + k + 3));
                dst.set(<Layout>, dOff3 + 2 * <Bytes>, src.getAtIndex(<Layout>, r2 + k + 3));
                dst.set(<Layout>, dOff3 + 3 * <Bytes>, src.getAtIndex(<Layout>, r3 + k + 3));
                dst.set(<Layout>, dOff3 + 4 * <Bytes>, src.getAtIndex(<Layout>, r4 + k + 3));
                dst.set(<Layout>, dOff3 + 5 * <Bytes>, src.getAtIndex(<Layout>, r5 + k + 3));
            }
            for (; k < kc; k++) {
                long dOff = dstBase + (long) k * 6 * <Bytes>;
                dst.set(<Layout>, dOff, src.getAtIndex(<Layout>, r0 + k));
                dst.set(<Layout>, dOff + <Bytes>, src.getAtIndex(<Layout>, r1 + k));
                dst.set(<Layout>, dOff + 2 * <Bytes>, src.getAtIndex(<Layout>, r2 + k));
                dst.set(<Layout>, dOff + 3 * <Bytes>, src.getAtIndex(<Layout>, r3 + k));
                dst.set(<Layout>, dOff + 4 * <Bytes>, src.getAtIndex(<Layout>, r4 + k));
                dst.set(<Layout>, dOff + 5 * <Bytes>, src.getAtIndex(<Layout>, r5 + k));
            }
        }
        if (tailRows > 0) {
            long dstBase = (long) fullPanels * 6 * kc * <Bytes>;
            for (int r = 0; r < 6; r++) {
                if (r < tailRows) {
                    long srcRow = (long)(rowStart + fullPanels * 6 + r) * m + colStart;
                    for (int k = 0; k < kc; k++) {
                        dst.set(<Layout>, dstBase + (long) k * 6 * <Bytes> + (long) r * <Bytes>, src.getAtIndex(<Layout>, srcRow + k));
                    }
                } else {
                    for (int k = 0; k < kc; k++) {
                        dst.set(<Layout>, dstBase + (long) k * 6 * <Bytes> + (long) r * <Bytes>, <Zero>);
                    }
                }
            }
        }
    }

    static void packB_panel_Arm_<Title>(MemorySegment src, MemorySegment dst, int rowStart, int colStart, int kc, int nc, int p) {
        int fullPanels = nc / 16; int tailCols = nc % 16;
        for (int panel = 0; panel < fullPanels; panel++) {
            long dstBase = (long) panel * 16 * kc * <Bytes>;
            for (int k = 0; k < kc; k++) {
                long srcOff = ((long)(rowStart + k) * p + colStart + (long) panel * 16) * <Bytes>;
                MemorySegment.copy(src, srcOff, dst, dstBase + (long) k * 16 * <Bytes>, 16L * <Bytes>);
            }
        }
        if (tailCols > 0) {
            long dstBase = (long) fullPanels * 16 * kc * <Bytes>;
            for (int k = 0; k < kc; k++) {
                long srcOff = ((long)(rowStart + k) * p + colStart + (long) fullPanels * 16) * <Bytes>;
                long dstOff = dstBase + (long) k * 16 * <Bytes>;
                MemorySegment.copy(src, srcOff, dst, dstOff, (long) tailCols * <Bytes>);
                dst.asSlice(dstOff + (long) tailCols * <Bytes>, (long)(16 - tailCols) * <Bytes>).fill((byte) 0);
            }
        }
    }

    static void packA_panel_Aarch_<Title>(MemorySegment src, MemorySegment dst, int rowStart, int mc, int colStart, int kc, int m) {
        int fullPanels = mc / 8; int tailRows = mc % 8;
        for (int panel = 0; panel < fullPanels; panel++) {
            long dstBase = (long) panel * 8 * kc * <Bytes>;
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
                long dOff = dstBase + (long) k * 8 * <Bytes>;
                dst.set(<Layout>, dOff, src.getAtIndex(<Layout>, r0 + k));
                dst.set(<Layout>, dOff + <Bytes>, src.getAtIndex(<Layout>, r1 + k));
                dst.set(<Layout>, dOff + 2 * <Bytes>, src.getAtIndex(<Layout>, r2 + k));
                dst.set(<Layout>, dOff + 3 * <Bytes>, src.getAtIndex(<Layout>, r3 + k));
                dst.set(<Layout>, dOff + 4 * <Bytes>, src.getAtIndex(<Layout>, r4 + k));
                dst.set(<Layout>, dOff + 5 * <Bytes>, src.getAtIndex(<Layout>, r5 + k));
                dst.set(<Layout>, dOff + 6 * <Bytes>, src.getAtIndex(<Layout>, r6 + k));
                dst.set(<Layout>, dOff + 7 * <Bytes>, src.getAtIndex(<Layout>, r7 + k));

                long dOff1 = dOff + 8 * <Bytes>;
                dst.set(<Layout>, dOff1, src.getAtIndex(<Layout>, r0 + k + 1));
                dst.set(<Layout>, dOff1 + <Bytes>, src.getAtIndex(<Layout>, r1 + k + 1));
                dst.set(<Layout>, dOff1 + 2 * <Bytes>, src.getAtIndex(<Layout>, r2 + k + 1));
                dst.set(<Layout>, dOff1 + 3 * <Bytes>, src.getAtIndex(<Layout>, r3 + k + 1));
                dst.set(<Layout>, dOff1 + 4 * <Bytes>, src.getAtIndex(<Layout>, r4 + k + 1));
                dst.set(<Layout>, dOff1 + 5 * <Bytes>, src.getAtIndex(<Layout>, r5 + k + 1));
                dst.set(<Layout>, dOff1 + 6 * <Bytes>, src.getAtIndex(<Layout>, r6 + k + 1));
                dst.set(<Layout>, dOff1 + 7 * <Bytes>, src.getAtIndex(<Layout>, r7 + k + 1));

                long dOff2 = dOff1 + 8 * <Bytes>;
                dst.set(<Layout>, dOff2, src.getAtIndex(<Layout>, r0 + k + 2));
                dst.set(<Layout>, dOff2 + <Bytes>, src.getAtIndex(<Layout>, r1 + k + 2));
                dst.set(<Layout>, dOff2 + 2 * <Bytes>, src.getAtIndex(<Layout>, r2 + k + 2));
                dst.set(<Layout>, dOff2 + 3 * <Bytes>, src.getAtIndex(<Layout>, r3 + k + 2));
                dst.set(<Layout>, dOff2 + 4 * <Bytes>, src.getAtIndex(<Layout>, r4 + k + 2));
                dst.set(<Layout>, dOff2 + 5 * <Bytes>, src.getAtIndex(<Layout>, r5 + k + 2));
                dst.set(<Layout>, dOff2 + 6 * <Bytes>, src.getAtIndex(<Layout>, r6 + k + 2));
                dst.set(<Layout>, dOff2 + 7 * <Bytes>, src.getAtIndex(<Layout>, r7 + k + 2));

                long dOff3 = dOff2 + 8 * <Bytes>;
                dst.set(<Layout>, dOff3, src.getAtIndex(<Layout>, r0 + k + 3));
                dst.set(<Layout>, dOff3 + <Bytes>, src.getAtIndex(<Layout>, r1 + k + 3));
                dst.set(<Layout>, dOff3 + 2 * <Bytes>, src.getAtIndex(<Layout>, r2 + k + 3));
                dst.set(<Layout>, dOff3 + 3 * <Bytes>, src.getAtIndex(<Layout>, r3 + k + 3));
                dst.set(<Layout>, dOff3 + 4 * <Bytes>, src.getAtIndex(<Layout>, r4 + k + 3));
                dst.set(<Layout>, dOff3 + 5 * <Bytes>, src.getAtIndex(<Layout>, r5 + k + 3));
                dst.set(<Layout>, dOff3 + 6 * <Bytes>, src.getAtIndex(<Layout>, r6 + k + 3));
                dst.set(<Layout>, dOff3 + 7 * <Bytes>, src.getAtIndex(<Layout>, r7 + k + 3));
            }
            for (; k < kc; k++) {
                long dOff = dstBase + (long) k * 8 * <Bytes>;
                dst.set(<Layout>, dOff, src.getAtIndex(<Layout>, r0 + k));
                dst.set(<Layout>, dOff + <Bytes>, src.getAtIndex(<Layout>, r1 + k));
                dst.set(<Layout>, dOff + 2 * <Bytes>, src.getAtIndex(<Layout>, r2 + k));
                dst.set(<Layout>, dOff + 3 * <Bytes>, src.getAtIndex(<Layout>, r3 + k));
                dst.set(<Layout>, dOff + 4 * <Bytes>, src.getAtIndex(<Layout>, r4 + k));
                dst.set(<Layout>, dOff + 5 * <Bytes>, src.getAtIndex(<Layout>, r5 + k));
                dst.set(<Layout>, dOff + 6 * <Bytes>, src.getAtIndex(<Layout>, r6 + k));
                dst.set(<Layout>, dOff + 7 * <Bytes>, src.getAtIndex(<Layout>, r7 + k));
            }
        }
        if (tailRows > 0) {
            long dstBase = (long) fullPanels * 8 * kc * <Bytes>;
            for (int r = 0; r < 8; r++) {
                if (r < tailRows) {
                    long srcRow = (long)(rowStart + fullPanels * 8 + r) * m + colStart;
                    for (int k = 0; k < kc; k++) {
                        dst.set(<Layout>, dstBase + (long) k * 8 * <Bytes> + (long) r * <Bytes>, src.getAtIndex(<Layout>, srcRow + k));
                    }
                } else {
                    for (int k = 0; k < kc; k++) {
                        dst.set(<Layout>, dstBase + (long) k * 8 * <Bytes> + (long) r * <Bytes>, <Zero>);
                    }
                }
            }
        }
    }

    static void packB_panel_Aarch_<Title>(MemorySegment src, MemorySegment dst, int rowStart, int colStart, int kc, int nc, int p) {
        int fullPanels = nc / 12; int tailCols = nc % 12;
        for (int panel = 0; panel < fullPanels; panel++) {
            long dstBase = (long) panel * 16 * kc * <Bytes>;
            for (int k = 0; k < kc; k++) {
                long srcOff = ((long)(rowStart + k) * p + colStart + (long) panel * 12) * <Bytes>;
                long dstOff = dstBase + (long) k * 16 * <Bytes>;
                MemorySegment.copy(src, srcOff, dst, dstOff, 12L * <Bytes>);
                dst.asSlice(dstOff + 12L * <Bytes>, 4L * <Bytes>).fill((byte) 0);
            }
        }
        if (tailCols > 0) {
            long dstBase = (long) fullPanels * 16 * kc * <Bytes>;
            for (int k = 0; k < kc; k++) {
                long srcOff = ((long)(rowStart + k) * p + colStart + (long) fullPanels * 12) * <Bytes>;
                long dstOff = dstBase + (long) k * 16 * <Bytes>;
                MemorySegment.copy(src, srcOff, dst, dstOff, (long) tailCols * <Bytes>);
                dst.asSlice(dstOff + (long) tailCols * <Bytes>, (long)(16 - tailCols) * <Bytes>).fill((byte) 0);
            }
        }
    }
    static final class GEBPTask_Arm_<Title> extends RecursiveAction {
        final MemorySegment A, pB, C; int n, m, p_cols, rowStart, rowEnd, pc, kc, jc, nc;
        GEBPTask_Arm_<Title>(MemorySegment A, MemorySegment pB, MemorySegment C, int n, int m, int p_cols, int rowStart, int rowEnd, int pc, int kc, int jc, int nc) {
            this.A = A; this.pB = pB; this.C = C; this.n = n; this.m = m; this.p_cols = p_cols;
            this.rowStart = rowStart; this.rowEnd = rowEnd; this.pc = pc; this.kc = kc; this.jc = jc; this.nc = nc;
        }
        @Override
        protected void compute() {
            int mc = rowEnd - rowStart;
            if (mc <= MC) {
                MemorySegment pA = tlPackedA_Arm_<Title>.get();
                packA_panel_Arm_<Title>(A, pA, rowStart, mc, pc, kc, m);
                gebpMacroKernel_Arm_<Title>(pA, pB, C, rowStart, mc, jc, nc, kc, p_cols);
            } else {
                int mid = rowStart + (mc / 2) - ((mc / 2) % 6);
                if (mid == rowStart) mid += 6;
                invokeAll(new GEBPTask_Arm_<Title>(A, pB, C, n, m, p_cols, rowStart, mid, pc, kc, jc, nc), new GEBPTask_Arm_<Title>(A, pB, C, n, m, p_cols, mid, rowEnd, pc, kc, jc, nc));
            }
        }
    }

    static final class GEBPTask_Aarch_<Title> extends RecursiveAction {
        final MemorySegment A, pB, C; int n, m, p_cols, rowStart, rowEnd, pc, kc, jc, nc;
        GEBPTask_Aarch_<Title>(MemorySegment A, MemorySegment pB, MemorySegment C, int n, int m, int p_cols, int rowStart, int rowEnd, int pc, int kc, int jc, int nc) {
            this.A = A; this.pB = pB; this.C = C; this.n = n; this.m = m; this.p_cols = p_cols;
            this.rowStart = rowStart; this.rowEnd = rowEnd; this.pc = pc; this.kc = kc; this.jc = jc; this.nc = nc;
        }
        @Override
        protected void compute() {
            int mc = rowEnd - rowStart;
            if (mc <= MC) {
                MemorySegment pA = tlPackedA_Aarch_<Title>.get();
                packA_panel_Aarch_<Title>(A, pA, rowStart, mc, pc, kc, m);
                gebpMacroKernel_Aarch_<Title>(pA, pB, C, rowStart, mc, jc, nc, kc, p_cols);
            } else {
                int mid = rowStart + (mc / 2) - ((mc / 2) % 8);
                if (mid == rowStart) mid += 8;
                invokeAll(new GEBPTask_Aarch_<Title>(A, pB, C, n, m, p_cols, rowStart, mid, pc, kc, jc, nc), new GEBPTask_Aarch_<Title>(A, pB, C, n, m, p_cols, mid, rowEnd, pc, kc, jc, nc));
            }
        }
    }

    private static void gebpMacroKernel_Arm_<Title>(MemorySegment pA, MemorySegment pB, MemorySegment C, int rowStart, int mc, int jc, int nc, int kc, int p) {
        int nrPanels = (nc + 15) / 16; int fullIPanels = mc / 6;
        for (int jp = 0; jp < nrPanels; jp++) {
            int jr = jp * 16; int actualNR = Math.min(16, nc - jr);
            long bBase = (long) jp * 16 * kc * <Bytes>;
            if (actualNR == 16) {
                for (int ip = 0; ip < fullIPanels; ip++) {
                    microKernel6x16_<Title>(pA, (long) ip * 6 * kc * <Bytes>, pB, bBase, C, rowStart + ip * 6, jc + jr, kc, p);
                }
                // Issue 3 Resolved: Capture tail rows for full NR bounds
                int tailIRows = mc % 6;
                if (tailIRows > 0) {
                    microKernelScalar_<Title>(pA, (long) fullIPanels * 6 * kc * <Bytes>, 0, pB, bBase, C, rowStart + fullIPanels * 6, jc + jr, kc, p, tailIRows, 16, 6, 16);
                }
            } else {
                microKernelScalar_<Title>(pA, (long) fullIPanels * 6 * kc * <Bytes>, 0, pB, bBase, C, rowStart, jc + jr, kc, p, mc, actualNR, 6, 16);
            }
        }
    }

    private static void gebpMacroKernel_Aarch_<Title>(MemorySegment pA, MemorySegment pB, MemorySegment C, int rowStart, int mc, int jc, int nc, int kc, int p) {
        int nrPanels = (nc + 11) / 12; int fullIPanels = mc / 8;
        for (int jp = 0; jp < nrPanels; jp++) {
            int jr = jp * 12; int actualNR = Math.min(12, nc - jr);
            long bBase = (long) jp * 16 * kc * <Bytes>;
            if (actualNR == 12) {
                for (int ip = 0; ip < fullIPanels; ip++) {
                    microKernel8x12_<Title>(pA, (long) ip * 8 * kc * <Bytes>, pB, bBase, C, rowStart + ip * 8, jc + jr, kc, p);
                }
                // Issue 3 Resolved: Capture tail rows for full NR bounds
                int tailIRows = mc % 8;
                if (tailIRows > 0) {
                    microKernelScalar_<Title>(pA, (long) fullIPanels * 8 * kc * <Bytes>, 0, pB, bBase, C, rowStart + fullIPanels * 8, jc + jr, kc, p, tailIRows, 12, 8, 16);
                }
            } else {
                microKernelScalar_<Title>(pA, (long) fullIPanels * 8 * kc * <Bytes>, 0, pB, bBase, C, rowStart, jc + jr, kc, p, mc, actualNR, 8, 16);
            }
        }
    }

    private static void microKernelScalar_<Title>(MemorySegment pA, long aBase, int rOff, MemorySegment pB, long bBase, MemorySegment C, int ci, int cj, int kc, int N, int mr, int nr, int MR_dim, int NR_dim) {
        for (int k = 0; k < kc; k++) {
            long aOff = aBase + (long) k * MR_dim * <Bytes> + (long) rOff * <Bytes>;
            long bOff = bBase + (long) k * NR_dim * <Bytes>;
            for (int r = 0; r < mr; r++) {
                <primitive> aVal = pA.get(<Layout>, aOff + (long) r * <Bytes>);
                for (int c = 0; c < nr; c++) {
                    long cIdx = (long)(ci + r) * N + cj + c;
                    C.setAtIndex(<Layout>, cIdx, (<primitive>)(C.getAtIndex(<Layout>, cIdx) + (aVal * pB.get(<Layout>, bOff + (long) c * <Bytes>))));
                }
            }
        }
    }
"""

AVX2_HYBRID_TEMPLATE = """
    static class AVX2_<Title> extends RecursiveAction {
        MemorySegment A, B_T, C; int n, m, p, startRow, endRow;
        AVX2_<Title>(MemorySegment A, MemorySegment B_T, MemorySegment C, int n, int m, int p, int startRow, int endRow) {
            this.A = A; this.B_T = B_T; this.C = C; this.n = n; this.m = m; this.p = p; this.startRow = startRow; this.endRow = endRow;
        }
        @Override
        protected void compute() {
            if (endRow - startRow <= THRESHOLD) {
                int safeRowEnd = endRow - ((endRow - startRow) % 2); int safeColEnd = p - (p % 2);
                for (int i = startRow; i < safeRowEnd; i += 2) {
                    for (int j = 0; j < safeColEnd; j += 2) {
                        hybridKernel2x2_<Title>(A, B_T, C, m, p, i, j);
                    }
                }
                if (safeRowEnd < endRow) {
                    for (int j = 0; j < safeColEnd; j++) scalarDotProduct_<Title>(A, B_T, C, m, p, safeRowEnd, j);
                }
                if (safeColEnd < p) {
                    for (int i = startRow; i < safeRowEnd; i++) scalarDotProduct_<Title>(A, B_T, C, m, p, i, safeColEnd);
                }
                if (safeRowEnd < endRow && safeColEnd < p) {
                    scalarDotProduct_<Title>(A, B_T, C, m, p, safeRowEnd, safeColEnd);
                }
            } else {
                int mid = startRow + (endRow - startRow) / 2;
                invokeAll(new AVX2_<Title>(A, B_T, C, n, m, p, startRow, mid), new AVX2_<Title>(A, B_T, C, n, m, p, mid, endRow));
            }
        }
    }

    private static void hybridKernel2x2_<Title>(MemorySegment A, MemorySegment B_T, MemorySegment C, int m, int p, int i, int j) {
        var vSum00 = <VectorClass>.zero(<Species>); var vSum01 = <VectorClass>.zero(<Species>);
        var vSum10 = <VectorClass>.zero(<Species>); var vSum11 = <VectorClass>.zero(<Species>);
        long k = 0; long loopBound = <Species>.loopBound(m);
        for (; k < loopBound; k += <Species>.length()) {
            var vA0 = <VectorClass>.fromMemorySegment(<Species>, A, ((long) i * m + k) * <Bytes>, ByteOrder.nativeOrder());
            var vA1 = <VectorClass>.fromMemorySegment(<Species>, A, ((long) (i + 1) * m + k) * <Bytes>, ByteOrder.nativeOrder());
            var vB0 = <VectorClass>.fromMemorySegment(<Species>, B_T, ((long) j * m + k) * <Bytes>, ByteOrder.nativeOrder());
            var vB1 = <VectorClass>.fromMemorySegment(<Species>, B_T, ((long) (j + 1) * m + k) * <Bytes>, ByteOrder.nativeOrder());
            <FMA00>; <FMA01>;
            <FMA10>; <FMA11>;
        }
        <primitive> sum00 = vSum00.reduceLanes(VectorOperators.ADD); <primitive> sum01 = vSum01.reduceLanes(VectorOperators.ADD);
        <primitive> sum10 = vSum10.reduceLanes(VectorOperators.ADD); <primitive> sum11 = vSum11.reduceLanes(VectorOperators.ADD);
        for (; k < m; k++) {
            <primitive> a0 = A.getAtIndex(<Layout>, ((long) i * m + k)); <primitive> a1 = A.getAtIndex(<Layout>, ((long) (i + 1) * m + k));
            <primitive> b0 = B_T.getAtIndex(<Layout>, ((long) j * m + k)); <primitive> b1 = B_T.getAtIndex(<Layout>, ((long) (j + 1) * m + k));
            sum00 += a0 * b0; sum01 += a0 * b1; sum10 += a1 * b0; sum11 += a1 * b1;
        }
        C.setAtIndex(<Layout>, ((long) i * p + j), sum00); C.setAtIndex(<Layout>, ((long) i * p + j + 1), sum01);
        C.setAtIndex(<Layout>, ((long) (i + 1) * p + j), sum10); C.setAtIndex(<Layout>, ((long) (i + 1) * p + j + 1), sum11);
    }

    private static void scalarDotProduct_<Title>(MemorySegment A, MemorySegment B_T, MemorySegment C, int m, int p, int i, int j) {
        <primitive> sum = <Zero>;
        for (int k = 0; k < m; k++) {
            sum += A.getAtIndex(<Layout>, ((long) i * m + k)) * B_T.getAtIndex(<Layout>, ((long) j * m + k));
        }
        C.setAtIndex(<Layout>, ((long) i * p + j), sum);
    }

    // High performance sequential tiled transpose loop to prevent ForkJoin pool thread contention
    private static MemorySegment fastTranspose2D_<Title>(MemorySegment src, Arena arena, int rows, int cols) {
        MemorySegment dst = arena.allocate((long) rows * cols * <Bytes>);
        int TILE = 64;
        for (int rB = 0; rB < rows; rB += TILE) {
            int rMax = Math.min(rB + TILE, rows);
            for (int cB = 0; cB < cols; cB += TILE) {
                int cMax = Math.min(cB + TILE, cols);
                for (int i = rB; i < rMax; i++) {
                    long iStride = (long) i * cols;
                    for (int j = cB; j < cMax; j++) {
                        dst.setAtIndex(<Layout>, (long) j * rows + i, src.getAtIndex(<Layout>, iStride + j));
                    }
                }
            }
        }
        return dst;
    }
"""

def make_6x16_kernel(t):
    lines = ["    private static void microKernel6x16_{Title}(MemorySegment pA, long aBase, MemorySegment pB, long bBase, MemorySegment C, int ci, int cj, int kc, int N) {{".format(Title=t["Title"])]
    for i in range(6):
        for j in range(2):
            lines.append(f"        var c{i}{j} = {t['VectorClass']}.zero({t['Species']});")
    lines.append("        for (int k = 0; k < kc; k++) {")
    lines.append(f"            long aOff = aBase + (long) k * 6 * {t['Bytes']};")
    lines.append(f"            long bOff = bBase + (long) k * 16 * {t['Bytes']};")
    lines.append(f"            var b0 = {t['VectorClass']}.fromMemorySegment({t['Species']}, pB, bOff, NATIVE);")
    lines.append(f"            var b1 = {t['VectorClass']}.fromMemorySegment({t['Species']}, pB, bOff + 8 * {t['Bytes']}, NATIVE);")
    for i in range(6):
        lines.append(f"            var a{i} = {t['VectorClass']}.broadcast({t['Species']}, pA.get({t['Layout']}, aOff + {i} * {t['Bytes']}));")
        lines.append(f"            {t['FMA'].format(c=f'c{i}0', a=f'a{i}', b='b0')};")
        lines.append(f"            {t['FMA'].format(c=f'c{i}1', a=f'a{i}', b='b1')};")
    lines.append("        }")
    for i in range(6):
        lines.append(f"        long row{i} = ((long)(ci + {i}) * N + cj) * {t['Bytes']};")
        lines.append(f"        {t['VectorClass']}.fromMemorySegment({t['Species']}, C, row{i}, NATIVE).add(c{i}0).intoMemorySegment(C, row{i}, NATIVE);")
        lines.append(f"        {t['VectorClass']}.fromMemorySegment({t['Species']}, C, row{i} + 8 * {t['Bytes']}, NATIVE).add(c{i}1).intoMemorySegment(C, row{i} + 8 * {t['Bytes']}, NATIVE);")
    lines.append("    }")
    return "\n".join(lines)

def make_8x12_kernel(t):
    lines = ["    private static void microKernel8x12_{Title}(MemorySegment pA, long aBase, MemorySegment pB, long bBase, MemorySegment C, int ci, int cj, int kc, int N) {{".format(Title=t["Title"])]
    for i in range(8):
        for j in range(2):
            lines.append(f"        var c{i}{j} = {t['VectorClass']}.zero({t['Species']});")
    lines.append("        for (int k = 0; k < kc; k++) {")
    lines.append(f"            long aOff = aBase + (long) k * 8 * {t['Bytes']};")
    lines.append(f"            long bOff = bBase + (long) k * 16 * {t['Bytes']};")
    lines.append(f"            var b0 = {t['VectorClass']}.fromMemorySegment({t['Species']}, pB, bOff, NATIVE);")
    lines.append(f"            var b1 = {t['VectorClass']}.fromMemorySegment({t['Species']}, pB, bOff + 8 * {t['Bytes']}, NATIVE);")
    for i in range(8):
        lines.append(f"            var a{i} = {t['VectorClass']}.broadcast({t['Species']}, pA.get({t['Layout']}, aOff + {i} * {t['Bytes']}));")
        lines.append(f"            {t['FMA'].format(c=f'c{i}0', a=f'a{i}', b='b0')};")
        lines.append(f"            {t['FMA'].format(c=f'c{i}1', a=f'a{i}', b='b1')};")
    lines.append("        }")
    for i in range(8):
        lines.append(f"        long row{i} = ((long)(ci + {i}) * N + cj) * {t['Bytes']};")
        lines.append(f"        {t['VectorClass']}.fromMemorySegment({t['Species']}, C, row{i}, NATIVE).add(c{i}0).intoMemorySegment(C, row{i}, NATIVE);")
        for lane in range(4):
            lines.append(f"        C.setAtIndex({t['Layout']}, (long)(ci + {i}) * N + cj + 8 + {lane}, ({t['primitive']})(C.getAtIndex({t['Layout']}, (long)(ci + {i}) * N + cj + 8 + {lane}) + c{i}1.lane({lane})));")
    lines.append("    }")
    return "\n".join(lines)

def generate_code():
    generated_methods = []
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    
    template_path = os.path.join(project_root, "src", "main", "resources", "templates", "MatMulOps.template")
    output_path = os.path.join(project_root, "src", "main", "java", "jnum", "jnumops", "MatMulOps.java")
    
    for t in TYPE_MAPPINGS:
        block = ROUTER_TEMPLATE + TILING_PIPELINE_TEMPLATE
        block += "\n" + make_6x16_kernel(t)
        block += "\n" + make_8x12_kernel(t)
        block += "\n" + AVX2_HYBRID_TEMPLATE
        
        method_code = block \
            .replace("<Title>", t["Title"]) \
            .replace("<primitive>", t["primitive"]) \
            .replace("<VectorClass>", t["VectorClass"]) \
            .replace("<Species>", t["Species"]) \
            .replace("<Layout>", t["Layout"]) \
            .replace("<Bytes>", t["Bytes"]) \
            .replace("<Zero>", t["Zero"]) \
            .replace("<FMA00>", t["FMA00"]) \
            .replace("<FMA01>", t["FMA01"]) \
            .replace("<FMA10>", t["FMA10"]) \
            .replace("<FMA11>", t["FMA11"])
            
        generated_methods.append(method_code)

    try:
        with open(template_path, "r") as file:
            template_content = file.read()
    except FileNotFoundError:
        print(f"ERROR: Could not find template path at {template_path}")
        return

    final_java_code = template_content.replace("// --- GENERATED METHODS ---", "\n".join(generated_methods))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as file:
        file.write(final_java_code)
    
    print("Successfully generated MatMulOps.java at target destination!")

if __name__ == "__main__":
    generate_code()