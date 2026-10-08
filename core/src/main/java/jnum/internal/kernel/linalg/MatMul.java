package jnum.internal.kernel.linalg;

import static jnum.internal.Constants.*;

import java.lang.foreign.Arena;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SequenceLayout;
import java.nio.ByteOrder;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorSpecies;
import jdk.incubator.vector.VectorOperators;
import jnum.NDArray;

public final class MatMul {

    private static final VectorSpecies<Float> SPECIES = SPECIES_F32;
    private static final VectorSpecies<Integer> SPECIESINT = SPECIES_I32;
    private static final VectorSpecies<Double> SPECIESDB = SPECIES_F64;
    
    private static final ForkJoinPool POOL = new ForkJoinPool(AVAILABLE_CORES);
    private static final int THRESHOLD = 64;
    private static final ByteOrder NATIVE = NATIVE_ORDER;

    // BLIS cache-blocking constants
    private static final int MR = 6;
    private static final int NR = NR_AVX2_F32;       // Compile-time constant 16 (powers >> 4, & 15)
    private static final int NR_DB = NR_AVX2_F64;    // Compile-time constant 8  (powers >> 3, & 7)
    private static final int NR_INT = NR_AVX2_I32;   // Compile-time constant 16 (powers >> 4, & 15)
    private static final int MC = MATMUL_MC;
    private static final int KC = MATMUL_KC;
    private static final int NC_ARM = MATMUL_NC;
    private static final int NC_AARCH = MATMUL_NC;

    // Float AVX2 byte strides and unroll offsets (compile-time constant immediate displacements)
    private static final long STRIDE_VEC_BYTES_F32 = 32L; // 8 floats * 4 bytes
    private static final long STEP_A_ROW_BYTES_F32 = (long) MR * 4L; // 24L
    private static final long STEP_B_ROW_BYTES_F32 = (long) NR * 4L; // 64L
    private static final long STEP_A_K4_BYTES_F32  = 4 * STEP_A_ROW_BYTES_F32; // 96L
    private static final long STEP_B_K4_BYTES_F32  = 4 * STEP_B_ROW_BYTES_F32; // 256L

    // Double AVX2 byte strides and unroll offsets
    private static final long STRIDE_VEC_BYTES_F64 = 32L; // 4 doubles * 8 bytes
    private static final long STEP_A_ROW_BYTES_F64 = (long) MR * 8L; // 48L
    private static final long STEP_B_ROW_BYTES_F64 = (long) NR_DB * 8L; // 64L
    private static final long STEP_A_K4_BYTES_F64  = 4 * STEP_A_ROW_BYTES_F64; // 192L
    private static final long STEP_B_K4_BYTES_F64  = 4 * STEP_B_ROW_BYTES_F64; // 256L

    // Int AVX2 byte strides and unroll offsets
    private static final long STRIDE_VEC_BYTES_I32 = 32L; // 8 ints * 4 bytes
    private static final long STEP_A_ROW_BYTES_I32 = (long) MR * 4L; // 24L
    private static final long STEP_B_ROW_BYTES_I32 = (long) NR_INT * 4L; // 64L
    private static final long STEP_A_K4_BYTES_I32  = 4 * STEP_A_ROW_BYTES_I32; // 96L
    private static final long STEP_B_K4_BYTES_I32  = 4 * STEP_B_ROW_BYTES_I32; // 256L

    private MatMul() {
        throw new AssertionError();
    }
    private static MemorySegment allocateAlignedFloat(Arena arena, long count) {
        SequenceLayout layout = MemoryLayout.sequenceLayout(count, VF)
                                            .withByteAlignment(64);
        return arena.allocate(layout);
    }

    private static MemorySegment allocateAlignedDouble(Arena arena, long count) {
        SequenceLayout layout = MemoryLayout.sequenceLayout(count, VD)
                                            .withByteAlignment(64);
        return arena.allocate(layout);
    }

    private static MemorySegment allocateAlignedInt(Arena arena, long count) {
        SequenceLayout layout = MemoryLayout.sequenceLayout(count, VI)
                                            .withByteAlignment(64);
        return arena.allocate(layout);
    }

    // ThreadLocal panel buffers to eliminate hot-path allocations
    private static final ThreadLocal<MemorySegment> tlPackedA_Arm_Float =
        ThreadLocal.withInitial(() -> allocateAlignedFloat(Arena.ofAuto(), (long) MC * KC));
    private static final ThreadLocal<MemorySegment> tlPackedA_Aarch_Float =
        ThreadLocal.withInitial(() -> allocateAlignedFloat(Arena.ofAuto(), (long) MC * KC));
    private static final ThreadLocal<MemorySegment> tlPackedB_Arm_Float =
        ThreadLocal.withInitial(() -> allocateAlignedFloat(Arena.ofAuto(), (long) KC * NC_ARM));
    private static final ThreadLocal<MemorySegment> tlPackedB_Aarch_Float =
        ThreadLocal.withInitial(() -> allocateAlignedFloat(Arena.ofAuto(), (long) KC * NC_AARCH));

    private static final ThreadLocal<MemorySegment> tlPackedA_Arm_Double =
        ThreadLocal.withInitial(() -> allocateAlignedDouble(Arena.ofAuto(), (long) MC * KC));
    private static final ThreadLocal<MemorySegment> tlPackedA_Aarch_Double =
        ThreadLocal.withInitial(() -> allocateAlignedDouble(Arena.ofAuto(), (long) MC * KC));
    private static final ThreadLocal<MemorySegment> tlPackedB_Arm_Double =
        ThreadLocal.withInitial(() -> allocateAlignedDouble(Arena.ofAuto(), (long) KC * NC_ARM));
    private static final ThreadLocal<MemorySegment> tlPackedB_Aarch_Double =
        ThreadLocal.withInitial(() -> allocateAlignedDouble(Arena.ofAuto(), (long) KC * NC_AARCH));

    private static final ThreadLocal<MemorySegment> tlPackedA_Arm_Int =
        ThreadLocal.withInitial(() -> allocateAlignedInt(Arena.ofAuto(), (long) MC * KC));
    private static final ThreadLocal<MemorySegment> tlPackedA_Aarch_Int =
        ThreadLocal.withInitial(() -> allocateAlignedInt(Arena.ofAuto(), (long) MC * KC));
    private static final ThreadLocal<MemorySegment> tlPackedB_Arm_Int =
        ThreadLocal.withInitial(() -> allocateAlignedInt(Arena.ofAuto(), (long) KC * NC_ARM));
    private static final ThreadLocal<MemorySegment> tlPackedB_Aarch_Int =
        ThreadLocal.withInitial(() -> allocateAlignedInt(Arena.ofAuto(), (long) KC * NC_AARCH));

    // =========================================================================
    // FLOAT MATMUL
    // =========================================================================
    public static NDArray matmulFloat(NDArray a, NDArray b, NDArray resArray) {
        int n = (int) a.internalShapeUnsafe()[0]; 
        int m = (int) a.internalShapeUnsafe()[1]; 
        int p = (int) b.internalShapeUnsafe()[1];
        
        int maxDim = Math.max(n, Math.max(m, p));
        if (maxDim <= 4) {
            nanoKernel_Float(a, b, resArray, n, m, p);
        } else if (maxDim <= 128) {
            directTiled_Float(a, b, resArray, n, m, p);
        } else {
            long a_s0 = a.internalStridesUnsafe()[0], a_s1 = a.internalStridesUnsafe()[1];
            long b_s0 = b.internalStridesUnsafe()[0], b_s1 = b.internalStridesUnsafe()[1];
            if (maxDim <= 256) {
                blisSingleThread_Float(a.getData(), a_s0, a_s1, b.getData(), b_s0, b_s1, resArray.getData(), n, m, p);
            } else {
                if (IS_AARCH64) {
                    blisAarchMacro_Float(a.getData(), a_s0, a_s1, b.getData(), b_s0, b_s1, resArray.getData(), n, m, p);
                } else {
                    blisArmMacro_Float(a.getData(), a_s0, a_s1, b.getData(), b_s0, b_s1, resArray.getData(), n, m, p);
                }
            }
        }
        return resArray;
    }
// ---- Tier 0: Nano kernel (maxDim <= 4) - fully unrolled scalar, ZERO allocation ----
    private static void nanoKernel_Float(NDArray a, NDArray b, NDArray resArray, int n, int m, int p) {
        long[] aStrides = a.internalStridesUnsafe();
        long[] bStrides = b.internalStridesUnsafe();
        long[] cStrides = resArray.internalStridesUnsafe();
        MemorySegment memA = a.getData();
        MemorySegment memB = b.getData();
        MemorySegment memC = resArray.getData();

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < p; j++) {
                float sum = 0f;
                for (int k = 0; k < m; k++) {
                    sum += memA.get(VF, ((long) i * aStrides[0] + (long) k * aStrides[1]) * 4L)
                         * memB.get(VF, ((long) k * bStrides[0] + (long) j * bStrides[1]) * 4L);
                }
                memC.set(VF, ((long) i * cStrides[0] + (long) j * cStrides[1]) * 4L, sum);
            }
        }
    }

    // ---- Tier 1: Direct Register-Blocked SIMD Kernel (4 < maxDim <= 128) - zero packing, zero allocation ----
    private static void directTiled_Float(NDArray a, NDArray b, NDArray resArray, int n, int m, int p) {
        long[] aStrides = a.internalStridesUnsafe();
        long[] bStrides = b.internalStridesUnsafe();
        long[] cStrides = resArray.internalStridesUnsafe();
        long a_s0 = aStrides[0]; long a_s1 = aStrides[1];
        long b_s0 = bStrides[0]; long b_s1 = bStrides[1];
        long c_s0 = cStrides[0]; long c_s1 = cStrides[1];
        MemorySegment memA = a.getData();
        MemorySegment memB = b.getData();
        MemorySegment memC = resArray.getData();

        if (b_s1 == 1L && c_s1 == 1L) {
            int safeRowEnd = n - (n % 4);
            int safeColEnd = p - (p % 16);

            for (int i = 0; i < safeRowEnd; i += 4) {
                long aRow0 = (long)(i + 0) * a_s0;
                long aRow1 = (long)(i + 1) * a_s0;
                long aRow2 = (long)(i + 2) * a_s0;
                long aRow3 = (long)(i + 3) * a_s0;

                for (int j = 0; j < safeColEnd; j += 16) {
                    var acc00 = FloatVector.zero(SPECIES); var acc01 = FloatVector.zero(SPECIES);
                    var acc10 = FloatVector.zero(SPECIES); var acc11 = FloatVector.zero(SPECIES);
                    var acc20 = FloatVector.zero(SPECIES); var acc21 = FloatVector.zero(SPECIES);
                    var acc30 = FloatVector.zero(SPECIES); var acc31 = FloatVector.zero(SPECIES);

                    int k = 0;
                    for (; k <= m - 4; k += 4) {
                        // k + 0
                        long bOff0 = ((long)(k + 0) * b_s0 + j) * 4L;
                        var b0_0 = FloatVector.fromMemorySegment(SPECIES, memB, bOff0, NATIVE);
                        var b1_0 = FloatVector.fromMemorySegment(SPECIES, memB, bOff0 + 32L, NATIVE);

                        var a0_0 = FloatVector.broadcast(SPECIES, memA.get(VF, (aRow0 + (long)(k + 0) * a_s1) * 4L));
                        acc00 = a0_0.fma(b0_0, acc00); acc01 = a0_0.fma(b1_0, acc01);
                        var a1_0 = FloatVector.broadcast(SPECIES, memA.get(VF, (aRow1 + (long)(k + 0) * a_s1) * 4L));
                        acc10 = a1_0.fma(b0_0, acc10); acc11 = a1_0.fma(b1_0, acc11);
                        var a2_0 = FloatVector.broadcast(SPECIES, memA.get(VF, (aRow2 + (long)(k + 0) * a_s1) * 4L));
                        acc20 = a2_0.fma(b0_0, acc20); acc21 = a2_0.fma(b1_0, acc21);
                        var a3_0 = FloatVector.broadcast(SPECIES, memA.get(VF, (aRow3 + (long)(k + 0) * a_s1) * 4L));
                        acc30 = a3_0.fma(b0_0, acc30); acc31 = a3_0.fma(b1_0, acc31);

                        // k + 1
                        long bOff1 = ((long)(k + 1) * b_s0 + j) * 4L;
                        var b0_1 = FloatVector.fromMemorySegment(SPECIES, memB, bOff1, NATIVE);
                        var b1_1 = FloatVector.fromMemorySegment(SPECIES, memB, bOff1 + 32L, NATIVE);

                        var a0_1 = FloatVector.broadcast(SPECIES, memA.get(VF, (aRow0 + (long)(k + 1) * a_s1) * 4L));
                        acc00 = a0_1.fma(b0_1, acc00); acc01 = a0_1.fma(b1_1, acc01);
                        var a1_1 = FloatVector.broadcast(SPECIES, memA.get(VF, (aRow1 + (long)(k + 1) * a_s1) * 4L));
                        acc10 = a1_1.fma(b0_1, acc10); acc11 = a1_1.fma(b1_1, acc11);
                        var a2_1 = FloatVector.broadcast(SPECIES, memA.get(VF, (aRow2 + (long)(k + 1) * a_s1) * 4L));
                        acc20 = a2_1.fma(b0_1, acc20); acc21 = a2_1.fma(b1_1, acc21);
                        var a3_1 = FloatVector.broadcast(SPECIES, memA.get(VF, (aRow3 + (long)(k + 1) * a_s1) * 4L));
                        acc30 = a3_1.fma(b0_1, acc30); acc31 = a3_1.fma(b1_1, acc31);

                        // k + 2
                        long bOff2 = ((long)(k + 2) * b_s0 + j) * 4L;
                        var b0_2 = FloatVector.fromMemorySegment(SPECIES, memB, bOff2, NATIVE);
                        var b1_2 = FloatVector.fromMemorySegment(SPECIES, memB, bOff2 + 32L, NATIVE);

                        var a0_2 = FloatVector.broadcast(SPECIES, memA.get(VF, (aRow0 + (long)(k + 2) * a_s1) * 4L));
                        acc00 = a0_2.fma(b0_2, acc00); acc01 = a0_2.fma(b1_2, acc01);
                        var a1_2 = FloatVector.broadcast(SPECIES, memA.get(VF, (aRow1 + (long)(k + 2) * a_s1) * 4L));
                        acc10 = a1_2.fma(b0_2, acc10); acc11 = a1_2.fma(b1_2, acc11);
                        var a2_2 = FloatVector.broadcast(SPECIES, memA.get(VF, (aRow2 + (long)(k + 2) * a_s1) * 4L));
                        acc20 = a2_2.fma(b0_2, acc20); acc21 = a2_2.fma(b1_2, acc21);
                        var a3_2 = FloatVector.broadcast(SPECIES, memA.get(VF, (aRow3 + (long)(k + 2) * a_s1) * 4L));
                        acc30 = a3_2.fma(b0_2, acc30); acc31 = a3_2.fma(b1_2, acc31);

                        // k + 3
                        long bOff3 = ((long)(k + 3) * b_s0 + j) * 4L;
                        var b0_3 = FloatVector.fromMemorySegment(SPECIES, memB, bOff3, NATIVE);
                        var b1_3 = FloatVector.fromMemorySegment(SPECIES, memB, bOff3 + 32L, NATIVE);

                        var a0_3 = FloatVector.broadcast(SPECIES, memA.get(VF, (aRow0 + (long)(k + 3) * a_s1) * 4L));
                        acc00 = a0_3.fma(b0_3, acc00); acc01 = a0_3.fma(b1_3, acc01);
                        var a1_3 = FloatVector.broadcast(SPECIES, memA.get(VF, (aRow1 + (long)(k + 3) * a_s1) * 4L));
                        acc10 = a1_3.fma(b0_3, acc10); acc11 = a1_3.fma(b1_3, acc11);
                        var a2_3 = FloatVector.broadcast(SPECIES, memA.get(VF, (aRow2 + (long)(k + 3) * a_s1) * 4L));
                        acc20 = a2_3.fma(b0_3, acc20); acc21 = a2_3.fma(b1_3, acc21);
                        var a3_3 = FloatVector.broadcast(SPECIES, memA.get(VF, (aRow3 + (long)(k + 3) * a_s1) * 4L));
                        acc30 = a3_3.fma(b0_3, acc30); acc31 = a3_3.fma(b1_3, acc31);
                    }

                    for (; k < m; k++) {
                        long bOff = ((long) k * b_s0 + j) * 4L;
                        var b0 = FloatVector.fromMemorySegment(SPECIES, memB, bOff, NATIVE);
                        var b1 = FloatVector.fromMemorySegment(SPECIES, memB, bOff + 32L, NATIVE);

                        var a0 = FloatVector.broadcast(SPECIES, memA.get(VF, (aRow0 + (long) k * a_s1) * 4L));
                        acc00 = a0.fma(b0, acc00); acc01 = a0.fma(b1, acc01);
                        var a1 = FloatVector.broadcast(SPECIES, memA.get(VF, (aRow1 + (long) k * a_s1) * 4L));
                        acc10 = a1.fma(b0, acc10); acc11 = a1.fma(b1, acc11);
                        var a2 = FloatVector.broadcast(SPECIES, memA.get(VF, (aRow2 + (long) k * a_s1) * 4L));
                        acc20 = a2.fma(b0, acc20); acc21 = a2.fma(b1, acc21);
                        var a3 = FloatVector.broadcast(SPECIES, memA.get(VF, (aRow3 + (long) k * a_s1) * 4L));
                        acc30 = a3.fma(b0, acc30); acc31 = a3.fma(b1, acc31);
                    }

                    long cRow0 = ((long)(i + 0) * c_s0 + j) * 4L;
                    acc00.intoMemorySegment(memC, cRow0, NATIVE);
                    acc01.intoMemorySegment(memC, cRow0 + 32L, NATIVE);

                    long cRow1 = ((long)(i + 1) * c_s0 + j) * 4L;
                    acc10.intoMemorySegment(memC, cRow1, NATIVE);
                    acc11.intoMemorySegment(memC, cRow1 + 32L, NATIVE);

                    long cRow2 = ((long)(i + 2) * c_s0 + j) * 4L;
                    acc20.intoMemorySegment(memC, cRow2, NATIVE);
                    acc21.intoMemorySegment(memC, cRow2 + 32L, NATIVE);

                    long cRow3 = ((long)(i + 3) * c_s0 + j) * 4L;
                    acc30.intoMemorySegment(memC, cRow3, NATIVE);
                    acc31.intoMemorySegment(memC, cRow3 + 32L, NATIVE);
                }
            }

            // Cleanup tail rows and columns
            if (safeRowEnd < n) {
                for (int ii = safeRowEnd; ii < n; ii++) {
                    for (int jj = 0; jj < p; jj++) {
                        float sum = 0f;
                        for (int kk = 0; kk < m; kk++) {
                            sum += memA.get(VF, ((long) ii * a_s0 + (long) kk * a_s1) * 4L)
                                 * memB.get(VF, ((long) kk * b_s0 + (long) jj * b_s1) * 4L);
                        }
                        memC.set(VF, ((long) ii * c_s0 + (long) jj * c_s1) * 4L, sum);
                    }
                }
            }
            if (safeColEnd < p) {
                for (int ii = 0; ii < safeRowEnd; ii++) {
                    for (int jj = safeColEnd; jj < p; jj++) {
                        float sum = 0f;
                        for (int kk = 0; kk < m; kk++) {
                            sum += memA.get(VF, ((long) ii * a_s0 + (long) kk * a_s1) * 4L)
                                 * memB.get(VF, ((long) kk * b_s0 + (long) jj * b_s1) * 4L);
                        }
                        memC.set(VF, ((long) ii * c_s0 + (long) jj * c_s1) * 4L, sum);
                    }
                }
            }
        } else {
            // General strided path
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < p; j++) {
                    float sum = 0f;
                    for (int k = 0; k < m; k++) {
                        sum += memA.get(VF, ((long) i * a_s0 + (long) k * a_s1) * 4L)
                             * memB.get(VF, ((long) k * b_s0 + (long) j * b_s1) * 4L);
                    }
                    memC.set(VF, ((long) i * c_s0 + (long) j * c_s1) * 4L, sum);
                }
            }
        }
    }

        // ---- Tier 2: Single-thread BLIS (maxDim <= 256) - zero allocation, no ForkJoin ----
    private static void blisSingleThread_Float(MemorySegment A, long a_s0, long a_s1,
                                                MemorySegment B, long b_s0, long b_s1,
                                                MemorySegment C, int n, int m, int p) {
        if (IS_AARCH64) {
            MemorySegment pB = tlPackedB_Aarch_Float.get();
            for (int jc = 0; jc < p; jc += NC_AARCH) {
                int nc = Math.min(NC_AARCH, p - jc);
                for (int pc = 0; pc < m; pc += KC) {
                    int kc = Math.min(KC, m - pc);
                    boolean isFirstKBlock = (pc == 0);
                    packB_panel_Aarch_Float(B, pB, pc, jc, kc, nc, b_s0, b_s1);
                    for (int ic = 0; ic < n; ic += MC) {
                        int mc = Math.min(MC, n - ic);
                        MemorySegment pA = tlPackedA_Aarch_Float.get();
                        packA_panel_Aarch_Float(A, pA, ic, mc, pc, kc, a_s0, a_s1);
                        gebpMacroKernel_Aarch_Float(pA, pB, C, ic, mc, jc, nc, kc, p, isFirstKBlock);
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
                    packB_panel_Arm_Float(B, pB, pc, jc, kc, nc, b_s0, b_s1);
                    for (int ic = 0; ic < n; ic += MC) {
                        int mc = Math.min(MC, n - ic);
                        MemorySegment pA = tlPackedA_Arm_Float.get();
                        packA_panel_Arm_Float(A, pA, ic, mc, pc, kc, a_s0, a_s1);
                        gebpMacroKernel_Arm_Float(pA, pB, C, ic, mc, jc, nc, kc, p, isFirstKBlock);
                    }
                }
            }
        }
    }

    // ---- Tier 3: Parallel BLIS Macro-Kernels (maxDim > 256) ----
    private static void blisArmMacro_Float(MemorySegment A, long a_s0, long a_s1,
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

    private static void blisAarchMacro_Float(MemorySegment A, long a_s0, long a_s1,
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

    private static void packA_panel_Arm_Float(MemorySegment src, MemorySegment dst, int rowStart, int mc, int colStart, int kc, long a_s0, long a_s1) {
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

    private static void packB_panel_Arm_Float(MemorySegment src, MemorySegment dst, int rowStart, int colStart, int kc, int nc, long b_s0, long b_s1) {
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

    private static void packB_panels_range_Arm_Float(MemorySegment src, MemorySegment dst, int rowStart, int colStart, int kc, long b_s0, long b_s1, int pStart, int pEnd) {
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

    private static void packB_tail_Arm_Float(MemorySegment src, MemorySegment dst, int rowStart, int colStart, int kc, int nc, long b_s0, long b_s1, int fullPanels, int tailCols) {
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

    static final class PackBTask_Arm_Float extends RecursiveAction {
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

    private static void packA_panel_Aarch_Float(MemorySegment src, MemorySegment dst, int rowStart, int mc, int colStart, int kc, long a_s0, long a_s1) {
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

    private static void packB_panel_Aarch_Float(MemorySegment src, MemorySegment dst, int rowStart, int colStart, int kc, int nc, long b_s0, long b_s1) {
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

    private static void packB_panels_range_Aarch_Float(MemorySegment src, MemorySegment dst, int rowStart, int colStart, int kc, long b_s0, long b_s1, int pStart, int pEnd) {
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

    private static void packB_tail_Aarch_Float(MemorySegment src, MemorySegment dst, int rowStart, int colStart, int kc, int nc, long b_s0, long b_s1, int fullPanels, int tailCols) {
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

    static final class PackBTask_Aarch_Float extends RecursiveAction {
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

    static final class GEBPTask_Arm_Float extends RecursiveAction {
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

    static final class GEBPTask_Aarch_Float extends RecursiveAction {
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

    static final class JcTask_Arm_Float extends RecursiveAction {
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

    static final class JcTask_Aarch_Float extends RecursiveAction {
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
private static void gebpMacroKernel_Arm_Float(MemorySegment pA, MemorySegment pB, MemorySegment C,
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

    private static void gebpMacroKernel_Aarch_Float(MemorySegment pA, MemorySegment pB, MemorySegment C,
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
    private static void microKernel6x16_Float(MemorySegment pA, long aBase, MemorySegment pB, long bBase,
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
    private static void microKernel2x16_Float(MemorySegment pA, long aBase, int rOff,
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
    private static void microKernel1x16_Float(MemorySegment pA, long aBase, int rOff,
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
    private static void microKernel8x12_Float(MemorySegment pA, long aBase, MemorySegment pB, long bBase,
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

    private static void microKernelScalar_Float(MemorySegment pA, long aBase, int rOff,
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
            vSum00 = vSum00.add(vA0.mul(vB0)); vSum01 = vSum01.add(vA0.mul(vB1));
            vSum10 = vSum10.add(vA1.mul(vB0)); vSum11 = vSum11.add(vA1.mul(vB1));
        }
        float sum00 = vSum00.reduceLanes(VectorOperators.ADD); float sum01 = vSum01.reduceLanes(VectorOperators.ADD);
        float sum10 = vSum10.reduceLanes(VectorOperators.ADD); float sum11 = vSum11.reduceLanes(VectorOperators.ADD);
        for (; k < m; k++) {
            float a0 = A.getAtIndex(VF, ((long) i * m + k)); float a1 = A.getAtIndex(VF, ((long) (i + 1) * m + k));
            float b0 = B_T.getAtIndex(VF, ((long) j * m + k)); float b1 = B_T.getAtIndex(VF, ((long) (j + 1) * m + k));
            sum00 += a0 * b0; sum01 += a0 * b1; sum10 += a1 * b0; sum11 += a1 * b1;
        }
        C.setAtIndex(VF, ((long) i * p + j), sum00); C.setAtIndex(VF, ((long) i * p + j + 1), sum01);
        C.setAtIndex(VF, ((long) (i + 1) * p + j), sum10); C.setAtIndex(VF, ((long) (i + 1) * p + j + 1), sum11);
    }

    private static void scalarDotProduct_Float(MemorySegment A, MemorySegment B_T, MemorySegment C, int m, int p, int i, int j) {
        float sum = 0f;
        for (int k = 0; k < m; k++) {
            sum += A.getAtIndex(VF, ((long) i * m + k)) * B_T.getAtIndex(VF, ((long) j * m + k));
        }
        C.setAtIndex(VF, ((long) i * p + j), sum);
    }

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
                        dst.setAtIndex(VF, (long) j * rows + i, src.getAtIndex(VF, iStride + j));
                    }
                }
            }
        }
        return dst;
    }
    
        // =========================================================================
    // DOUBLE MATMUL
    // =========================================================================
    public static NDArray matmulDouble(NDArray a, NDArray b, NDArray resArray) {
        int n = (int) a.internalShapeUnsafe()[0]; 
        int m = (int) a.internalShapeUnsafe()[1]; 
        int p = (int) b.internalShapeUnsafe()[1];
        
        int maxDim = Math.max(n, Math.max(m, p));
        if (maxDim <= 4) {
            nanoKernel_Double(a, b, resArray, n, m, p);
        } else if (maxDim <= 128) {
            directTiled_Double(a, b, resArray, n, m, p);
        } else {
            long a_s0 = a.internalStridesUnsafe()[0], a_s1 = a.internalStridesUnsafe()[1];
            long b_s0 = b.internalStridesUnsafe()[0], b_s1 = b.internalStridesUnsafe()[1];
            if (maxDim <= 256) {
                blisSingleThread_Double(a.getData(), a_s0, a_s1, b.getData(), b_s0, b_s1, resArray.getData(), n, m, p);
            } else {
                if (IS_AARCH64) {
                    blisAarchMacro_Double(a.getData(), a_s0, a_s1, b.getData(), b_s0, b_s1, resArray.getData(), n, m, p);
                } else {
                    blisArmMacro_Double(a.getData(), a_s0, a_s1, b.getData(), b_s0, b_s1, resArray.getData(), n, m, p);
                }
            }
        }
        return resArray;
    }
// ---- Tier 0: Nano kernel (maxDim <= 4) - fully unrolled scalar, ZERO allocation ----
    private static void nanoKernel_Double(NDArray a, NDArray b, NDArray resArray, int n, int m, int p) {
        long[] aStrides = a.internalStridesUnsafe();
        long[] bStrides = b.internalStridesUnsafe();
        long[] cStrides = resArray.internalStridesUnsafe();
        MemorySegment memA = a.getData();
        MemorySegment memB = b.getData();
        MemorySegment memC = resArray.getData();

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < p; j++) {
                double sum = 0.0;
                for (int k = 0; k < m; k++) {
                    sum += memA.get(VD, ((long) i * aStrides[0] + (long) k * aStrides[1]) * 8L)
                         * memB.get(VD, ((long) k * bStrides[0] + (long) j * bStrides[1]) * 8L);
                }
                memC.set(VD, ((long) i * cStrides[0] + (long) j * cStrides[1]) * 8L, sum);
            }
        }
    }

    // ---- Tier 1: Direct Register-Blocked SIMD Kernel (4 < maxDim <= 128) - zero packing, zero allocation ----
    private static void directTiled_Double(NDArray a, NDArray b, NDArray resArray, int n, int m, int p) {
        long[] aStrides = a.internalStridesUnsafe();
        long[] bStrides = b.internalStridesUnsafe();
        long[] cStrides = resArray.internalStridesUnsafe();
        long a_s0 = aStrides[0]; long a_s1 = aStrides[1];
        long b_s0 = bStrides[0]; long b_s1 = bStrides[1];
        long c_s0 = cStrides[0]; long c_s1 = cStrides[1];
        MemorySegment memA = a.getData();
        MemorySegment memB = b.getData();
        MemorySegment memC = resArray.getData();

        long strideBytes = (long) SPECIESDB.length() * 8L;

        if (b_s1 == 1L && c_s1 == 1L) {
            int safeRowEnd = n - (n % 4);
            int safeColEnd = p - (p % NR_DB);

            for (int i = 0; i < safeRowEnd; i += 4) {
                long aRow0 = (long)(i + 0) * a_s0;
                long aRow1 = (long)(i + 1) * a_s0;
                long aRow2 = (long)(i + 2) * a_s0;
                long aRow3 = (long)(i + 3) * a_s0;

                for (int j = 0; j < safeColEnd; j += NR_DB) {
                    var acc00 = DoubleVector.zero(SPECIESDB); var acc01 = DoubleVector.zero(SPECIESDB);
                    var acc10 = DoubleVector.zero(SPECIESDB); var acc11 = DoubleVector.zero(SPECIESDB);
                    var acc20 = DoubleVector.zero(SPECIESDB); var acc21 = DoubleVector.zero(SPECIESDB);
                    var acc30 = DoubleVector.zero(SPECIESDB); var acc31 = DoubleVector.zero(SPECIESDB);

                    int k = 0;
                    for (; k <= m - 4; k += 4) {
                        // k + 0
                        long bOff0 = ((long)(k + 0) * b_s0 + j) * 8L;
                        var b0_0 = DoubleVector.fromMemorySegment(SPECIESDB, memB, bOff0, NATIVE);
                        var b1_0 = DoubleVector.fromMemorySegment(SPECIESDB, memB, bOff0 + strideBytes, NATIVE);

                        var a0_0 = DoubleVector.broadcast(SPECIESDB, memA.get(VD, (aRow0 + (long)(k + 0) * a_s1) * 8L));
                        acc00 = a0_0.fma(b0_0, acc00); acc01 = a0_0.fma(b1_0, acc01);
                        var a1_0 = DoubleVector.broadcast(SPECIESDB, memA.get(VD, (aRow1 + (long)(k + 0) * a_s1) * 8L));
                        acc10 = a1_0.fma(b0_0, acc10); acc11 = a1_0.fma(b1_0, acc11);
                        var a2_0 = DoubleVector.broadcast(SPECIESDB, memA.get(VD, (aRow2 + (long)(k + 0) * a_s1) * 8L));
                        acc20 = a2_0.fma(b0_0, acc20); acc21 = a2_0.fma(b1_0, acc21);
                        var a3_0 = DoubleVector.broadcast(SPECIESDB, memA.get(VD, (aRow3 + (long)(k + 0) * a_s1) * 8L));
                        acc30 = a3_0.fma(b0_0, acc30); acc31 = a3_0.fma(b1_0, acc31);

                        // k + 1
                        long bOff1 = ((long)(k + 1) * b_s0 + j) * 8L;
                        var b0_1 = DoubleVector.fromMemorySegment(SPECIESDB, memB, bOff1, NATIVE);
                        var b1_1 = DoubleVector.fromMemorySegment(SPECIESDB, memB, bOff1 + strideBytes, NATIVE);

                        var a0_1 = DoubleVector.broadcast(SPECIESDB, memA.get(VD, (aRow0 + (long)(k + 1) * a_s1) * 8L));
                        acc00 = a0_1.fma(b0_1, acc00); acc01 = a0_1.fma(b1_1, acc01);
                        var a1_1 = DoubleVector.broadcast(SPECIESDB, memA.get(VD, (aRow1 + (long)(k + 1) * a_s1) * 8L));
                        acc10 = a1_1.fma(b0_1, acc10); acc11 = a1_1.fma(b1_1, acc11);
                        var a2_1 = DoubleVector.broadcast(SPECIESDB, memA.get(VD, (aRow2 + (long)(k + 1) * a_s1) * 8L));
                        acc20 = a2_1.fma(b0_1, acc20); acc21 = a2_1.fma(b1_1, acc21);
                        var a3_1 = DoubleVector.broadcast(SPECIESDB, memA.get(VD, (aRow3 + (long)(k + 1) * a_s1) * 8L));
                        acc30 = a3_1.fma(b0_1, acc30); acc31 = a3_1.fma(b1_1, acc31);

                        // k + 2
                        long bOff2 = ((long)(k + 2) * b_s0 + j) * 8L;
                        var b0_2 = DoubleVector.fromMemorySegment(SPECIESDB, memB, bOff2, NATIVE);
                        var b1_2 = DoubleVector.fromMemorySegment(SPECIESDB, memB, bOff2 + strideBytes, NATIVE);

                        var a0_2 = DoubleVector.broadcast(SPECIESDB, memA.get(VD, (aRow0 + (long)(k + 2) * a_s1) * 8L));
                        acc00 = a0_2.fma(b0_2, acc00); acc01 = a0_2.fma(b1_2, acc01);
                        var a1_2 = DoubleVector.broadcast(SPECIESDB, memA.get(VD, (aRow1 + (long)(k + 2) * a_s1) * 8L));
                        acc10 = a1_2.fma(b0_2, acc10); acc11 = a1_2.fma(b1_2, acc11);
                        var a2_2 = DoubleVector.broadcast(SPECIESDB, memA.get(VD, (aRow2 + (long)(k + 2) * a_s1) * 8L));
                        acc20 = a2_2.fma(b0_2, acc20); acc21 = a2_2.fma(b1_2, acc21);
                        var a3_2 = DoubleVector.broadcast(SPECIESDB, memA.get(VD, (aRow3 + (long)(k + 2) * a_s1) * 8L));
                        acc30 = a3_2.fma(b0_2, acc30); acc31 = a3_2.fma(b1_2, acc31);

                        // k + 3
                        long bOff3 = ((long)(k + 3) * b_s0 + j) * 8L;
                        var b0_3 = DoubleVector.fromMemorySegment(SPECIESDB, memB, bOff3, NATIVE);
                        var b1_3 = DoubleVector.fromMemorySegment(SPECIESDB, memB, bOff3 + strideBytes, NATIVE);

                        var a0_3 = DoubleVector.broadcast(SPECIESDB, memA.get(VD, (aRow0 + (long)(k + 3) * a_s1) * 8L));
                        acc00 = a0_3.fma(b0_3, acc00); acc01 = a0_3.fma(b1_3, acc01);
                        var a1_3 = DoubleVector.broadcast(SPECIESDB, memA.get(VD, (aRow1 + (long)(k + 3) * a_s1) * 8L));
                        acc10 = a1_3.fma(b0_3, acc10); acc11 = a1_3.fma(b1_3, acc11);
                        var a2_3 = DoubleVector.broadcast(SPECIESDB, memA.get(VD, (aRow2 + (long)(k + 3) * a_s1) * 8L));
                        acc20 = a2_3.fma(b0_3, acc20); acc21 = a2_3.fma(b1_3, acc21);
                        var a3_3 = DoubleVector.broadcast(SPECIESDB, memA.get(VD, (aRow3 + (long)(k + 3) * a_s1) * 8L));
                        acc30 = a3_3.fma(b0_3, acc30); acc31 = a3_3.fma(b1_3, acc31);
                    }

                    for (; k < m; k++) {
                        long bOff = ((long) k * b_s0 + j) * 8L;
                        var b0 = DoubleVector.fromMemorySegment(SPECIESDB, memB, bOff, NATIVE);
                        var b1 = DoubleVector.fromMemorySegment(SPECIESDB, memB, bOff + strideBytes, NATIVE);

                        var a0 = DoubleVector.broadcast(SPECIESDB, memA.get(VD, (aRow0 + (long) k * a_s1) * 8L));
                        acc00 = a0.fma(b0, acc00); acc01 = a0.fma(b1, acc01);
                        var a1 = DoubleVector.broadcast(SPECIESDB, memA.get(VD, (aRow1 + (long) k * a_s1) * 8L));
                        acc10 = a1.fma(b0, acc10); acc11 = a1.fma(b1, acc11);
                        var a2 = DoubleVector.broadcast(SPECIESDB, memA.get(VD, (aRow2 + (long) k * a_s1) * 8L));
                        acc20 = a2.fma(b0, acc20); acc21 = a2.fma(b1, acc21);
                        var a3 = DoubleVector.broadcast(SPECIESDB, memA.get(VD, (aRow3 + (long) k * a_s1) * 8L));
                        acc30 = a3.fma(b0, acc30); acc31 = a3.fma(b1, acc31);
                    }

                    long cRow0 = ((long)(i + 0) * c_s0 + j) * 8L;
                    acc00.intoMemorySegment(memC, cRow0, NATIVE);
                    acc01.intoMemorySegment(memC, cRow0 + strideBytes, NATIVE);

                    long cRow1 = ((long)(i + 1) * c_s0 + j) * 8L;
                    acc10.intoMemorySegment(memC, cRow1, NATIVE);
                    acc11.intoMemorySegment(memC, cRow1 + strideBytes, NATIVE);

                    long cRow2 = ((long)(i + 2) * c_s0 + j) * 8L;
                    acc20.intoMemorySegment(memC, cRow2, NATIVE);
                    acc21.intoMemorySegment(memC, cRow2 + strideBytes, NATIVE);

                    long cRow3 = ((long)(i + 3) * c_s0 + j) * 8L;
                    acc30.intoMemorySegment(memC, cRow3, NATIVE);
                    acc31.intoMemorySegment(memC, cRow3 + strideBytes, NATIVE);
                }
            }

            // Cleanup tail rows and columns
            if (safeRowEnd < n) {
                for (int ii = safeRowEnd; ii < n; ii++) {
                    for (int jj = 0; jj < p; jj++) {
                        double sum = 0.0;
                        for (int kk = 0; kk < m; kk++) {
                            sum += memA.get(VD, ((long) ii * a_s0 + (long) kk * a_s1) * 8L)
                                 * memB.get(VD, ((long) kk * b_s0 + (long) jj * b_s1) * 8L);
                        }
                        memC.set(VD, ((long) ii * c_s0 + (long) jj * c_s1) * 8L, sum);
                    }
                }
            }
            if (safeColEnd < p) {
                for (int ii = 0; ii < safeRowEnd; ii++) {
                    for (int jj = safeColEnd; jj < p; jj++) {
                        double sum = 0.0;
                        for (int kk = 0; kk < m; kk++) {
                            sum += memA.get(VD, ((long) ii * a_s0 + (long) kk * a_s1) * 8L)
                                 * memB.get(VD, ((long) kk * b_s0 + (long) jj * b_s1) * 8L);
                        }
                        memC.set(VD, ((long) ii * c_s0 + (long) jj * c_s1) * 8L, sum);
                    }
                }
            }
        } else {
            // General strided path
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < p; j++) {
                    double sum = 0.0;
                    for (int k = 0; k < m; k++) {
                        sum += memA.get(VD, ((long) i * a_s0 + (long) k * a_s1) * 8L)
                             * memB.get(VD, ((long) k * b_s0 + (long) j * b_s1) * 8L);
                    }
                    memC.set(VD, ((long) i * c_s0 + (long) j * c_s1) * 8L, sum);
                }
            }
        }
    }

        // ---- Tier 2: Single-thread BLIS (maxDim <= 256) - zero allocation, no ForkJoin ----
    private static void blisSingleThread_Double(MemorySegment A, long a_s0, long a_s1,
                                                MemorySegment B, long b_s0, long b_s1,
                                                MemorySegment C, int n, int m, int p) {
        if (IS_AARCH64) {
            MemorySegment pB = tlPackedB_Aarch_Double.get();
            for (int jc = 0; jc < p; jc += NC_AARCH) {
                int nc = Math.min(NC_AARCH, p - jc);
                for (int pc = 0; pc < m; pc += KC) {
                    int kc = Math.min(KC, m - pc);
                    boolean isFirstKBlock = (pc == 0);
                    packB_panel_Aarch_Double(B, pB, pc, jc, kc, nc, b_s0, b_s1);
                    for (int ic = 0; ic < n; ic += MC) {
                        int mc = Math.min(MC, n - ic);
                        MemorySegment pA = tlPackedA_Aarch_Double.get();
                        packA_panel_Aarch_Double(A, pA, ic, mc, pc, kc, a_s0, a_s1);
                        gebpMacroKernel_Aarch_Double(pA, pB, C, ic, mc, jc, nc, kc, p, isFirstKBlock);
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
                    packB_panel_Arm_Double(B, pB, pc, jc, kc, nc, b_s0, b_s1);
                    for (int ic = 0; ic < n; ic += MC) {
                        int mc = Math.min(MC, n - ic);
                        MemorySegment pA = tlPackedA_Arm_Double.get();
                        packA_panel_Arm_Double(A, pA, ic, mc, pc, kc, a_s0, a_s1);
                        gebpMacroKernel_Arm_Double(pA, pB, C, ic, mc, jc, nc, kc, p, isFirstKBlock);
                    }
                }
            }
        }
    }

    // ---- Tier 3: Parallel BLIS Macro-Kernels (maxDim > 256) ----
    private static void blisArmMacro_Double(MemorySegment A, long a_s0, long a_s1,
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

    private static void blisAarchMacro_Double(MemorySegment A, long a_s0, long a_s1,
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

    private static void packA_panel_Arm_Double(MemorySegment src, MemorySegment dst, int rowStart, int mc, int colStart, int kc, long a_s0, long a_s1) {
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

    private static void packB_panel_Arm_Double(MemorySegment src, MemorySegment dst, int rowStart, int colStart, int kc, int nc, long b_s0, long b_s1) {
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

    private static void packB_panels_range_Arm_Double(MemorySegment src, MemorySegment dst, int rowStart, int colStart, int kc, long b_s0, long b_s1, int pStart, int pEnd) {
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

    private static void packB_tail_Arm_Double(MemorySegment src, MemorySegment dst, int rowStart, int colStart, int kc, int nc, long b_s0, long b_s1, int fullPanels, int tailCols) {
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

    static final class PackBTask_Arm_Double extends RecursiveAction {
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

    private static void packA_panel_Aarch_Double(MemorySegment src, MemorySegment dst, int rowStart, int mc, int colStart, int kc, long a_s0, long a_s1) {
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

    private static void packB_panel_Aarch_Double(MemorySegment src, MemorySegment dst, int rowStart, int colStart, int kc, int nc, long b_s0, long b_s1) {
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

    private static void packB_panels_range_Aarch_Double(MemorySegment src, MemorySegment dst, int rowStart, int colStart, int kc, long b_s0, long b_s1, int pStart, int pEnd) {
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

    private static void packB_tail_Aarch_Double(MemorySegment src, MemorySegment dst, int rowStart, int colStart, int kc, int nc, long b_s0, long b_s1, int fullPanels, int tailCols) {
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

    static final class PackBTask_Aarch_Double extends RecursiveAction {
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

    static final class GEBPTask_Arm_Double extends RecursiveAction {
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

    static final class GEBPTask_Aarch_Double extends RecursiveAction {
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

    static final class JcTask_Arm_Double extends RecursiveAction {
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

    static final class JcTask_Aarch_Double extends RecursiveAction {
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
private static void gebpMacroKernel_Arm_Double(MemorySegment pA, MemorySegment pB, MemorySegment C,
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

    private static void gebpMacroKernel_Aarch_Double(MemorySegment pA, MemorySegment pB, MemorySegment C,
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
    private static void microKernel6x16_Double(MemorySegment pA, long aBase, MemorySegment pB, long bBase,
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
    private static void microKernel2x16_Double(MemorySegment pA, long aBase, int rOff,
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
    private static void microKernel1x16_Double(MemorySegment pA, long aBase, int rOff,
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
    private static void microKernel8x12_Double(MemorySegment pA, long aBase, MemorySegment pB, long bBase,
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

    private static void microKernelScalar_Double(MemorySegment pA, long aBase, int rOff,
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
            vSum00 = vSum00.add(vA0.mul(vB0)); vSum01 = vSum01.add(vA0.mul(vB1));
            vSum10 = vSum10.add(vA1.mul(vB0)); vSum11 = vSum11.add(vA1.mul(vB1));
        }
        double sum00 = vSum00.reduceLanes(VectorOperators.ADD); double sum01 = vSum01.reduceLanes(VectorOperators.ADD);
        double sum10 = vSum10.reduceLanes(VectorOperators.ADD); double sum11 = vSum11.reduceLanes(VectorOperators.ADD);
        for (; k < m; k++) {
            double a0 = A.getAtIndex(VD, ((long) i * m + k)); double a1 = A.getAtIndex(VD, ((long) (i + 1) * m + k));
            double b0 = B_T.getAtIndex(VD, ((long) j * m + k)); double b1 = B_T.getAtIndex(VD, ((long) (j + 1) * m + k));
            sum00 += a0 * b0; sum01 += a0 * b1; sum10 += a1 * b0; sum11 += a1 * b1;
        }
        C.setAtIndex(VD, ((long) i * p + j), sum00); C.setAtIndex(VD, ((long) i * p + j + 1), sum01);
        C.setAtIndex(VD, ((long) (i + 1) * p + j), sum10); C.setAtIndex(VD, ((long) (i + 1) * p + j + 1), sum11);
    }

    private static void scalarDotProduct_Double(MemorySegment A, MemorySegment B_T, MemorySegment C, int m, int p, int i, int j) {
        double sum = 0.0;
        for (int k = 0; k < m; k++) {
            sum += A.getAtIndex(VD, ((long) i * m + k)) * B_T.getAtIndex(VD, ((long) j * m + k));
        }
        C.setAtIndex(VD, ((long) i * p + j), sum);
    }

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
                        dst.setAtIndex(VD, (long) j * rows + i, src.getAtIndex(VD, iStride + j));
                    }
                }
            }
        }
        return dst;
    }
    
        // =========================================================================
    // INT MATMUL
    // =========================================================================
    public static NDArray matmulInt(NDArray a, NDArray b, NDArray resArray) {
        int n = (int) a.internalShapeUnsafe()[0]; 
        int m = (int) a.internalShapeUnsafe()[1]; 
        int p = (int) b.internalShapeUnsafe()[1];
        
        int maxDim = Math.max(n, Math.max(m, p));
        if (maxDim <= 4) {
            nanoKernel_Int(a, b, resArray, n, m, p);
        } else if (maxDim <= 128) {
            directTiled_Int(a, b, resArray, n, m, p);
        } else {
            long a_s0 = a.internalStridesUnsafe()[0], a_s1 = a.internalStridesUnsafe()[1];
            long b_s0 = b.internalStridesUnsafe()[0], b_s1 = b.internalStridesUnsafe()[1];
            if (maxDim <= 256) {
                blisSingleThread_Int(a.getData(), a_s0, a_s1, b.getData(), b_s0, b_s1, resArray.getData(), n, m, p);
            } else {
                if (IS_AARCH64) {
                    blisAarchMacro_Int(a.getData(), a_s0, a_s1, b.getData(), b_s0, b_s1, resArray.getData(), n, m, p);
                } else {
                    blisArmMacro_Int(a.getData(), a_s0, a_s1, b.getData(), b_s0, b_s1, resArray.getData(), n, m, p);
                }
            }
        }
        return resArray;
    }
// ---- Tier 0: Nano kernel (maxDim <= 4) - fully unrolled scalar, ZERO allocation ----
    private static void nanoKernel_Int(NDArray a, NDArray b, NDArray resArray, int n, int m, int p) {
        long[] aStrides = a.internalStridesUnsafe();
        long[] bStrides = b.internalStridesUnsafe();
        long[] cStrides = resArray.internalStridesUnsafe();
        MemorySegment memA = a.getData();
        MemorySegment memB = b.getData();
        MemorySegment memC = resArray.getData();

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < p; j++) {
                int sum = 0;
                for (int k = 0; k < m; k++) {
                    sum += memA.get(VI, ((long) i * aStrides[0] + (long) k * aStrides[1]) * 4L)
                         * memB.get(VI, ((long) k * bStrides[0] + (long) j * bStrides[1]) * 4L);
                }
                memC.set(VI, ((long) i * cStrides[0] + (long) j * cStrides[1]) * 4L, sum);
            }
        }
    }

    // ---- Tier 1: Direct Register-Blocked SIMD Kernel (4 < maxDim <= 128) - zero packing, zero allocation ----
    private static void directTiled_Int(NDArray a, NDArray b, NDArray resArray, int n, int m, int p) {
        long[] aStrides = a.internalStridesUnsafe();
        long[] bStrides = b.internalStridesUnsafe();
        long[] cStrides = resArray.internalStridesUnsafe();
        long a_s0 = aStrides[0]; long a_s1 = aStrides[1];
        long b_s0 = bStrides[0]; long b_s1 = bStrides[1];
        long c_s0 = cStrides[0]; long c_s1 = cStrides[1];
        MemorySegment memA = a.getData();
        MemorySegment memB = b.getData();
        MemorySegment memC = resArray.getData();

        long strideBytes = (long) SPECIESINT.length() * 4L;

        if (b_s1 == 1L && c_s1 == 1L) {
            int safeRowEnd = n - (n % 4);
            int safeColEnd = p - (p % NR_INT);

            for (int i = 0; i < safeRowEnd; i += 4) {
                long aRow0 = (long)(i + 0) * a_s0;
                long aRow1 = (long)(i + 1) * a_s0;
                long aRow2 = (long)(i + 2) * a_s0;
                long aRow3 = (long)(i + 3) * a_s0;

                for (int j = 0; j < safeColEnd; j += NR_INT) {
                    var acc00 = IntVector.zero(SPECIESINT); var acc01 = IntVector.zero(SPECIESINT);
                    var acc10 = IntVector.zero(SPECIESINT); var acc11 = IntVector.zero(SPECIESINT);
                    var acc20 = IntVector.zero(SPECIESINT); var acc21 = IntVector.zero(SPECIESINT);
                    var acc30 = IntVector.zero(SPECIESINT); var acc31 = IntVector.zero(SPECIESINT);

                    int k = 0;
                    for (; k <= m - 4; k += 4) {
                        // k + 0
                        long bOff0 = ((long)(k + 0) * b_s0 + j) * 4L;
                        var b0_0 = IntVector.fromMemorySegment(SPECIESINT, memB, bOff0, NATIVE);
                        var b1_0 = IntVector.fromMemorySegment(SPECIESINT, memB, bOff0 + strideBytes, NATIVE);

                        var a0_0 = IntVector.broadcast(SPECIESINT, memA.get(VI, (aRow0 + (long)(k + 0) * a_s1) * 4L));
                        acc00 = acc00.add(a0_0.mul(b0_0)); acc01 = acc01.add(a0_0.mul(b1_0));
                        var a1_0 = IntVector.broadcast(SPECIESINT, memA.get(VI, (aRow1 + (long)(k + 0) * a_s1) * 4L));
                        acc10 = acc10.add(a1_0.mul(b0_0)); acc11 = acc11.add(a1_0.mul(b1_0));
                        var a2_0 = IntVector.broadcast(SPECIESINT, memA.get(VI, (aRow2 + (long)(k + 0) * a_s1) * 4L));
                        acc20 = acc20.add(a2_0.mul(b0_0)); acc21 = acc21.add(a2_0.mul(b1_0));
                        var a3_0 = IntVector.broadcast(SPECIESINT, memA.get(VI, (aRow3 + (long)(k + 0) * a_s1) * 4L));
                        acc30 = acc30.add(a3_0.mul(b0_0)); acc31 = acc31.add(a3_0.mul(b1_0));

                        // k + 1
                        long bOff1 = ((long)(k + 1) * b_s0 + j) * 4L;
                        var b0_1 = IntVector.fromMemorySegment(SPECIESINT, memB, bOff1, NATIVE);
                        var b1_1 = IntVector.fromMemorySegment(SPECIESINT, memB, bOff1 + strideBytes, NATIVE);

                        var a0_1 = IntVector.broadcast(SPECIESINT, memA.get(VI, (aRow0 + (long)(k + 1) * a_s1) * 4L));
                        acc00 = acc00.add(a0_1.mul(b0_1)); acc01 = acc01.add(a0_1.mul(b1_1));
                        var a1_1 = IntVector.broadcast(SPECIESINT, memA.get(VI, (aRow1 + (long)(k + 1) * a_s1) * 4L));
                        acc10 = acc10.add(a1_1.mul(b0_1)); acc11 = acc11.add(a1_1.mul(b1_1));
                        var a2_1 = IntVector.broadcast(SPECIESINT, memA.get(VI, (aRow2 + (long)(k + 1) * a_s1) * 4L));
                        acc20 = acc20.add(a2_1.mul(b0_1)); acc21 = acc21.add(a2_1.mul(b1_1));
                        var a3_1 = IntVector.broadcast(SPECIESINT, memA.get(VI, (aRow3 + (long)(k + 1) * a_s1) * 4L));
                        acc30 = acc30.add(a3_1.mul(b0_1)); acc31 = acc31.add(a3_1.mul(b1_1));

                        // k + 2
                        long bOff2 = ((long)(k + 2) * b_s0 + j) * 4L;
                        var b0_2 = IntVector.fromMemorySegment(SPECIESINT, memB, bOff2, NATIVE);
                        var b1_2 = IntVector.fromMemorySegment(SPECIESINT, memB, bOff2 + strideBytes, NATIVE);

                        var a0_2 = IntVector.broadcast(SPECIESINT, memA.get(VI, (aRow0 + (long)(k + 2) * a_s1) * 4L));
                        acc00 = acc00.add(a0_2.mul(b0_2)); acc01 = acc01.add(a0_2.mul(b1_2));
                        var a1_2 = IntVector.broadcast(SPECIESINT, memA.get(VI, (aRow1 + (long)(k + 2) * a_s1) * 4L));
                        acc10 = acc10.add(a1_2.mul(b0_2)); acc11 = acc11.add(a1_2.mul(b1_2));
                        var a2_2 = IntVector.broadcast(SPECIESINT, memA.get(VI, (aRow2 + (long)(k + 2) * a_s1) * 4L));
                        acc20 = acc20.add(a2_2.mul(b0_2)); acc21 = acc21.add(a2_2.mul(b1_2));
                        var a3_2 = IntVector.broadcast(SPECIESINT, memA.get(VI, (aRow3 + (long)(k + 2) * a_s1) * 4L));
                        acc30 = acc30.add(a3_2.mul(b0_2)); acc31 = acc31.add(a3_2.mul(b1_2));

                        // k + 3
                        long bOff3 = ((long)(k + 3) * b_s0 + j) * 4L;
                        var b0_3 = IntVector.fromMemorySegment(SPECIESINT, memB, bOff3, NATIVE);
                        var b1_3 = IntVector.fromMemorySegment(SPECIESINT, memB, bOff3 + strideBytes, NATIVE);

                        var a0_3 = IntVector.broadcast(SPECIESINT, memA.get(VI, (aRow0 + (long)(k + 3) * a_s1) * 4L));
                        acc00 = acc00.add(a0_3.mul(b0_3)); acc01 = acc01.add(a0_3.mul(b1_3));
                        var a1_3 = IntVector.broadcast(SPECIESINT, memA.get(VI, (aRow1 + (long)(k + 3) * a_s1) * 4L));
                        acc10 = acc10.add(a1_3.mul(b0_3)); acc11 = acc11.add(a1_3.mul(b1_3));
                        var a2_3 = IntVector.broadcast(SPECIESINT, memA.get(VI, (aRow2 + (long)(k + 3) * a_s1) * 4L));
                        acc20 = acc20.add(a2_3.mul(b0_3)); acc21 = acc21.add(a2_3.mul(b1_3));
                        var a3_3 = IntVector.broadcast(SPECIESINT, memA.get(VI, (aRow3 + (long)(k + 3) * a_s1) * 4L));
                        acc30 = acc30.add(a3_3.mul(b0_3)); acc31 = acc31.add(a3_3.mul(b1_3));
                    }

                    for (; k < m; k++) {
                        long bOff = ((long) k * b_s0 + j) * 4L;
                        var b0 = IntVector.fromMemorySegment(SPECIESINT, memB, bOff, NATIVE);
                        var b1 = IntVector.fromMemorySegment(SPECIESINT, memB, bOff + strideBytes, NATIVE);

                        var a0 = IntVector.broadcast(SPECIESINT, memA.get(VI, (aRow0 + (long) k * a_s1) * 4L));
                        acc00 = acc00.add(a0.mul(b0)); acc01 = acc01.add(a0.mul(b1));
                        var a1 = IntVector.broadcast(SPECIESINT, memA.get(VI, (aRow1 + (long) k * a_s1) * 4L));
                        acc10 = acc10.add(a1.mul(b0)); acc11 = acc11.add(a1.mul(b1));
                        var a2 = IntVector.broadcast(SPECIESINT, memA.get(VI, (aRow2 + (long) k * a_s1) * 4L));
                        acc20 = acc20.add(a2.mul(b0)); acc21 = acc21.add(a2.mul(b1));
                        var a3 = IntVector.broadcast(SPECIESINT, memA.get(VI, (aRow3 + (long) k * a_s1) * 4L));
                        acc30 = acc30.add(a3.mul(b0)); acc31 = acc31.add(a3.mul(b1));
                    }

                    long cRow0 = ((long)(i + 0) * c_s0 + j) * 4L;
                    acc00.intoMemorySegment(memC, cRow0, NATIVE);
                    acc01.intoMemorySegment(memC, cRow0 + strideBytes, NATIVE);

                    long cRow1 = ((long)(i + 1) * c_s0 + j) * 4L;
                    acc10.intoMemorySegment(memC, cRow1, NATIVE);
                    acc11.intoMemorySegment(memC, cRow1 + strideBytes, NATIVE);

                    long cRow2 = ((long)(i + 2) * c_s0 + j) * 4L;
                    acc20.intoMemorySegment(memC, cRow2, NATIVE);
                    acc21.intoMemorySegment(memC, cRow2 + strideBytes, NATIVE);

                    long cRow3 = ((long)(i + 3) * c_s0 + j) * 4L;
                    acc30.intoMemorySegment(memC, cRow3, NATIVE);
                    acc31.intoMemorySegment(memC, cRow3 + strideBytes, NATIVE);
                }
            }

            if (safeRowEnd < n) {
                for (int ii = safeRowEnd; ii < n; ii++) {
                    for (int jj = 0; jj < p; jj++) {
                        int sum = 0;
                        for (int kk = 0; kk < m; kk++) {
                            sum += memA.get(VI, ((long) ii * a_s0 + (long) kk * a_s1) * 4L)
                                 * memB.get(VI, ((long) kk * b_s0 + (long) jj * b_s1) * 4L);
                        }
                        memC.set(VI, ((long) ii * c_s0 + (long) jj * c_s1) * 4L, sum);
                    }
                }
            }
            if (safeColEnd < p) {
                for (int ii = 0; ii < safeRowEnd; ii++) {
                    for (int jj = safeColEnd; jj < p; jj++) {
                        int sum = 0;
                        for (int kk = 0; kk < m; kk++) {
                            sum += memA.get(VI, ((long) ii * a_s0 + (long) kk * a_s1) * 4L)
                                 * memB.get(VI, ((long) kk * b_s0 + (long) jj * b_s1) * 4L);
                        }
                        memC.set(VI, ((long) ii * c_s0 + (long) jj * c_s1) * 4L, sum);
                    }
                }
            }
        } else {
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < p; j++) {
                    int sum = 0;
                    for (int k = 0; k < m; k++) {
                        sum += memA.get(VI, ((long) i * a_s0 + (long) k * a_s1) * 4L)
                             * memB.get(VI, ((long) k * b_s0 + (long) j * b_s1) * 4L);
                    }
                    memC.set(VI, ((long) i * c_s0 + (long) j * c_s1) * 4L, sum);
                }
            }
        }
    }

        // ---- Tier 2: Single-thread BLIS (maxDim <= 256) - zero allocation, no ForkJoin ----
    private static void blisSingleThread_Int(MemorySegment A, long a_s0, long a_s1,
                                                MemorySegment B, long b_s0, long b_s1,
                                                MemorySegment C, int n, int m, int p) {
        if (IS_AARCH64) {
            MemorySegment pB = tlPackedB_Aarch_Int.get();
            for (int jc = 0; jc < p; jc += NC_AARCH) {
                int nc = Math.min(NC_AARCH, p - jc);
                for (int pc = 0; pc < m; pc += KC) {
                    int kc = Math.min(KC, m - pc);
                    boolean isFirstKBlock = (pc == 0);
                    packB_panel_Aarch_Int(B, pB, pc, jc, kc, nc, b_s0, b_s1);
                    for (int ic = 0; ic < n; ic += MC) {
                        int mc = Math.min(MC, n - ic);
                        MemorySegment pA = tlPackedA_Aarch_Int.get();
                        packA_panel_Aarch_Int(A, pA, ic, mc, pc, kc, a_s0, a_s1);
                        gebpMacroKernel_Aarch_Int(pA, pB, C, ic, mc, jc, nc, kc, p, isFirstKBlock);
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
                    packB_panel_Arm_Int(B, pB, pc, jc, kc, nc, b_s0, b_s1);
                    for (int ic = 0; ic < n; ic += MC) {
                        int mc = Math.min(MC, n - ic);
                        MemorySegment pA = tlPackedA_Arm_Int.get();
                        packA_panel_Arm_Int(A, pA, ic, mc, pc, kc, a_s0, a_s1);
                        gebpMacroKernel_Arm_Int(pA, pB, C, ic, mc, jc, nc, kc, p, isFirstKBlock);
                    }
                }
            }
        }
    }

    // ---- Tier 3: Parallel BLIS Macro-Kernels (maxDim > 256) ----
    private static void blisArmMacro_Int(MemorySegment A, long a_s0, long a_s1,
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

    private static void blisAarchMacro_Int(MemorySegment A, long a_s0, long a_s1,
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

    private static void packA_panel_Arm_Int(MemorySegment src, MemorySegment dst, int rowStart, int mc, int colStart, int kc, long a_s0, long a_s1) {
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

    private static void packB_panel_Arm_Int(MemorySegment src, MemorySegment dst, int rowStart, int colStart, int kc, int nc, long b_s0, long b_s1) {
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

    private static void packB_panels_range_Arm_Int(MemorySegment src, MemorySegment dst, int rowStart, int colStart, int kc, long b_s0, long b_s1, int pStart, int pEnd) {
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

    private static void packB_tail_Arm_Int(MemorySegment src, MemorySegment dst, int rowStart, int colStart, int kc, int nc, long b_s0, long b_s1, int fullPanels, int tailCols) {
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

    static final class PackBTask_Arm_Int extends RecursiveAction {
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

    private static void packA_panel_Aarch_Int(MemorySegment src, MemorySegment dst, int rowStart, int mc, int colStart, int kc, long a_s0, long a_s1) {
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

    private static void packB_panel_Aarch_Int(MemorySegment src, MemorySegment dst, int rowStart, int colStart, int kc, int nc, long b_s0, long b_s1) {
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

    private static void packB_panels_range_Aarch_Int(MemorySegment src, MemorySegment dst, int rowStart, int colStart, int kc, long b_s0, long b_s1, int pStart, int pEnd) {
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

    private static void packB_tail_Aarch_Int(MemorySegment src, MemorySegment dst, int rowStart, int colStart, int kc, int nc, long b_s0, long b_s1, int fullPanels, int tailCols) {
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

    static final class PackBTask_Aarch_Int extends RecursiveAction {
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

    static final class GEBPTask_Arm_Int extends RecursiveAction {
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

    static final class GEBPTask_Aarch_Int extends RecursiveAction {
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

    static final class JcTask_Arm_Int extends RecursiveAction {
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

    static final class JcTask_Aarch_Int extends RecursiveAction {
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
private static void gebpMacroKernel_Arm_Int(MemorySegment pA, MemorySegment pB, MemorySegment C,
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

    private static void gebpMacroKernel_Aarch_Int(MemorySegment pA, MemorySegment pB, MemorySegment C,
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
    private static void microKernel6x16_Int(MemorySegment pA, long aBase, MemorySegment pB, long bBase,
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
    private static void microKernel2x16_Int(MemorySegment pA, long aBase, int rOff,
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
    private static void microKernel1x16_Int(MemorySegment pA, long aBase, int rOff,
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
    private static void microKernel8x12_Int(MemorySegment pA, long aBase, MemorySegment pB, long bBase,
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

    private static void microKernelScalar_Int(MemorySegment pA, long aBase, int rOff,
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
            int a0 = A.getAtIndex(VI, ((long) i * m + k)); int a1 = A.getAtIndex(VI, ((long) (i + 1) * m + k));
            int b0 = B_T.getAtIndex(VI, ((long) j * m + k)); int b1 = B_T.getAtIndex(VI, ((long) (j + 1) * m + k));
            sum00 += a0 * b0; sum01 += a0 * b1; sum10 += a1 * b0; sum11 += a1 * b1;
        }
        C.setAtIndex(VI, ((long) i * p + j), sum00); C.setAtIndex(VI, ((long) i * p + j + 1), sum01);
        C.setAtIndex(VI, ((long) (i + 1) * p + j), sum10); C.setAtIndex(VI, ((long) (i + 1) * p + j + 1), sum11);
    }

    private static void scalarDotProduct_Int(MemorySegment A, MemorySegment B_T, MemorySegment C, int m, int p, int i, int j) {
        int sum = 0;
        for (int k = 0; k < m; k++) {
            sum += A.getAtIndex(VI, ((long) i * m + k)) * B_T.getAtIndex(VI, ((long) j * m + k));
        }
        C.setAtIndex(VI, ((long) i * p + j), sum);
    }

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
                        dst.setAtIndex(VI, (long) j * rows + i, src.getAtIndex(VI, iStride + j));
                    }
                }
            }
        }
        return dst;
    }
}
