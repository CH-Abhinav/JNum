package jnum.internal.kernel.linalg.matmul;

import static jnum.internal.Constants.*;

import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jnum.NDArray;

/**
 * Tier 1: Direct Register-Blocked SIMD Kernel (4 < maxDim <= 128).
 * Operates directly on source memory segments without panel packing
 * or heap allocation, unrolling 4 rows x 2 SIMD vectors across columns.
 */
public final class MatMulTier1DirectTiled {

    private static final ByteOrder NATIVE = NATIVE_ORDER;

    private MatMulTier1DirectTiled() {
        throw new AssertionError("Utility class");
    }

    // =========================================================================
    // FLOAT DIRECT TILED (4x2 vectors)
    // =========================================================================
    public static void matmulFloat(NDArray a, NDArray b, NDArray resArray, int n, int m, int p) {
        long[] aStrides = a.internalStridesUnsafe();
        long[] bStrides = b.internalStridesUnsafe();
        long[] cStrides = resArray.internalStridesUnsafe();
        long a_s0 = aStrides[0]; long a_s1 = aStrides[1];
        long b_s0 = bStrides[0]; long b_s1 = bStrides[1];
        long c_s0 = cStrides[0]; long c_s1 = cStrides[1];
        MemorySegment memA = a.getData();
        MemorySegment memB = b.getData();
        MemorySegment memC = resArray.getData();

        int nr = NR_F32;
        long vecBytes = VEC_BYTES_F32;

        if (b_s1 == 1L && c_s1 == 1L) {
            int safeRowEnd = n - (n % 4);
            int safeColEnd = p - (p % nr);

            for (int i = 0; i < safeRowEnd; i += 4) {
                long aRow0 = (long)(i + 0) * a_s0;
                long aRow1 = (long)(i + 1) * a_s0;
                long aRow2 = (long)(i + 2) * a_s0;
                long aRow3 = (long)(i + 3) * a_s0;

                for (int j = 0; j < safeColEnd; j += nr) {
                    var acc00 = FloatVector.zero(SPECIES_F32); var acc01 = FloatVector.zero(SPECIES_F32);
                    var acc10 = FloatVector.zero(SPECIES_F32); var acc11 = FloatVector.zero(SPECIES_F32);
                    var acc20 = FloatVector.zero(SPECIES_F32); var acc21 = FloatVector.zero(SPECIES_F32);
                    var acc30 = FloatVector.zero(SPECIES_F32); var acc31 = FloatVector.zero(SPECIES_F32);

                    int k = 0;
                    for (; k <= m - 4; k += 4) {
                        // k + 0
                        long bOff0 = ((long)(k + 0) * b_s0 + j) * 4L;
                        var b0_0 = FloatVector.fromMemorySegment(SPECIES_F32, memB, bOff0, NATIVE);
                        var b1_0 = FloatVector.fromMemorySegment(SPECIES_F32, memB, bOff0 + vecBytes, NATIVE);

                        var a0_0 = FloatVector.broadcast(SPECIES_F32, memA.get(VF, (aRow0 + (long)(k + 0) * a_s1) * 4L));
                        acc00 = a0_0.fma(b0_0, acc00); acc01 = a0_0.fma(b1_0, acc01);
                        var a1_0 = FloatVector.broadcast(SPECIES_F32, memA.get(VF, (aRow1 + (long)(k + 0) * a_s1) * 4L));
                        acc10 = a1_0.fma(b0_0, acc10); acc11 = a1_0.fma(b1_0, acc11);
                        var a2_0 = FloatVector.broadcast(SPECIES_F32, memA.get(VF, (aRow2 + (long)(k + 0) * a_s1) * 4L));
                        acc20 = a2_0.fma(b0_0, acc20); acc21 = a2_0.fma(b1_0, acc21);
                        var a3_0 = FloatVector.broadcast(SPECIES_F32, memA.get(VF, (aRow3 + (long)(k + 0) * a_s1) * 4L));
                        acc30 = a3_0.fma(b0_0, acc30); acc31 = a3_0.fma(b1_0, acc31);

                        // k + 1
                        long bOff1 = ((long)(k + 1) * b_s0 + j) * 4L;
                        var b0_1 = FloatVector.fromMemorySegment(SPECIES_F32, memB, bOff1, NATIVE);
                        var b1_1 = FloatVector.fromMemorySegment(SPECIES_F32, memB, bOff1 + vecBytes, NATIVE);

                        var a0_1 = FloatVector.broadcast(SPECIES_F32, memA.get(VF, (aRow0 + (long)(k + 1) * a_s1) * 4L));
                        acc00 = a0_1.fma(b0_1, acc00); acc01 = a0_1.fma(b1_1, acc01);
                        var a1_1 = FloatVector.broadcast(SPECIES_F32, memA.get(VF, (aRow1 + (long)(k + 1) * a_s1) * 4L));
                        acc10 = a1_1.fma(b0_1, acc10); acc11 = a1_1.fma(b1_1, acc11);
                        var a2_1 = FloatVector.broadcast(SPECIES_F32, memA.get(VF, (aRow2 + (long)(k + 1) * a_s1) * 4L));
                        acc20 = a2_1.fma(b0_1, acc20); acc21 = a2_1.fma(b1_1, acc21);
                        var a3_1 = FloatVector.broadcast(SPECIES_F32, memA.get(VF, (aRow3 + (long)(k + 1) * a_s1) * 4L));
                        acc30 = a3_1.fma(b0_1, acc30); acc31 = a3_1.fma(b1_1, acc31);

                        // k + 2
                        long bOff2 = ((long)(k + 2) * b_s0 + j) * 4L;
                        var b0_2 = FloatVector.fromMemorySegment(SPECIES_F32, memB, bOff2, NATIVE);
                        var b1_2 = FloatVector.fromMemorySegment(SPECIES_F32, memB, bOff2 + vecBytes, NATIVE);

                        var a0_2 = FloatVector.broadcast(SPECIES_F32, memA.get(VF, (aRow0 + (long)(k + 2) * a_s1) * 4L));
                        acc00 = a0_2.fma(b0_2, acc00); acc01 = a0_2.fma(b1_2, acc01);
                        var a1_2 = FloatVector.broadcast(SPECIES_F32, memA.get(VF, (aRow1 + (long)(k + 2) * a_s1) * 4L));
                        acc10 = a1_2.fma(b0_2, acc10); acc11 = a1_2.fma(b1_2, acc11);
                        var a2_2 = FloatVector.broadcast(SPECIES_F32, memA.get(VF, (aRow2 + (long)(k + 2) * a_s1) * 4L));
                        acc20 = a2_2.fma(b0_2, acc20); acc21 = a2_2.fma(b1_2, acc21);
                        var a3_2 = FloatVector.broadcast(SPECIES_F32, memA.get(VF, (aRow3 + (long)(k + 2) * a_s1) * 4L));
                        acc30 = a3_2.fma(b0_2, acc30); acc31 = a3_2.fma(b1_2, acc31);

                        // k + 3
                        long bOff3 = ((long)(k + 3) * b_s0 + j) * 4L;
                        var b0_3 = FloatVector.fromMemorySegment(SPECIES_F32, memB, bOff3, NATIVE);
                        var b1_3 = FloatVector.fromMemorySegment(SPECIES_F32, memB, bOff3 + vecBytes, NATIVE);

                        var a0_3 = FloatVector.broadcast(SPECIES_F32, memA.get(VF, (aRow0 + (long)(k + 3) * a_s1) * 4L));
                        acc00 = a0_3.fma(b0_3, acc00); acc01 = a0_3.fma(b1_3, acc01);
                        var a1_3 = FloatVector.broadcast(SPECIES_F32, memA.get(VF, (aRow1 + (long)(k + 3) * a_s1) * 4L));
                        acc10 = a1_3.fma(b0_3, acc10); acc11 = a1_3.fma(b1_3, acc11);
                        var a2_3 = FloatVector.broadcast(SPECIES_F32, memA.get(VF, (aRow2 + (long)(k + 3) * a_s1) * 4L));
                        acc20 = a2_3.fma(b0_3, acc20); acc21 = a2_3.fma(b1_3, acc21);
                        var a3_3 = FloatVector.broadcast(SPECIES_F32, memA.get(VF, (aRow3 + (long)(k + 3) * a_s1) * 4L));
                        acc30 = a3_3.fma(b0_3, acc30); acc31 = a3_3.fma(b1_3, acc31);
                    }

                    for (; k < m; k++) {
                        long bOff = ((long) k * b_s0 + j) * 4L;
                        var b0 = FloatVector.fromMemorySegment(SPECIES_F32, memB, bOff, NATIVE);
                        var b1 = FloatVector.fromMemorySegment(SPECIES_F32, memB, bOff + vecBytes, NATIVE);

                        var a0 = FloatVector.broadcast(SPECIES_F32, memA.get(VF, (aRow0 + (long) k * a_s1) * 4L));
                        acc00 = a0.fma(b0, acc00); acc01 = a0.fma(b1, acc01);
                        var a1 = FloatVector.broadcast(SPECIES_F32, memA.get(VF, (aRow1 + (long) k * a_s1) * 4L));
                        acc10 = a1.fma(b0, acc10); acc11 = a1.fma(b1, acc11);
                        var a2 = FloatVector.broadcast(SPECIES_F32, memA.get(VF, (aRow2 + (long) k * a_s1) * 4L));
                        acc20 = a2.fma(b0, acc20); acc21 = a2.fma(b1, acc21);
                        var a3 = FloatVector.broadcast(SPECIES_F32, memA.get(VF, (aRow3 + (long) k * a_s1) * 4L));
                        acc30 = a3.fma(b0, acc30); acc31 = a3.fma(b1, acc31);
                    }

                    long cRow0 = ((long)(i + 0) * c_s0 + j) * 4L;
                    acc00.intoMemorySegment(memC, cRow0, NATIVE);
                    acc01.intoMemorySegment(memC, cRow0 + vecBytes, NATIVE);

                    long cRow1 = ((long)(i + 1) * c_s0 + j) * 4L;
                    acc10.intoMemorySegment(memC, cRow1, NATIVE);
                    acc11.intoMemorySegment(memC, cRow1 + vecBytes, NATIVE);

                    long cRow2 = ((long)(i + 2) * c_s0 + j) * 4L;
                    acc20.intoMemorySegment(memC, cRow2, NATIVE);
                    acc21.intoMemorySegment(memC, cRow2 + vecBytes, NATIVE);

                    long cRow3 = ((long)(i + 3) * c_s0 + j) * 4L;
                    acc30.intoMemorySegment(memC, cRow3, NATIVE);
                    acc31.intoMemorySegment(memC, cRow3 + vecBytes, NATIVE);
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

    // =========================================================================
    // DOUBLE DIRECT TILED (4x2 vectors)
    // =========================================================================
    public static void matmulDouble(NDArray a, NDArray b, NDArray resArray, int n, int m, int p) {
        long[] aStrides = a.internalStridesUnsafe();
        long[] bStrides = b.internalStridesUnsafe();
        long[] cStrides = resArray.internalStridesUnsafe();
        long a_s0 = aStrides[0]; long a_s1 = aStrides[1];
        long b_s0 = bStrides[0]; long b_s1 = bStrides[1];
        long c_s0 = cStrides[0]; long c_s1 = cStrides[1];
        MemorySegment memA = a.getData();
        MemorySegment memB = b.getData();
        MemorySegment memC = resArray.getData();

        int nr = NR_F64;
        long vecBytes = VEC_BYTES_F64;

        if (b_s1 == 1L && c_s1 == 1L) {
            int safeRowEnd = n - (n % 4);
            int safeColEnd = p - (p % nr);

            for (int i = 0; i < safeRowEnd; i += 4) {
                long aRow0 = (long)(i + 0) * a_s0;
                long aRow1 = (long)(i + 1) * a_s0;
                long aRow2 = (long)(i + 2) * a_s0;
                long aRow3 = (long)(i + 3) * a_s0;

                for (int j = 0; j < safeColEnd; j += nr) {
                    var acc00 = DoubleVector.zero(SPECIES_F64); var acc01 = DoubleVector.zero(SPECIES_F64);
                    var acc10 = DoubleVector.zero(SPECIES_F64); var acc11 = DoubleVector.zero(SPECIES_F64);
                    var acc20 = DoubleVector.zero(SPECIES_F64); var acc21 = DoubleVector.zero(SPECIES_F64);
                    var acc30 = DoubleVector.zero(SPECIES_F64); var acc31 = DoubleVector.zero(SPECIES_F64);

                    int k = 0;
                    for (; k <= m - 4; k += 4) {
                        long bOff0 = ((long)(k + 0) * b_s0 + j) * 8L;
                        var b0_0 = DoubleVector.fromMemorySegment(SPECIES_F64, memB, bOff0, NATIVE);
                        var b1_0 = DoubleVector.fromMemorySegment(SPECIES_F64, memB, bOff0 + vecBytes, NATIVE);

                        var a0_0 = DoubleVector.broadcast(SPECIES_F64, memA.get(VD, (aRow0 + (long)(k + 0) * a_s1) * 8L));
                        acc00 = a0_0.fma(b0_0, acc00); acc01 = a0_0.fma(b1_0, acc01);
                        var a1_0 = DoubleVector.broadcast(SPECIES_F64, memA.get(VD, (aRow1 + (long)(k + 0) * a_s1) * 8L));
                        acc10 = a1_0.fma(b0_0, acc10); acc11 = a1_0.fma(b1_0, acc11);
                        var a2_0 = DoubleVector.broadcast(SPECIES_F64, memA.get(VD, (aRow2 + (long)(k + 0) * a_s1) * 8L));
                        acc20 = a2_0.fma(b0_0, acc20); acc21 = a2_0.fma(b1_0, acc21);
                        var a3_0 = DoubleVector.broadcast(SPECIES_F64, memA.get(VD, (aRow3 + (long)(k + 0) * a_s1) * 8L));
                        acc30 = a3_0.fma(b0_0, acc30); acc31 = a3_0.fma(b1_0, acc31);

                        long bOff1 = ((long)(k + 1) * b_s0 + j) * 8L;
                        var b0_1 = DoubleVector.fromMemorySegment(SPECIES_F64, memB, bOff1, NATIVE);
                        var b1_1 = DoubleVector.fromMemorySegment(SPECIES_F64, memB, bOff1 + vecBytes, NATIVE);

                        var a0_1 = DoubleVector.broadcast(SPECIES_F64, memA.get(VD, (aRow0 + (long)(k + 1) * a_s1) * 8L));
                        acc00 = a0_1.fma(b0_1, acc00); acc01 = a0_1.fma(b1_1, acc01);
                        var a1_1 = DoubleVector.broadcast(SPECIES_F64, memA.get(VD, (aRow1 + (long)(k + 1) * a_s1) * 8L));
                        acc10 = a1_1.fma(b0_1, acc10); acc11 = a1_1.fma(b1_1, acc11);
                        var a2_1 = DoubleVector.broadcast(SPECIES_F64, memA.get(VD, (aRow2 + (long)(k + 1) * a_s1) * 8L));
                        acc20 = a2_1.fma(b0_1, acc20); acc21 = a2_1.fma(b1_1, acc21);
                        var a3_1 = DoubleVector.broadcast(SPECIES_F64, memA.get(VD, (aRow3 + (long)(k + 1) * a_s1) * 8L));
                        acc30 = a3_1.fma(b0_1, acc30); acc31 = a3_1.fma(b1_1, acc31);

                        long bOff2 = ((long)(k + 2) * b_s0 + j) * 8L;
                        var b0_2 = DoubleVector.fromMemorySegment(SPECIES_F64, memB, bOff2, NATIVE);
                        var b1_2 = DoubleVector.fromMemorySegment(SPECIES_F64, memB, bOff2 + vecBytes, NATIVE);

                        var a0_2 = DoubleVector.broadcast(SPECIES_F64, memA.get(VD, (aRow0 + (long)(k + 2) * a_s1) * 8L));
                        acc00 = a0_2.fma(b0_2, acc00); acc01 = a0_2.fma(b1_2, acc01);
                        var a1_2 = DoubleVector.broadcast(SPECIES_F64, memA.get(VD, (aRow1 + (long)(k + 2) * a_s1) * 8L));
                        acc10 = a1_2.fma(b0_2, acc10); acc11 = a1_2.fma(b1_2, acc11);
                        var a2_2 = DoubleVector.broadcast(SPECIES_F64, memA.get(VD, (aRow2 + (long)(k + 2) * a_s1) * 8L));
                        acc20 = a2_2.fma(b0_2, acc20); acc21 = a2_2.fma(b1_2, acc21);
                        var a3_2 = DoubleVector.broadcast(SPECIES_F64, memA.get(VD, (aRow3 + (long)(k + 2) * a_s1) * 8L));
                        acc30 = a3_2.fma(b0_2, acc30); acc31 = a3_2.fma(b1_2, acc31);

                        long bOff3 = ((long)(k + 3) * b_s0 + j) * 8L;
                        var b0_3 = DoubleVector.fromMemorySegment(SPECIES_F64, memB, bOff3, NATIVE);
                        var b1_3 = DoubleVector.fromMemorySegment(SPECIES_F64, memB, bOff3 + vecBytes, NATIVE);

                        var a0_3 = DoubleVector.broadcast(SPECIES_F64, memA.get(VD, (aRow0 + (long)(k + 3) * a_s1) * 8L));
                        acc00 = a0_3.fma(b0_3, acc00); acc01 = a0_3.fma(b1_3, acc01);
                        var a1_3 = DoubleVector.broadcast(SPECIES_F64, memA.get(VD, (aRow1 + (long)(k + 3) * a_s1) * 8L));
                        acc10 = a1_3.fma(b0_3, acc10); acc11 = a1_3.fma(b1_3, acc11);
                        var a2_3 = DoubleVector.broadcast(SPECIES_F64, memA.get(VD, (aRow2 + (long)(k + 3) * a_s1) * 8L));
                        acc20 = a2_3.fma(b0_3, acc20); acc21 = a2_3.fma(b1_3, acc21);
                        var a3_3 = DoubleVector.broadcast(SPECIES_F64, memA.get(VD, (aRow3 + (long)(k + 3) * a_s1) * 8L));
                        acc30 = a3_3.fma(b0_3, acc30); acc31 = a3_3.fma(b1_3, acc31);
                    }

                    for (; k < m; k++) {
                        long bOff = ((long) k * b_s0 + j) * 8L;
                        var b0 = DoubleVector.fromMemorySegment(SPECIES_F64, memB, bOff, NATIVE);
                        var b1 = DoubleVector.fromMemorySegment(SPECIES_F64, memB, bOff + vecBytes, NATIVE);

                        var a0 = DoubleVector.broadcast(SPECIES_F64, memA.get(VD, (aRow0 + (long) k * a_s1) * 8L));
                        acc00 = a0.fma(b0, acc00); acc01 = a0.fma(b1, acc01);
                        var a1 = DoubleVector.broadcast(SPECIES_F64, memA.get(VD, (aRow1 + (long) k * a_s1) * 8L));
                        acc10 = a1.fma(b0, acc10); acc11 = a1.fma(b1, acc11);
                        var a2 = DoubleVector.broadcast(SPECIES_F64, memA.get(VD, (aRow2 + (long) k * a_s1) * 8L));
                        acc20 = a2.fma(b0, acc20); acc21 = a2.fma(b1, acc21);
                        var a3 = DoubleVector.broadcast(SPECIES_F64, memA.get(VD, (aRow3 + (long) k * a_s1) * 8L));
                        acc30 = a3.fma(b0, acc30); acc31 = a3.fma(b1, acc31);
                    }

                    long cRow0 = ((long)(i + 0) * c_s0 + j) * 8L;
                    acc00.intoMemorySegment(memC, cRow0, NATIVE);
                    acc01.intoMemorySegment(memC, cRow0 + vecBytes, NATIVE);

                    long cRow1 = ((long)(i + 1) * c_s0 + j) * 8L;
                    acc10.intoMemorySegment(memC, cRow1, NATIVE);
                    acc11.intoMemorySegment(memC, cRow1 + vecBytes, NATIVE);

                    long cRow2 = ((long)(i + 2) * c_s0 + j) * 8L;
                    acc20.intoMemorySegment(memC, cRow2, NATIVE);
                    acc21.intoMemorySegment(memC, cRow2 + vecBytes, NATIVE);

                    long cRow3 = ((long)(i + 3) * c_s0 + j) * 8L;
                    acc30.intoMemorySegment(memC, cRow3, NATIVE);
                    acc31.intoMemorySegment(memC, cRow3 + vecBytes, NATIVE);
                }
            }

            if (safeRowEnd < n) {
                for (int ii = safeRowEnd; ii < n; ii++) {
                    for (int jj = 0; jj < p; jj++) {
                        double sum = 0.0d;
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
                        double sum = 0.0d;
                        for (int kk = 0; kk < m; kk++) {
                            sum += memA.get(VD, ((long) ii * a_s0 + (long) kk * a_s1) * 8L)
                                 * memB.get(VD, ((long) kk * b_s0 + (long) jj * b_s1) * 8L);
                        }
                        memC.set(VD, ((long) ii * c_s0 + (long) jj * c_s1) * 8L, sum);
                    }
                }
            }
        } else {
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < p; j++) {
                    double sum = 0.0d;
                    for (int k = 0; k < m; k++) {
                        sum += memA.get(VD, ((long) i * a_s0 + (long) k * a_s1) * 8L)
                             * memB.get(VD, ((long) k * b_s0 + (long) j * b_s1) * 8L);
                    }
                    memC.set(VD, ((long) i * c_s0 + (long) j * c_s1) * 8L, sum);
                }
            }
        }
    }

    // =========================================================================
    // INT DIRECT TILED (4x2 vectors)
    // =========================================================================
    public static void matmulInt(NDArray a, NDArray b, NDArray resArray, int n, int m, int p) {
        long[] aStrides = a.internalStridesUnsafe();
        long[] bStrides = b.internalStridesUnsafe();
        long[] cStrides = resArray.internalStridesUnsafe();
        long a_s0 = aStrides[0]; long a_s1 = aStrides[1];
        long b_s0 = bStrides[0]; long b_s1 = bStrides[1];
        long c_s0 = cStrides[0]; long c_s1 = cStrides[1];
        MemorySegment memA = a.getData();
        MemorySegment memB = b.getData();
        MemorySegment memC = resArray.getData();

        int nr = NR_I32;
        long vecBytes = VEC_BYTES_I32;

        if (b_s1 == 1L && c_s1 == 1L) {
            int safeRowEnd = n - (n % 4);
            int safeColEnd = p - (p % nr);

            for (int i = 0; i < safeRowEnd; i += 4) {
                long aRow0 = (long)(i + 0) * a_s0;
                long aRow1 = (long)(i + 1) * a_s0;
                long aRow2 = (long)(i + 2) * a_s0;
                long aRow3 = (long)(i + 3) * a_s0;

                for (int j = 0; j < safeColEnd; j += nr) {
                    var acc00 = IntVector.zero(SPECIES_I32); var acc01 = IntVector.zero(SPECIES_I32);
                    var acc10 = IntVector.zero(SPECIES_I32); var acc11 = IntVector.zero(SPECIES_I32);
                    var acc20 = IntVector.zero(SPECIES_I32); var acc21 = IntVector.zero(SPECIES_I32);
                    var acc30 = IntVector.zero(SPECIES_I32); var acc31 = IntVector.zero(SPECIES_I32);

                    int k = 0;
                    for (; k <= m - 4; k += 4) {
                        long bOff0 = ((long)(k + 0) * b_s0 + j) * 4L;
                        var b0_0 = IntVector.fromMemorySegment(SPECIES_I32, memB, bOff0, NATIVE);
                        var b1_0 = IntVector.fromMemorySegment(SPECIES_I32, memB, bOff0 + vecBytes, NATIVE);

                        var a0_0 = IntVector.broadcast(SPECIES_I32, memA.get(VI, (aRow0 + (long)(k + 0) * a_s1) * 4L));
                        acc00 = a0_0.mul(b0_0).add(acc00); acc01 = a0_0.mul(b1_0).add(acc01);
                        var a1_0 = IntVector.broadcast(SPECIES_I32, memA.get(VI, (aRow1 + (long)(k + 0) * a_s1) * 4L));
                        acc10 = a1_0.mul(b0_0).add(acc10); acc11 = a1_0.mul(b1_0).add(acc11);
                        var a2_0 = IntVector.broadcast(SPECIES_I32, memA.get(VI, (aRow2 + (long)(k + 0) * a_s1) * 4L));
                        acc20 = a2_0.mul(b0_0).add(acc20); acc21 = a2_0.mul(b1_0).add(acc21);
                        var a3_0 = IntVector.broadcast(SPECIES_I32, memA.get(VI, (aRow3 + (long)(k + 0) * a_s1) * 4L));
                        acc30 = a3_0.mul(b0_0).add(acc30); acc31 = a3_0.mul(b1_0).add(acc31);

                        long bOff1 = ((long)(k + 1) * b_s0 + j) * 4L;
                        var b0_1 = IntVector.fromMemorySegment(SPECIES_I32, memB, bOff1, NATIVE);
                        var b1_1 = IntVector.fromMemorySegment(SPECIES_I32, memB, bOff1 + vecBytes, NATIVE);

                        var a0_1 = IntVector.broadcast(SPECIES_I32, memA.get(VI, (aRow0 + (long)(k + 1) * a_s1) * 4L));
                        acc00 = a0_1.mul(b0_1).add(acc00); acc01 = a0_1.mul(b1_1).add(acc01);
                        var a1_1 = IntVector.broadcast(SPECIES_I32, memA.get(VI, (aRow1 + (long)(k + 1) * a_s1) * 4L));
                        acc10 = a1_1.mul(b0_1).add(acc10); acc11 = a1_1.mul(b1_1).add(acc11);
                        var a2_1 = IntVector.broadcast(SPECIES_I32, memA.get(VI, (aRow2 + (long)(k + 1) * a_s1) * 4L));
                        acc20 = a2_1.mul(b0_1).add(acc20); acc21 = a2_1.mul(b1_1).add(acc21);
                        var a3_1 = IntVector.broadcast(SPECIES_I32, memA.get(VI, (aRow3 + (long)(k + 1) * a_s1) * 4L));
                        acc30 = a3_1.mul(b0_1).add(acc30); acc31 = a3_1.mul(b1_1).add(acc31);

                        long bOff2 = ((long)(k + 2) * b_s0 + j) * 4L;
                        var b0_2 = IntVector.fromMemorySegment(SPECIES_I32, memB, bOff2, NATIVE);
                        var b1_2 = IntVector.fromMemorySegment(SPECIES_I32, memB, bOff2 + vecBytes, NATIVE);

                        var a0_2 = IntVector.broadcast(SPECIES_I32, memA.get(VI, (aRow0 + (long)(k + 2) * a_s1) * 4L));
                        acc00 = a0_2.mul(b0_2).add(acc00); acc01 = a0_2.mul(b1_2).add(acc01);
                        var a1_2 = IntVector.broadcast(SPECIES_I32, memA.get(VI, (aRow1 + (long)(k + 2) * a_s1) * 4L));
                        acc10 = a1_2.mul(b0_2).add(acc10); acc11 = a1_2.mul(b1_2).add(acc11);
                        var a2_2 = IntVector.broadcast(SPECIES_I32, memA.get(VI, (aRow2 + (long)(k + 2) * a_s1) * 4L));
                        acc20 = a2_2.mul(b0_2).add(acc20); acc21 = a2_2.mul(b1_2).add(acc21);
                        var a3_2 = IntVector.broadcast(SPECIES_I32, memA.get(VI, (aRow3 + (long)(k + 2) * a_s1) * 4L));
                        acc30 = a3_2.mul(b0_2).add(acc30); acc31 = a3_2.mul(b1_2).add(acc31);

                        long bOff3 = ((long)(k + 3) * b_s0 + j) * 4L;
                        var b0_3 = IntVector.fromMemorySegment(SPECIES_I32, memB, bOff3, NATIVE);
                        var b1_3 = IntVector.fromMemorySegment(SPECIES_I32, memB, bOff3 + vecBytes, NATIVE);

                        var a0_3 = IntVector.broadcast(SPECIES_I32, memA.get(VI, (aRow0 + (long)(k + 3) * a_s1) * 4L));
                        acc00 = a0_3.mul(b0_3).add(acc00); acc01 = a0_3.mul(b1_3).add(acc01);
                        var a1_3 = IntVector.broadcast(SPECIES_I32, memA.get(VI, (aRow1 + (long)(k + 3) * a_s1) * 4L));
                        acc10 = a1_3.mul(b0_3).add(acc10); acc11 = a1_3.mul(b1_3).add(acc11);
                        var a2_3 = IntVector.broadcast(SPECIES_I32, memA.get(VI, (aRow2 + (long)(k + 3) * a_s1) * 4L));
                        acc20 = a2_3.mul(b0_3).add(acc20); acc21 = a2_3.mul(b1_3).add(acc21);
                        var a3_3 = IntVector.broadcast(SPECIES_I32, memA.get(VI, (aRow3 + (long)(k + 3) * a_s1) * 4L));
                        acc30 = a3_3.mul(b0_3).add(acc30); acc31 = a3_3.mul(b1_3).add(acc31);
                    }

                    for (; k < m; k++) {
                        long bOff = ((long) k * b_s0 + j) * 4L;
                        var b0 = IntVector.fromMemorySegment(SPECIES_I32, memB, bOff, NATIVE);
                        var b1 = IntVector.fromMemorySegment(SPECIES_I32, memB, bOff + vecBytes, NATIVE);

                        var a0 = IntVector.broadcast(SPECIES_I32, memA.get(VI, (aRow0 + (long) k * a_s1) * 4L));
                        acc00 = a0.mul(b0).add(acc00); acc01 = a0.mul(b1).add(acc01);
                        var a1 = IntVector.broadcast(SPECIES_I32, memA.get(VI, (aRow1 + (long) k * a_s1) * 4L));
                        acc10 = a1.mul(b0).add(acc10); acc11 = a1.mul(b1).add(acc11);
                        var a2 = IntVector.broadcast(SPECIES_I32, memA.get(VI, (aRow2 + (long) k * a_s1) * 4L));
                        acc20 = a2.mul(b0).add(acc20); acc21 = a2.mul(b1).add(acc21);
                        var a3 = IntVector.broadcast(SPECIES_I32, memA.get(VI, (aRow3 + (long) k * a_s1) * 4L));
                        acc30 = a3.mul(b0).add(acc30); acc31 = a3.mul(b1).add(acc31);
                    }

                    long cRow0 = ((long)(i + 0) * c_s0 + j) * 4L;
                    acc00.intoMemorySegment(memC, cRow0, NATIVE);
                    acc01.intoMemorySegment(memC, cRow0 + vecBytes, NATIVE);

                    long cRow1 = ((long)(i + 1) * c_s0 + j) * 4L;
                    acc10.intoMemorySegment(memC, cRow1, NATIVE);
                    acc11.intoMemorySegment(memC, cRow1 + vecBytes, NATIVE);

                    long cRow2 = ((long)(i + 2) * c_s0 + j) * 4L;
                    acc20.intoMemorySegment(memC, cRow2, NATIVE);
                    acc21.intoMemorySegment(memC, cRow2 + vecBytes, NATIVE);

                    long cRow3 = ((long)(i + 3) * c_s0 + j) * 4L;
                    acc30.intoMemorySegment(memC, cRow3, NATIVE);
                    acc31.intoMemorySegment(memC, cRow3 + vecBytes, NATIVE);
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
}
