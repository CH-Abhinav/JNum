package jnum.internal.kernel.linalg.matmul;

import static jnum.internal.Constants.*;

import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorMask;
import jnum.NDArray;

/**
 * Tier 1: Direct Register-Blocked SIMD Kernel (4 < maxDim <= 128).
 *
 * Key Platform & JDK 25-27 Optimizations:
 * 1. 2-Way Unrolled Ping-Pong Register Reuse: Uses 12 live SIMD registers
 *    (8 accumulators + 2 B vectors + 2 A broadcast registers), avoiding C2 register
 *    spills and staying well below JDK 27's LiveNodeCountInliningCutoff.
 * 2. Pure 64-Bit Byte-Pointer Induction Variables: Eliminates int-to-long casts and
 *    64-bit multiplications inside the hot k-loop so C2 hoists ScopedMemoryAccess checks.
 * 3. Vectorized Tail Rows & VectorMask Tail Columns: Replaces slow triple-nested scalar
 *    cleanup loops with 1x2V SIMD row cleanup and VectorMask column cleanup.
 */
public final class MatMulTier1DirectTiled {

    private static final ByteOrder NATIVE = NATIVE_ORDER;

    private MatMulTier1DirectTiled() {
        throw new AssertionError("Utility class");
    }

    // =========================================================================
    // 1. FLOAT DIRECT TILED (4 x 2V register blocking + SIMD tail cleanup)
    // =========================================================================

    public static void matmulFloat(NDArray a, NDArray b, NDArray resArray, int n, int m, int p) {
        long[] aStrides = a.internalStridesUnsafe();
        long[] bStrides = b.internalStridesUnsafe();
        long[] cStrides = resArray.internalStridesUnsafe();
        final long a_s0 = aStrides[0], a_s1 = aStrides[1];
        final long b_s0 = bStrides[0], b_s1 = bStrides[1];
        final long c_s0 = cStrides[0], c_s1 = cStrides[1];

        final MemorySegment memA = a.getData();
        final MemorySegment memB = b.getData();
        final MemorySegment memC = resArray.getData();

        final int nr = NR_F32;
        final long vecBytes = VEC_BYTES_F32;
        final long aRowBytes = a_s0 * 4L;
        final long aColBytes = a_s1 * 4L;
        final long bRowBytes = b_s0 * 4L;
        final long cRowBytes = c_s0 * 4L;

        if (b_s1 == 1L && c_s1 == 1L) {
            final int safeRowEnd = n & ~3;
            final int safeColEnd = p - (p % nr);
            final int remCols = p - safeColEnd;
            final int lanes0 = Math.min(VL_F32, remCols);
            final int lanes1 = Math.max(0, remCols - VL_F32);
            final VectorMask<Float> mask0 = (remCols > 0) ? SPECIES_F32.indexInRange(0, lanes0) : null;
            final VectorMask<Float> mask1 = (lanes1 > 0) ? SPECIES_F32.indexInRange(0, lanes1) : null;

            for (int i = 0; i < safeRowEnd; i += 4) {
                final long aBase0 = (long) i * aRowBytes;
                final long aBase1 = aBase0 + aRowBytes;
                final long aBase2 = aBase1 + aRowBytes;
                final long aBase3 = aBase2 + aRowBytes;
                final long cBase0 = (long) i * cRowBytes;

                for (int j = 0; j < safeColEnd; j += nr) {
                    var z = FloatVector.zero(SPECIES_F32);
                    FloatVector acc00 = z, acc01 = z, acc10 = z, acc11 = z;
                    FloatVector acc20 = z, acc21 = z, acc30 = z, acc31 = z;

                    long aOff0 = aBase0, aOff1 = aBase1, aOff2 = aBase2, aOff3 = aBase3;
                    long bOff = (long) j * 4L;
                    int k = 0;
                    for (; k <= m - 2; k += 2) {
                        var b0 = FloatVector.fromMemorySegment(SPECIES_F32, memB, bOff, NATIVE);
                        var b1 = FloatVector.fromMemorySegment(SPECIES_F32, memB, bOff + vecBytes, NATIVE);

                        var a0 = FloatVector.broadcast(SPECIES_F32, memA.get(VF, aOff0));
                        var a1 = FloatVector.broadcast(SPECIES_F32, memA.get(VF, aOff1));
                        acc00 = a0.fma(b0, acc00); acc01 = a0.fma(b1, acc01);
                        acc10 = a1.fma(b0, acc10); acc11 = a1.fma(b1, acc11);

                        a0 = FloatVector.broadcast(SPECIES_F32, memA.get(VF, aOff2));
                        a1 = FloatVector.broadcast(SPECIES_F32, memA.get(VF, aOff3));
                        acc20 = a0.fma(b0, acc20); acc21 = a0.fma(b1, acc21);
                        acc30 = a1.fma(b0, acc30); acc31 = a1.fma(b1, acc31);

                        long bOff1 = bOff + bRowBytes;
                        b0 = FloatVector.fromMemorySegment(SPECIES_F32, memB, bOff1, NATIVE);
                        b1 = FloatVector.fromMemorySegment(SPECIES_F32, memB, bOff1 + vecBytes, NATIVE);

                        a0 = FloatVector.broadcast(SPECIES_F32, memA.get(VF, aOff0 + aColBytes));
                        a1 = FloatVector.broadcast(SPECIES_F32, memA.get(VF, aOff1 + aColBytes));
                        acc00 = a0.fma(b0, acc00); acc01 = a0.fma(b1, acc01);
                        acc10 = a1.fma(b0, acc10); acc11 = a1.fma(b1, acc11);

                        a0 = FloatVector.broadcast(SPECIES_F32, memA.get(VF, aOff2 + aColBytes));
                        a1 = FloatVector.broadcast(SPECIES_F32, memA.get(VF, aOff3 + aColBytes));
                        acc20 = a0.fma(b0, acc20); acc21 = a0.fma(b1, acc21);
                        acc30 = a1.fma(b0, acc30); acc31 = a1.fma(b1, acc31);

                        long aStep2 = aColBytes * 2L;
                        aOff0 += aStep2; aOff1 += aStep2; aOff2 += aStep2; aOff3 += aStep2;
                        bOff = bOff1 + bRowBytes;
                    }
                    for (; k < m; k++) {
                        var b0 = FloatVector.fromMemorySegment(SPECIES_F32, memB, bOff, NATIVE);
                        var b1 = FloatVector.fromMemorySegment(SPECIES_F32, memB, bOff + vecBytes, NATIVE);

                        var a0 = FloatVector.broadcast(SPECIES_F32, memA.get(VF, aOff0));
                        var a1 = FloatVector.broadcast(SPECIES_F32, memA.get(VF, aOff1));
                        acc00 = a0.fma(b0, acc00); acc01 = a0.fma(b1, acc01);
                        acc10 = a1.fma(b0, acc10); acc11 = a1.fma(b1, acc11);

                        a0 = FloatVector.broadcast(SPECIES_F32, memA.get(VF, aOff2));
                        a1 = FloatVector.broadcast(SPECIES_F32, memA.get(VF, aOff3));
                        acc20 = a0.fma(b0, acc20); acc21 = a0.fma(b1, acc21);
                        acc30 = a1.fma(b0, acc30); acc31 = a1.fma(b1, acc31);

                        aOff0 += aColBytes; aOff1 += aColBytes; aOff2 += aColBytes; aOff3 += aColBytes;
                        bOff += bRowBytes;
                    }

                    long cRow0 = cBase0 + (long) j * 4L;
                    long cRow1 = cRow0 + cRowBytes;
                    long cRow2 = cRow1 + cRowBytes;
                    long cRow3 = cRow2 + cRowBytes;
                    acc00.intoMemorySegment(memC, cRow0, NATIVE); acc01.intoMemorySegment(memC, cRow0 + vecBytes, NATIVE);
                    acc10.intoMemorySegment(memC, cRow1, NATIVE); acc11.intoMemorySegment(memC, cRow1 + vecBytes, NATIVE);
                    acc20.intoMemorySegment(memC, cRow2, NATIVE); acc21.intoMemorySegment(memC, cRow2 + vecBytes, NATIVE);
                    acc30.intoMemorySegment(memC, cRow3, NATIVE); acc31.intoMemorySegment(memC, cRow3 + vecBytes, NATIVE);
                }

                if (remCols > 0) {
                    var z = FloatVector.zero(SPECIES_F32);
                    FloatVector acc00 = z, acc01 = z, acc10 = z, acc11 = z;
                    FloatVector acc20 = z, acc21 = z, acc30 = z, acc31 = z;

                    long aOff0 = aBase0, aOff1 = aBase1, aOff2 = aBase2, aOff3 = aBase3;
                    long bOff = (long) safeColEnd * 4L;
                    for (int k = 0; k < m; k++) {
                        var b0 = FloatVector.fromMemorySegment(SPECIES_F32, memB, bOff, NATIVE, mask0);
                        var b1 = (lanes1 > 0) ? FloatVector.fromMemorySegment(SPECIES_F32, memB, bOff + vecBytes, NATIVE, mask1) : z;

                        var a0 = FloatVector.broadcast(SPECIES_F32, memA.get(VF, aOff0));
                        var a1 = FloatVector.broadcast(SPECIES_F32, memA.get(VF, aOff1));
                        acc00 = a0.fma(b0, acc00); acc01 = a0.fma(b1, acc01);
                        acc10 = a1.fma(b0, acc10); acc11 = a1.fma(b1, acc11);

                        a0 = FloatVector.broadcast(SPECIES_F32, memA.get(VF, aOff2));
                        a1 = FloatVector.broadcast(SPECIES_F32, memA.get(VF, aOff3));
                        acc20 = a0.fma(b0, acc20); acc21 = a0.fma(b1, acc21);
                        acc30 = a1.fma(b0, acc30); acc31 = a1.fma(b1, acc31);

                        aOff0 += aColBytes; aOff1 += aColBytes; aOff2 += aColBytes; aOff3 += aColBytes;
                        bOff += bRowBytes;
                    }

                    long cRow0 = cBase0 + (long) safeColEnd * 4L;
                    long cRow1 = cRow0 + cRowBytes;
                    long cRow2 = cRow1 + cRowBytes;
                    long cRow3 = cRow2 + cRowBytes;
                    acc00.intoMemorySegment(memC, cRow0, NATIVE, mask0);
                    acc10.intoMemorySegment(memC, cRow1, NATIVE, mask0);
                    acc20.intoMemorySegment(memC, cRow2, NATIVE, mask0);
                    acc30.intoMemorySegment(memC, cRow3, NATIVE, mask0);
                    if (lanes1 > 0) {
                        acc01.intoMemorySegment(memC, cRow0 + vecBytes, NATIVE, mask1);
                        acc11.intoMemorySegment(memC, cRow1 + vecBytes, NATIVE, mask1);
                        acc21.intoMemorySegment(memC, cRow2 + vecBytes, NATIVE, mask1);
                        acc31.intoMemorySegment(memC, cRow3 + vecBytes, NATIVE, mask1);
                    }
                }
            }

            // Vectorized 1x2V cleanup for tail rows (safeRowEnd .. n)
            for (int ii = safeRowEnd; ii < n; ii++) {
                final long aBase = (long) ii * aRowBytes;
                final long cBase = (long) ii * cRowBytes;
                for (int j = 0; j < safeColEnd; j += nr) {
                    var z = FloatVector.zero(SPECIES_F32);
                    FloatVector acc0 = z, acc1 = z;
                    long aOff = aBase;
                    long bOff = (long) j * 4L;
                    for (int k = 0; k < m; k++) {
                        var b0 = FloatVector.fromMemorySegment(SPECIES_F32, memB, bOff, NATIVE);
                        var b1 = FloatVector.fromMemorySegment(SPECIES_F32, memB, bOff + vecBytes, NATIVE);
                        var a0 = FloatVector.broadcast(SPECIES_F32, memA.get(VF, aOff));
                        acc0 = a0.fma(b0, acc0);
                        acc1 = a0.fma(b1, acc1);
                        aOff += aColBytes;
                        bOff += bRowBytes;
                    }
                    long cRow = cBase + (long) j * 4L;
                    acc0.intoMemorySegment(memC, cRow, NATIVE);
                    acc1.intoMemorySegment(memC, cRow + vecBytes, NATIVE);
                }
                if (remCols > 0) {
                    var z = FloatVector.zero(SPECIES_F32);
                    FloatVector acc0 = z, acc1 = z;
                    long aOff = aBase;
                    long bOff = (long) safeColEnd * 4L;
                    for (int k = 0; k < m; k++) {
                        var b0 = FloatVector.fromMemorySegment(SPECIES_F32, memB, bOff, NATIVE, mask0);
                        var b1 = (lanes1 > 0) ? FloatVector.fromMemorySegment(SPECIES_F32, memB, bOff + vecBytes, NATIVE, mask1) : z;
                        var a0 = FloatVector.broadcast(SPECIES_F32, memA.get(VF, aOff));
                        acc0 = a0.fma(b0, acc0);
                        acc1 = a0.fma(b1, acc1);
                        aOff += aColBytes;
                        bOff += bRowBytes;
                    }
                    long cRow = cBase + (long) safeColEnd * 4L;
                    acc0.intoMemorySegment(memC, cRow, NATIVE, mask0);
                    if (lanes1 > 0) {
                        acc1.intoMemorySegment(memC, cRow + vecBytes, NATIVE, mask1);
                    }
                }
            }
        } else {
            final long bColBytes = b_s1 * 4L;
            final long cColBytes = c_s1 * 4L;
            for (int i = 0; i < n; i++) {
                long aRowBase = (long) i * aRowBytes;
                long cRowBase = (long) i * cRowBytes;
                for (int j = 0; j < p; j++) {
                    float sum = 0.0f;
                    long aOff = aRowBase;
                    long bOff = (long) j * bColBytes;
                    for (int k = 0; k < m; k++) {
                        sum = Math.fma(memA.get(VF, aOff), memB.get(VF, bOff), sum);
                        aOff += aColBytes;
                        bOff += bRowBytes;
                    }
                    memC.set(VF, cRowBase + (long) j * cColBytes, sum);
                }
            }
        }
    }

    // =========================================================================
    // 2. DOUBLE DIRECT TILED (4 x 2V register blocking + SIMD tail cleanup)
    // =========================================================================

    public static void matmulDouble(NDArray a, NDArray b, NDArray resArray, int n, int m, int p) {
        long[] aStrides = a.internalStridesUnsafe();
        long[] bStrides = b.internalStridesUnsafe();
        long[] cStrides = resArray.internalStridesUnsafe();
        final long a_s0 = aStrides[0], a_s1 = aStrides[1];
        final long b_s0 = bStrides[0], b_s1 = bStrides[1];
        final long c_s0 = cStrides[0], c_s1 = cStrides[1];

        final MemorySegment memA = a.getData();
        final MemorySegment memB = b.getData();
        final MemorySegment memC = resArray.getData();

        final int nr = NR_F64;
        final long vecBytes = VEC_BYTES_F64;
        final long aRowBytes = a_s0 * 8L;
        final long aColBytes = a_s1 * 8L;
        final long bRowBytes = b_s0 * 8L;
        final long cRowBytes = c_s0 * 8L;

        if (b_s1 == 1L && c_s1 == 1L) {
            final int safeRowEnd = n & ~3;
            final int safeColEnd = p - (p % nr);
            final int remCols = p - safeColEnd;
            final int lanes0 = Math.min(VL_F64, remCols);
            final int lanes1 = Math.max(0, remCols - VL_F64);
            final VectorMask<Double> mask0 = (remCols > 0) ? SPECIES_F64.indexInRange(0, lanes0) : null;
            final VectorMask<Double> mask1 = (lanes1 > 0) ? SPECIES_F64.indexInRange(0, lanes1) : null;

            for (int i = 0; i < safeRowEnd; i += 4) {
                final long aBase0 = (long) i * aRowBytes;
                final long aBase1 = aBase0 + aRowBytes;
                final long aBase2 = aBase1 + aRowBytes;
                final long aBase3 = aBase2 + aRowBytes;
                final long cBase0 = (long) i * cRowBytes;

                for (int j = 0; j < safeColEnd; j += nr) {
                    var z = DoubleVector.zero(SPECIES_F64);
                    DoubleVector acc00 = z, acc01 = z, acc10 = z, acc11 = z;
                    DoubleVector acc20 = z, acc21 = z, acc30 = z, acc31 = z;

                    long aOff0 = aBase0, aOff1 = aBase1, aOff2 = aBase2, aOff3 = aBase3;
                    long bOff = (long) j * 8L;
                    int k = 0;
                    for (; k <= m - 2; k += 2) {
                        var b0 = DoubleVector.fromMemorySegment(SPECIES_F64, memB, bOff, NATIVE);
                        var b1 = DoubleVector.fromMemorySegment(SPECIES_F64, memB, bOff + vecBytes, NATIVE);

                        var a0 = DoubleVector.broadcast(SPECIES_F64, memA.get(VD, aOff0));
                        var a1 = DoubleVector.broadcast(SPECIES_F64, memA.get(VD, aOff1));
                        acc00 = a0.fma(b0, acc00); acc01 = a0.fma(b1, acc01);
                        acc10 = a1.fma(b0, acc10); acc11 = a1.fma(b1, acc11);

                        a0 = DoubleVector.broadcast(SPECIES_F64, memA.get(VD, aOff2));
                        a1 = DoubleVector.broadcast(SPECIES_F64, memA.get(VD, aOff3));
                        acc20 = a0.fma(b0, acc20); acc21 = a0.fma(b1, acc21);
                        acc30 = a1.fma(b0, acc30); acc31 = a1.fma(b1, acc31);

                        long bOff1 = bOff + bRowBytes;
                        b0 = DoubleVector.fromMemorySegment(SPECIES_F64, memB, bOff1, NATIVE);
                        b1 = DoubleVector.fromMemorySegment(SPECIES_F64, memB, bOff1 + vecBytes, NATIVE);

                        a0 = DoubleVector.broadcast(SPECIES_F64, memA.get(VD, aOff0 + aColBytes));
                        a1 = DoubleVector.broadcast(SPECIES_F64, memA.get(VD, aOff1 + aColBytes));
                        acc00 = a0.fma(b0, acc00); acc01 = a0.fma(b1, acc01);
                        acc10 = a1.fma(b0, acc10); acc11 = a1.fma(b1, acc11);

                        a0 = DoubleVector.broadcast(SPECIES_F64, memA.get(VD, aOff2 + aColBytes));
                        a1 = DoubleVector.broadcast(SPECIES_F64, memA.get(VD, aOff3 + aColBytes));
                        acc20 = a0.fma(b0, acc20); acc21 = a0.fma(b1, acc21);
                        acc30 = a1.fma(b0, acc30); acc31 = a1.fma(b1, acc31);

                        long aStep2 = aColBytes * 2L;
                        aOff0 += aStep2; aOff1 += aStep2; aOff2 += aStep2; aOff3 += aStep2;
                        bOff = bOff1 + bRowBytes;
                    }
                    for (; k < m; k++) {
                        var b0 = DoubleVector.fromMemorySegment(SPECIES_F64, memB, bOff, NATIVE);
                        var b1 = DoubleVector.fromMemorySegment(SPECIES_F64, memB, bOff + vecBytes, NATIVE);

                        var a0 = DoubleVector.broadcast(SPECIES_F64, memA.get(VD, aOff0));
                        var a1 = DoubleVector.broadcast(SPECIES_F64, memA.get(VD, aOff1));
                        acc00 = a0.fma(b0, acc00); acc01 = a0.fma(b1, acc01);
                        acc10 = a1.fma(b0, acc10); acc11 = a1.fma(b1, acc11);

                        a0 = DoubleVector.broadcast(SPECIES_F64, memA.get(VD, aOff2));
                        a1 = DoubleVector.broadcast(SPECIES_F64, memA.get(VD, aOff3));
                        acc20 = a0.fma(b0, acc20); acc21 = a0.fma(b1, acc21);
                        acc30 = a1.fma(b0, acc30); acc31 = a1.fma(b1, acc31);

                        aOff0 += aColBytes; aOff1 += aColBytes; aOff2 += aColBytes; aOff3 += aColBytes;
                        bOff += bRowBytes;
                    }

                    long cRow0 = cBase0 + (long) j * 8L;
                    long cRow1 = cRow0 + cRowBytes;
                    long cRow2 = cRow1 + cRowBytes;
                    long cRow3 = cRow2 + cRowBytes;
                    acc00.intoMemorySegment(memC, cRow0, NATIVE); acc01.intoMemorySegment(memC, cRow0 + vecBytes, NATIVE);
                    acc10.intoMemorySegment(memC, cRow1, NATIVE); acc11.intoMemorySegment(memC, cRow1 + vecBytes, NATIVE);
                    acc20.intoMemorySegment(memC, cRow2, NATIVE); acc21.intoMemorySegment(memC, cRow2 + vecBytes, NATIVE);
                    acc30.intoMemorySegment(memC, cRow3, NATIVE); acc31.intoMemorySegment(memC, cRow3 + vecBytes, NATIVE);
                }

                if (remCols > 0) {
                    var z = DoubleVector.zero(SPECIES_F64);
                    DoubleVector acc00 = z, acc01 = z, acc10 = z, acc11 = z;
                    DoubleVector acc20 = z, acc21 = z, acc30 = z, acc31 = z;

                    long aOff0 = aBase0, aOff1 = aBase1, aOff2 = aBase2, aOff3 = aBase3;
                    long bOff = (long) safeColEnd * 8L;
                    for (int k = 0; k < m; k++) {
                        var b0 = DoubleVector.fromMemorySegment(SPECIES_F64, memB, bOff, NATIVE, mask0);
                        var b1 = (lanes1 > 0) ? DoubleVector.fromMemorySegment(SPECIES_F64, memB, bOff + vecBytes, NATIVE, mask1) : z;

                        var a0 = DoubleVector.broadcast(SPECIES_F64, memA.get(VD, aOff0));
                        var a1 = DoubleVector.broadcast(SPECIES_F64, memA.get(VD, aOff1));
                        acc00 = a0.fma(b0, acc00); acc01 = a0.fma(b1, acc01);
                        acc10 = a1.fma(b0, acc10); acc11 = a1.fma(b1, acc11);

                        a0 = DoubleVector.broadcast(SPECIES_F64, memA.get(VD, aOff2));
                        a1 = DoubleVector.broadcast(SPECIES_F64, memA.get(VD, aOff3));
                        acc20 = a0.fma(b0, acc20); acc21 = a0.fma(b1, acc21);
                        acc30 = a1.fma(b0, acc30); acc31 = a1.fma(b1, acc31);

                        aOff0 += aColBytes; aOff1 += aColBytes; aOff2 += aColBytes; aOff3 += aColBytes;
                        bOff += bRowBytes;
                    }

                    long cRow0 = cBase0 + (long) safeColEnd * 8L;
                    long cRow1 = cRow0 + cRowBytes;
                    long cRow2 = cRow1 + cRowBytes;
                    long cRow3 = cRow2 + cRowBytes;
                    acc00.intoMemorySegment(memC, cRow0, NATIVE, mask0);
                    acc10.intoMemorySegment(memC, cRow1, NATIVE, mask0);
                    acc20.intoMemorySegment(memC, cRow2, NATIVE, mask0);
                    acc30.intoMemorySegment(memC, cRow3, NATIVE, mask0);
                    if (lanes1 > 0) {
                        acc01.intoMemorySegment(memC, cRow0 + vecBytes, NATIVE, mask1);
                        acc11.intoMemorySegment(memC, cRow1 + vecBytes, NATIVE, mask1);
                        acc21.intoMemorySegment(memC, cRow2 + vecBytes, NATIVE, mask1);
                        acc31.intoMemorySegment(memC, cRow3 + vecBytes, NATIVE, mask1);
                    }
                }
            }

            for (int ii = safeRowEnd; ii < n; ii++) {
                final long aBase = (long) ii * aRowBytes;
                final long cBase = (long) ii * cRowBytes;
                for (int j = 0; j < safeColEnd; j += nr) {
                    var z = DoubleVector.zero(SPECIES_F64);
                    DoubleVector acc0 = z, acc1 = z;
                    long aOff = aBase;
                    long bOff = (long) j * 8L;
                    for (int k = 0; k < m; k++) {
                        var b0 = DoubleVector.fromMemorySegment(SPECIES_F64, memB, bOff, NATIVE);
                        var b1 = DoubleVector.fromMemorySegment(SPECIES_F64, memB, bOff + vecBytes, NATIVE);
                        var a0 = DoubleVector.broadcast(SPECIES_F64, memA.get(VD, aOff));
                        acc0 = a0.fma(b0, acc0);
                        acc1 = a0.fma(b1, acc1);
                        aOff += aColBytes;
                        bOff += bRowBytes;
                    }
                    long cRow = cBase + (long) j * 8L;
                    acc0.intoMemorySegment(memC, cRow, NATIVE);
                    acc1.intoMemorySegment(memC, cRow + vecBytes, NATIVE);
                }
                if (remCols > 0) {
                    var z = DoubleVector.zero(SPECIES_F64);
                    DoubleVector acc0 = z, acc1 = z;
                    long aOff = aBase;
                    long bOff = (long) safeColEnd * 8L;
                    for (int k = 0; k < m; k++) {
                        var b0 = DoubleVector.fromMemorySegment(SPECIES_F64, memB, bOff, NATIVE, mask0);
                        var b1 = (lanes1 > 0) ? DoubleVector.fromMemorySegment(SPECIES_F64, memB, bOff + vecBytes, NATIVE, mask1) : z;
                        var a0 = DoubleVector.broadcast(SPECIES_F64, memA.get(VD, aOff));
                        acc0 = a0.fma(b0, acc0);
                        acc1 = a0.fma(b1, acc1);
                        aOff += aColBytes;
                        bOff += bRowBytes;
                    }
                    long cRow = cBase + (long) safeColEnd * 8L;
                    acc0.intoMemorySegment(memC, cRow, NATIVE, mask0);
                    if (lanes1 > 0) {
                        acc1.intoMemorySegment(memC, cRow + vecBytes, NATIVE, mask1);
                    }
                }
            }
        } else {
            final long bColBytes = b_s1 * 8L;
            final long cColBytes = c_s1 * 8L;
            for (int i = 0; i < n; i++) {
                long aRowBase = (long) i * aRowBytes;
                long cRowBase = (long) i * cRowBytes;
                for (int j = 0; j < p; j++) {
                    double sum = 0.0d;
                    long aOff = aRowBase;
                    long bOff = (long) j * bColBytes;
                    for (int k = 0; k < m; k++) {
                        sum = Math.fma(memA.get(VD, aOff), memB.get(VD, bOff), sum);
                        aOff += aColBytes;
                        bOff += bRowBytes;
                    }
                    memC.set(VD, cRowBase + (long) j * cColBytes, sum);
                }
            }
        }
    }

    // =========================================================================
    // 3. INT DIRECT TILED (4 x 2V register blocking + SIMD tail cleanup)
    // =========================================================================

    public static void matmulInt(NDArray a, NDArray b, NDArray resArray, int n, int m, int p) {
        long[] aStrides = a.internalStridesUnsafe();
        long[] bStrides = b.internalStridesUnsafe();
        long[] cStrides = resArray.internalStridesUnsafe();
        final long a_s0 = aStrides[0], a_s1 = aStrides[1];
        final long b_s0 = bStrides[0], b_s1 = bStrides[1];
        final long c_s0 = cStrides[0], c_s1 = cStrides[1];

        final MemorySegment memA = a.getData();
        final MemorySegment memB = b.getData();
        final MemorySegment memC = resArray.getData();

        final int nr = NR_I32;
        final long vecBytes = VEC_BYTES_I32;
        final long aRowBytes = a_s0 * 4L;
        final long aColBytes = a_s1 * 4L;
        final long bRowBytes = b_s0 * 4L;
        final long cRowBytes = c_s0 * 4L;

        if (b_s1 == 1L && c_s1 == 1L) {
            final int safeRowEnd = n & ~3;
            final int safeColEnd = p - (p % nr);
            final int remCols = p - safeColEnd;
            final int lanes0 = Math.min(VL_I32, remCols);
            final int lanes1 = Math.max(0, remCols - VL_I32);
            final VectorMask<Integer> mask0 = (remCols > 0) ? SPECIES_I32.indexInRange(0, lanes0) : null;
            final VectorMask<Integer> mask1 = (lanes1 > 0) ? SPECIES_I32.indexInRange(0, lanes1) : null;

            for (int i = 0; i < safeRowEnd; i += 4) {
                final long aBase0 = (long) i * aRowBytes;
                final long aBase1 = aBase0 + aRowBytes;
                final long aBase2 = aBase1 + aRowBytes;
                final long aBase3 = aBase2 + aRowBytes;
                final long cBase0 = (long) i * cRowBytes;

                for (int j = 0; j < safeColEnd; j += nr) {
                    var z = IntVector.zero(SPECIES_I32);
                    IntVector acc00 = z, acc01 = z, acc10 = z, acc11 = z;
                    IntVector acc20 = z, acc21 = z, acc30 = z, acc31 = z;

                    long aOff0 = aBase0, aOff1 = aBase1, aOff2 = aBase2, aOff3 = aBase3;
                    long bOff = (long) j * 4L;
                    int k = 0;
                    for (; k <= m - 2; k += 2) {
                        var b0 = IntVector.fromMemorySegment(SPECIES_I32, memB, bOff, NATIVE);
                        var b1 = IntVector.fromMemorySegment(SPECIES_I32, memB, bOff + vecBytes, NATIVE);

                        var a0 = IntVector.broadcast(SPECIES_I32, memA.get(VI, aOff0));
                        var a1 = IntVector.broadcast(SPECIES_I32, memA.get(VI, aOff1));
                        acc00 = a0.mul(b0).add(acc00); acc01 = a0.mul(b1).add(acc01);
                        acc10 = a1.mul(b0).add(acc10); acc11 = a1.mul(b1).add(acc11);

                        a0 = IntVector.broadcast(SPECIES_I32, memA.get(VI, aOff2));
                        a1 = IntVector.broadcast(SPECIES_I32, memA.get(VI, aOff3));
                        acc20 = a0.mul(b0).add(acc20); acc21 = a0.mul(b1).add(acc21);
                        acc30 = a1.mul(b0).add(acc30); acc31 = a1.mul(b1).add(acc31);

                        long bOff1 = bOff + bRowBytes;
                        b0 = IntVector.fromMemorySegment(SPECIES_I32, memB, bOff1, NATIVE);
                        b1 = IntVector.fromMemorySegment(SPECIES_I32, memB, bOff1 + vecBytes, NATIVE);

                        a0 = IntVector.broadcast(SPECIES_I32, memA.get(VI, aOff0 + aColBytes));
                        a1 = IntVector.broadcast(SPECIES_I32, memA.get(VI, aOff1 + aColBytes));
                        acc00 = a0.mul(b0).add(acc00); acc01 = a0.mul(b1).add(acc01);
                        acc10 = a1.mul(b0).add(acc10); acc11 = a1.mul(b1).add(acc11);

                        a0 = IntVector.broadcast(SPECIES_I32, memA.get(VI, aOff2 + aColBytes));
                        a1 = IntVector.broadcast(SPECIES_I32, memA.get(VI, aOff3 + aColBytes));
                        acc20 = a0.mul(b0).add(acc20); acc21 = a0.mul(b1).add(acc21);
                        acc30 = a1.mul(b0).add(acc30); acc31 = a1.mul(b1).add(acc31);

                        long aStep2 = aColBytes * 2L;
                        aOff0 += aStep2; aOff1 += aStep2; aOff2 += aStep2; aOff3 += aStep2;
                        bOff = bOff1 + bRowBytes;
                    }
                    for (; k < m; k++) {
                        var b0 = IntVector.fromMemorySegment(SPECIES_I32, memB, bOff, NATIVE);
                        var b1 = IntVector.fromMemorySegment(SPECIES_I32, memB, bOff + vecBytes, NATIVE);

                        var a0 = IntVector.broadcast(SPECIES_I32, memA.get(VI, aOff0));
                        var a1 = IntVector.broadcast(SPECIES_I32, memA.get(VI, aOff1));
                        acc00 = a0.mul(b0).add(acc00); acc01 = a0.mul(b1).add(acc01);
                        acc10 = a1.mul(b0).add(acc10); acc11 = a1.mul(b1).add(acc11);

                        a0 = IntVector.broadcast(SPECIES_I32, memA.get(VI, aOff2));
                        a1 = IntVector.broadcast(SPECIES_I32, memA.get(VI, aOff3));
                        acc20 = a0.mul(b0).add(acc20); acc21 = a0.mul(b1).add(acc21);
                        acc30 = a1.mul(b0).add(acc30); acc31 = a1.mul(b1).add(acc31);

                        aOff0 += aColBytes; aOff1 += aColBytes; aOff2 += aColBytes; aOff3 += aColBytes;
                        bOff += bRowBytes;
                    }

                    long cRow0 = cBase0 + (long) j * 4L;
                    long cRow1 = cRow0 + cRowBytes;
                    long cRow2 = cRow1 + cRowBytes;
                    long cRow3 = cRow2 + cRowBytes;
                    acc00.intoMemorySegment(memC, cRow0, NATIVE); acc01.intoMemorySegment(memC, cRow0 + vecBytes, NATIVE);
                    acc10.intoMemorySegment(memC, cRow1, NATIVE); acc11.intoMemorySegment(memC, cRow1 + vecBytes, NATIVE);
                    acc20.intoMemorySegment(memC, cRow2, NATIVE); acc21.intoMemorySegment(memC, cRow2 + vecBytes, NATIVE);
                    acc30.intoMemorySegment(memC, cRow3, NATIVE); acc31.intoMemorySegment(memC, cRow3 + vecBytes, NATIVE);
                }

                if (remCols > 0) {
                    var z = IntVector.zero(SPECIES_I32);
                    IntVector acc00 = z, acc01 = z, acc10 = z, acc11 = z;
                    IntVector acc20 = z, acc21 = z, acc30 = z, acc31 = z;

                    long aOff0 = aBase0, aOff1 = aBase1, aOff2 = aBase2, aOff3 = aBase3;
                    long bOff = (long) safeColEnd * 4L;
                    for (int k = 0; k < m; k++) {
                        var b0 = IntVector.fromMemorySegment(SPECIES_I32, memB, bOff, NATIVE, mask0);
                        var b1 = (lanes1 > 0) ? IntVector.fromMemorySegment(SPECIES_I32, memB, bOff + vecBytes, NATIVE, mask1) : z;

                        var a0 = IntVector.broadcast(SPECIES_I32, memA.get(VI, aOff0));
                        var a1 = IntVector.broadcast(SPECIES_I32, memA.get(VI, aOff1));
                        acc00 = a0.mul(b0).add(acc00); acc01 = a0.mul(b1).add(acc01);
                        acc10 = a1.mul(b0).add(acc10); acc11 = a1.mul(b1).add(acc11);

                        a0 = IntVector.broadcast(SPECIES_I32, memA.get(VI, aOff2));
                        a1 = IntVector.broadcast(SPECIES_I32, memA.get(VI, aOff3));
                        acc20 = a0.mul(b0).add(acc20); acc21 = a0.mul(b1).add(acc21);
                        acc30 = a1.mul(b0).add(acc30); acc31 = a1.mul(b1).add(acc31);

                        aOff0 += aColBytes; aOff1 += aColBytes; aOff2 += aColBytes; aOff3 += aColBytes;
                        bOff += bRowBytes;
                    }

                    long cRow0 = cBase0 + (long) safeColEnd * 4L;
                    long cRow1 = cRow0 + cRowBytes;
                    long cRow2 = cRow1 + cRowBytes;
                    long cRow3 = cRow2 + cRowBytes;
                    acc00.intoMemorySegment(memC, cRow0, NATIVE, mask0);
                    acc10.intoMemorySegment(memC, cRow1, NATIVE, mask0);
                    acc20.intoMemorySegment(memC, cRow2, NATIVE, mask0);
                    acc30.intoMemorySegment(memC, cRow3, NATIVE, mask0);
                    if (lanes1 > 0) {
                        acc01.intoMemorySegment(memC, cRow0 + vecBytes, NATIVE, mask1);
                        acc11.intoMemorySegment(memC, cRow1 + vecBytes, NATIVE, mask1);
                        acc21.intoMemorySegment(memC, cRow2 + vecBytes, NATIVE, mask1);
                        acc31.intoMemorySegment(memC, cRow3 + vecBytes, NATIVE, mask1);
                    }
                }
            }

            for (int ii = safeRowEnd; ii < n; ii++) {
                final long aBase = (long) ii * aRowBytes;
                final long cBase = (long) ii * cRowBytes;
                for (int j = 0; j < safeColEnd; j += nr) {
                    var z = IntVector.zero(SPECIES_I32);
                    IntVector acc0 = z, acc1 = z;
                    long aOff = aBase;
                    long bOff = (long) j * 4L;
                    for (int k = 0; k < m; k++) {
                        var b0 = IntVector.fromMemorySegment(SPECIES_I32, memB, bOff, NATIVE);
                        var b1 = IntVector.fromMemorySegment(SPECIES_I32, memB, bOff + vecBytes, NATIVE);
                        var a0 = IntVector.broadcast(SPECIES_I32, memA.get(VI, aOff));
                        acc0 = a0.mul(b0).add(acc0);
                        acc1 = a0.mul(b1).add(acc1);
                        aOff += aColBytes;
                        bOff += bRowBytes;
                    }
                    long cRow = cBase + (long) j * 4L;
                    acc0.intoMemorySegment(memC, cRow, NATIVE);
                    acc1.intoMemorySegment(memC, cRow + vecBytes, NATIVE);
                }
                if (remCols > 0) {
                    var z = IntVector.zero(SPECIES_I32);
                    IntVector acc0 = z, acc1 = z;
                    long aOff = aBase;
                    long bOff = (long) safeColEnd * 4L;
                    for (int k = 0; k < m; k++) {
                        var b0 = IntVector.fromMemorySegment(SPECIES_I32, memB, bOff, NATIVE, mask0);
                        var b1 = (lanes1 > 0) ? IntVector.fromMemorySegment(SPECIES_I32, memB, bOff + vecBytes, NATIVE, mask1) : z;
                        var a0 = IntVector.broadcast(SPECIES_I32, memA.get(VI, aOff));
                        acc0 = a0.mul(b0).add(acc0);
                        acc1 = a0.mul(b1).add(acc1);
                        aOff += aColBytes;
                        bOff += bRowBytes;
                    }
                    long cRow = cBase + (long) safeColEnd * 4L;
                    acc0.intoMemorySegment(memC, cRow, NATIVE, mask0);
                    if (lanes1 > 0) {
                        acc1.intoMemorySegment(memC, cRow + vecBytes, NATIVE, mask1);
                    }
                }
            }
        } else {
            final long bColBytes = b_s1 * 4L;
            final long cColBytes = c_s1 * 4L;
            for (int i = 0; i < n; i++) {
                long aRowBase = (long) i * aRowBytes;
                long cRowBase = (long) i * cRowBytes;
                for (int j = 0; j < p; j++) {
                    int sum = 0;
                    long aOff = aRowBase;
                    long bOff = (long) j * bColBytes;
                    for (int k = 0; k < m; k++) {
                        sum += memA.get(VI, aOff) * memB.get(VI, bOff);
                        aOff += aColBytes;
                        bOff += bRowBytes;
                    }
                    memC.set(VI, cRowBase + (long) j * cColBytes, sum);
                }
            }
        }
    }
}