package jnum.jnumops;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import jnum.NDArray;
import jnum.jnumops.MatMulOps.AVX2Double;
import jnum.jnumops.MatMulOps.AVX2Float;
import jnum.jnumops.MatMulOps.AVX2Int;
import jnum.jnumops.MatMulOps.FJ_Packed_Double;
import jnum.jnumops.MatMulOps.FJ_Packed_Float;
import jnum.jnumops.MatMulOps.FJ_Packed_Int;

public class MatMulOps {

    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Integer> SPECIESINT = IntVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Double> SPECIESDB = DoubleVector.SPECIES_PREFERRED;
    
    private static final ForkJoinPool POOL = ForkJoinPool.commonPool();
    private static final int THRESHOLD = 64;

    //Non-instantiable utility class
    private MatMulOps() {
        throw new AssertionError();
    }


    private static final int BLOCK_SIZE_Float = 64;
    private static final ThreadLocal<MemorySegment> threadLocalA_Float = ThreadLocal.withInitial(() -> Arena.ofAuto().allocate((long) BLOCK_SIZE_Float * BLOCK_SIZE_Float * 4L));
    private static final ThreadLocal<MemorySegment> threadLocalB_Float = ThreadLocal.withInitial(() -> Arena.ofAuto().allocate((long) BLOCK_SIZE_Float * BLOCK_SIZE_Float * 4L));
    private static final ThreadLocal<MemorySegment> threadLocalC_Float = ThreadLocal.withInitial(() -> Arena.ofAuto().allocate((long) BLOCK_SIZE_Float * BLOCK_SIZE_Float * 4L));

    public static NDArray matmulFloat(NDArray a, NDArray b, NDArray resArray) {
        int n = a.shape[0]; int m = a.shape[1]; int p = b.shape[1];
        try(Arena arena = Arena.ofShared()) {
            MemorySegment memA = a.isContiguous() ? a.data : a.contiguous(arena).data;
            MemorySegment memB_T = fastTranspose2D_Float(b.data, arena, m, p);
            MemorySegment memC = resArray.data;

            if (n >= 1024 || m >= 1024 || p >= 1024) {
                POOL.invoke(new FJ_Packed_Float(memA, memB_T, memC, n, m, p, 0, n));
            } else {
                POOL.invoke(new AVX2Float(memA, memB_T, memC, n, m, p, 0, n));
            }
        }
        return resArray;
    }

    private static void packBlock_Float(MemorySegment src, MemorySegment dest, int srcCols, int rowStart, int colStart, int maxRow, int maxCol) {
        for (int r = 0; r < BLOCK_SIZE_Float; r++) {
            if (rowStart + r < maxRow) {
                int validCols = Math.min(BLOCK_SIZE_Float, maxCol - colStart);
                MemorySegment.copy(src, ((long) (rowStart + r) * srcCols + colStart) * 4L, dest, (long) r * BLOCK_SIZE_Float * 4L, validCols * 4L);
                if (validCols < BLOCK_SIZE_Float) {
                    dest.asSlice(((long) r * BLOCK_SIZE_Float + validCols) * 4L, (BLOCK_SIZE_Float - validCols) * 4L).fill((byte) 0);
                }
            } else {
                dest.asSlice((long) r * BLOCK_SIZE_Float * 4L, BLOCK_SIZE_Float * 4L).fill((byte) 0);
            }
        }
    }

    private static void unpackBlock_Float(MemorySegment src, MemorySegment dest, int destCols, int rowStart, int colStart, int maxRow, int maxCol) {
        for (int r = 0; r < BLOCK_SIZE_Float; r++) {
            if (rowStart + r < maxRow) {
                int validCols = Math.min(BLOCK_SIZE_Float, maxCol - colStart);
                MemorySegment.copy(src, (long) r * BLOCK_SIZE_Float * 4L, dest, ((long) (rowStart + r) * destCols + colStart) * 4L, validCols * 4L);
            }
        }
    }

    public static void packedKernel2x2_Float(MemorySegment pA, MemorySegment pB, MemorySegment pC) {
        int loopBound = SPECIES.loopBound(BLOCK_SIZE_Float);
        for (int i = 0; i < BLOCK_SIZE_Float; i += 2) {
            for (int j = 0; j < BLOCK_SIZE_Float; j += 2) {
                var vSum00 = FloatVector.zero(SPECIES); var vSum01 = FloatVector.zero(SPECIES);
                var vSum10 = FloatVector.zero(SPECIES); var vSum11 = FloatVector.zero(SPECIES);
                int k = 0;
                for (; k < loopBound; k += SPECIES.length()) {
                    var vA0 = FloatVector.fromMemorySegment(SPECIES, pA, ((long) i * BLOCK_SIZE_Float + k) * 4L, ByteOrder.nativeOrder());
                    var vA1 = FloatVector.fromMemorySegment(SPECIES, pA, ((long) (i + 1) * BLOCK_SIZE_Float + k) * 4L, ByteOrder.nativeOrder());
                    var vB0 = FloatVector.fromMemorySegment(SPECIES, pB, ((long) j * BLOCK_SIZE_Float + k) * 4L, ByteOrder.nativeOrder());
                    var vB1 = FloatVector.fromMemorySegment(SPECIES, pB, ((long) (j + 1) * BLOCK_SIZE_Float + k) * 4L, ByteOrder.nativeOrder());
                    
                    vSum00 = vA0.fma(vB0, vSum00); vSum01 = vA0.fma(vB1, vSum01);
                    vSum10 = vA1.fma(vB0, vSum10); vSum11 = vA1.fma(vB1, vSum11);
                }
                float sum00 = vSum00.reduceLanes(VectorOperators.ADD); float sum01 = vSum01.reduceLanes(VectorOperators.ADD);
                float sum10 = vSum10.reduceLanes(VectorOperators.ADD); float sum11 = vSum11.reduceLanes(VectorOperators.ADD);
                for (; k < BLOCK_SIZE_Float; k++) {
                    float a0 = pA.getAtIndex(ValueLayout.JAVA_FLOAT, ((long) i * BLOCK_SIZE_Float + k));
                    float a1 = pA.getAtIndex(ValueLayout.JAVA_FLOAT, ((long) (i + 1) * BLOCK_SIZE_Float + k));
                    float b0 = pB.getAtIndex(ValueLayout.JAVA_FLOAT, ((long) j * BLOCK_SIZE_Float + k));
                    float b1 = pB.getAtIndex(ValueLayout.JAVA_FLOAT, ((long) (j + 1) * BLOCK_SIZE_Float + k));
                    sum00 += a0 * b0; sum01 += a0 * b1; sum10 += a1 * b0; sum11 += a1 * b1;
                }
                long idx00 = ((long) i * BLOCK_SIZE_Float + j); long idx01 = ((long) i * BLOCK_SIZE_Float + j + 1);
                long idx10 = ((long) (i + 1) * BLOCK_SIZE_Float + j); long idx11 = ((long) (i + 1) * BLOCK_SIZE_Float + j + 1);
                pC.setAtIndex(ValueLayout.JAVA_FLOAT, idx00, sum00 + pC.getAtIndex(ValueLayout.JAVA_FLOAT, idx00));
                pC.setAtIndex(ValueLayout.JAVA_FLOAT, idx01, sum01 + pC.getAtIndex(ValueLayout.JAVA_FLOAT, idx01));
                pC.setAtIndex(ValueLayout.JAVA_FLOAT, idx10, sum10 + pC.getAtIndex(ValueLayout.JAVA_FLOAT, idx10));
                pC.setAtIndex(ValueLayout.JAVA_FLOAT, idx11, sum11 + pC.getAtIndex(ValueLayout.JAVA_FLOAT, idx11));
            }
        }
    }

    static class FJ_Packed_Float extends RecursiveAction {
        MemorySegment A, B_T, C; int n, m, p, startRow, endRow;
        FJ_Packed_Float(MemorySegment A, MemorySegment B_T, MemorySegment C, int n, int m, int p, int startRow, int endRow) {
            this.A = A; this.B_T = B_T; this.C = C; this.n = n; this.m = m; this.p = p; this.startRow = startRow; this.endRow = endRow;
        }
        @Override
        protected void compute() {
            if (endRow - startRow <= Math.max(BLOCK_SIZE_Float, 128)) {
                MemorySegment pA = threadLocalA_Float.get(); MemorySegment pB = threadLocalB_Float.get(); MemorySegment pC = threadLocalC_Float.get();
                for (int iBlock = startRow; iBlock < endRow; iBlock += BLOCK_SIZE_Float) {
                    for (int jBlock = 0; jBlock < p; jBlock += BLOCK_SIZE_Float) {
                        pC.fill((byte) 0);
                        for (int kBlock = 0; kBlock < m; kBlock += BLOCK_SIZE_Float) {
                            packBlock_Float(A, pA, m, iBlock, kBlock, n, m);
                            packBlock_Float(B_T, pB, m, jBlock, kBlock, p, m);
                            packedKernel2x2_Float(pA, pB, pC);
                        }
                        unpackBlock_Float(pC, C, p, iBlock, jBlock, n, p); 
                    }
                }
            } else {
                int mid = startRow + (endRow - startRow) / 2;
                mid -= mid % BLOCK_SIZE_Float; 
                invokeAll(new FJ_Packed_Float(A, B_T, C, n, m, p, startRow, mid), new FJ_Packed_Float(A, B_T, C, n, m, p, mid, endRow));
            }
        }
    }

    static class AVX2Float extends RecursiveAction{
        MemorySegment A,B_T,C; int n,m,p,startRow,endRow;
        AVX2Float(MemorySegment A,MemorySegment B_T,MemorySegment C,int n,int m,int p,int startRow,int endRow){
            this.A = A; this.B_T = B_T; this.C = C; this.n = n; this.m = m; this.p = p; this.startRow = startRow; this.endRow = endRow;
        }
        @Override
        protected void compute(){
            if(endRow-startRow<=THRESHOLD){
                int safeRowEnd=endRow-((endRow-startRow)%2); int safeColEnd=p-(p%2);
                for(int i=startRow;i<safeRowEnd;i+=2){
                    for(int j=0;j<safeColEnd;j+=2){
                        hybridKernel2x2_Float(A, B_T, C, m, p, i, j);
                    }
                }
                if (safeRowEnd < endRow) {
                    for (int j = 0; j < safeColEnd; j++) { scalarDotProduct_Float(A, B_T, C, m, p, safeRowEnd, j); }
                }
                if (safeColEnd < p) {
                    for (int i = startRow; i < safeRowEnd; i++) { scalarDotProduct_Float(A, B_T, C, m, p, i, safeColEnd); }
                }
                if (safeRowEnd < endRow && safeColEnd < p) {
                    scalarDotProduct_Float(A, B_T, C, m, p, safeRowEnd, safeColEnd);
                }
            } else {
                int mid = startRow + (endRow - startRow) / 2;
                invokeAll(new AVX2Float(A, B_T, C, n, m, p, startRow, mid), new AVX2Float(A, B_T, C, n, m, p, mid, endRow));
            }
        }
    }

    private static void hybridKernel2x2_Float(MemorySegment A, MemorySegment B_T, MemorySegment C, int m, int p, int i, int j){
        var vSum00 = FloatVector.zero(SPECIES); var vSum01 = FloatVector.zero(SPECIES);
        var vSum10 = FloatVector.zero(SPECIES); var vSum11 = FloatVector.zero(SPECIES);
        long k = 0; long loopBound = SPECIES.loopBound(m);
        for(;k<loopBound;k+=SPECIES.length()){
            var vA0 = FloatVector.fromMemorySegment(SPECIES, A, ((long) (i) * m + k) * 4L, ByteOrder.nativeOrder());
            var vA1 = FloatVector.fromMemorySegment(SPECIES, A, ((long) (i + 1) * m + k) * 4L, ByteOrder.nativeOrder());
            var vB0 = FloatVector.fromMemorySegment(SPECIES, B_T, ((long) (j) * m + k) * 4L, ByteOrder.nativeOrder());
            var vB1 = FloatVector.fromMemorySegment(SPECIES, B_T, ((long) (j + 1) * m + k) * 4L, ByteOrder.nativeOrder());
            vSum00 = vA0.fma(vB0, vSum00); vSum01 = vA0.fma(vB1, vSum01);
            vSum10 = vA1.fma(vB0, vSum10); vSum11 = vA1.fma(vB1, vSum11);
        }
        float sum00 = vSum00.reduceLanes(VectorOperators.ADD); float sum01 = vSum01.reduceLanes(VectorOperators.ADD);
        float sum10 = vSum10.reduceLanes(VectorOperators.ADD); float sum11 = vSum11.reduceLanes(VectorOperators.ADD);
        for (; k < m; k++) {
            float a0 = A.getAtIndex(ValueLayout.JAVA_FLOAT, ((long) (i) * m + k)); float a1 = A.getAtIndex(ValueLayout.JAVA_FLOAT, ((long) (i + 1) * m + k));
            float b0 = B_T.getAtIndex(ValueLayout.JAVA_FLOAT, ((long) (j) * m + k)); float b1 = B_T.getAtIndex(ValueLayout.JAVA_FLOAT, ((long) (j + 1) * m + k));
            sum00 += a0 * b0; sum01 += a0 * b1; sum10 += a1 * b0; sum11 += a1 * b1;
        }
        C.setAtIndex(ValueLayout.JAVA_FLOAT, ((long) (i) * p + j), sum00); C.setAtIndex(ValueLayout.JAVA_FLOAT, ((long) (i) * p + j + 1), sum01);
        C.setAtIndex(ValueLayout.JAVA_FLOAT, ((long) (i + 1) * p + j), sum10); C.setAtIndex(ValueLayout.JAVA_FLOAT, ((long) (i + 1) * p + j + 1), sum11);
    }

    private static void scalarDotProduct_Float(MemorySegment A, MemorySegment B_T, MemorySegment C, int m, int p, int i, int j) {
        float sum = 0f;
        for (int k = 0; k < m; k++) {
            float a = A.getAtIndex(ValueLayout.JAVA_FLOAT, ((long) i * m + k));
            float b = B_T.getAtIndex(ValueLayout.JAVA_FLOAT, ((long) j * m + k));
            sum += a * b;
        }
        C.setAtIndex(ValueLayout.JAVA_FLOAT, ((long) i * p + j), sum);
    }

    private static MemorySegment fastTranspose2D_Float(MemorySegment src, Arena arena, int rows, int cols) {
        MemorySegment dst = arena.allocate((long) rows * cols * 4L);
        java.util.stream.IntStream.range(0, rows).parallel().forEach(i -> {
            for (int j = 0; j < cols; j++) {
                float val = src.getAtIndex(ValueLayout.JAVA_FLOAT, (long) i * cols + j);
                dst.setAtIndex(ValueLayout.JAVA_FLOAT, (long) j * rows + i, val);
            }
        });
        return dst;
    }


    private static final int BLOCK_SIZE_Double = 32;
    private static final ThreadLocal<MemorySegment> threadLocalA_Double = ThreadLocal.withInitial(() -> Arena.ofAuto().allocate((long) BLOCK_SIZE_Double * BLOCK_SIZE_Double * 8L));
    private static final ThreadLocal<MemorySegment> threadLocalB_Double = ThreadLocal.withInitial(() -> Arena.ofAuto().allocate((long) BLOCK_SIZE_Double * BLOCK_SIZE_Double * 8L));
    private static final ThreadLocal<MemorySegment> threadLocalC_Double = ThreadLocal.withInitial(() -> Arena.ofAuto().allocate((long) BLOCK_SIZE_Double * BLOCK_SIZE_Double * 8L));

    public static NDArray matmulDouble(NDArray a, NDArray b, NDArray resArray) {
        int n = a.shape[0]; int m = a.shape[1]; int p = b.shape[1];
        try(Arena arena = Arena.ofShared()) {
            MemorySegment memA = a.isContiguous() ? a.data : a.contiguous(arena).data;
            MemorySegment memB_T = fastTranspose2D_Double(b.data, arena, m, p);
            MemorySegment memC = resArray.data;

            if (n >= 1024 || m >= 1024 || p >= 1024) {
                POOL.invoke(new FJ_Packed_Double(memA, memB_T, memC, n, m, p, 0, n));
            } else {
                POOL.invoke(new AVX2Double(memA, memB_T, memC, n, m, p, 0, n));
            }
        }
        return resArray;
    }

    private static void packBlock_Double(MemorySegment src, MemorySegment dest, int srcCols, int rowStart, int colStart, int maxRow, int maxCol) {
        for (int r = 0; r < BLOCK_SIZE_Double; r++) {
            if (rowStart + r < maxRow) {
                int validCols = Math.min(BLOCK_SIZE_Double, maxCol - colStart);
                MemorySegment.copy(src, ((long) (rowStart + r) * srcCols + colStart) * 8L, dest, (long) r * BLOCK_SIZE_Double * 8L, validCols * 8L);
                if (validCols < BLOCK_SIZE_Double) {
                    dest.asSlice(((long) r * BLOCK_SIZE_Double + validCols) * 8L, (BLOCK_SIZE_Double - validCols) * 8L).fill((byte) 0);
                }
            } else {
                dest.asSlice((long) r * BLOCK_SIZE_Double * 8L, BLOCK_SIZE_Double * 8L).fill((byte) 0);
            }
        }
    }

    private static void unpackBlock_Double(MemorySegment src, MemorySegment dest, int destCols, int rowStart, int colStart, int maxRow, int maxCol) {
        for (int r = 0; r < BLOCK_SIZE_Double; r++) {
            if (rowStart + r < maxRow) {
                int validCols = Math.min(BLOCK_SIZE_Double, maxCol - colStart);
                MemorySegment.copy(src, (long) r * BLOCK_SIZE_Double * 8L, dest, ((long) (rowStart + r) * destCols + colStart) * 8L, validCols * 8L);
            }
        }
    }

    public static void packedKernel2x2_Double(MemorySegment pA, MemorySegment pB, MemorySegment pC) {
        int loopBound = SPECIESDB.loopBound(BLOCK_SIZE_Double);
        for (int i = 0; i < BLOCK_SIZE_Double; i += 2) {
            for (int j = 0; j < BLOCK_SIZE_Double; j += 2) {
                var vSum00 = DoubleVector.zero(SPECIESDB); var vSum01 = DoubleVector.zero(SPECIESDB);
                var vSum10 = DoubleVector.zero(SPECIESDB); var vSum11 = DoubleVector.zero(SPECIESDB);
                int k = 0;
                for (; k < loopBound; k += SPECIESDB.length()) {
                    var vA0 = DoubleVector.fromMemorySegment(SPECIESDB, pA, ((long) i * BLOCK_SIZE_Double + k) * 8L, ByteOrder.nativeOrder());
                    var vA1 = DoubleVector.fromMemorySegment(SPECIESDB, pA, ((long) (i + 1) * BLOCK_SIZE_Double + k) * 8L, ByteOrder.nativeOrder());
                    var vB0 = DoubleVector.fromMemorySegment(SPECIESDB, pB, ((long) j * BLOCK_SIZE_Double + k) * 8L, ByteOrder.nativeOrder());
                    var vB1 = DoubleVector.fromMemorySegment(SPECIESDB, pB, ((long) (j + 1) * BLOCK_SIZE_Double + k) * 8L, ByteOrder.nativeOrder());
                    
                    vSum00 = vA0.fma(vB0, vSum00); vSum01 = vA0.fma(vB1, vSum01);
                    vSum10 = vA1.fma(vB0, vSum10); vSum11 = vA1.fma(vB1, vSum11);
                }
                double sum00 = vSum00.reduceLanes(VectorOperators.ADD); double sum01 = vSum01.reduceLanes(VectorOperators.ADD);
                double sum10 = vSum10.reduceLanes(VectorOperators.ADD); double sum11 = vSum11.reduceLanes(VectorOperators.ADD);
                for (; k < BLOCK_SIZE_Double; k++) {
                    double a0 = pA.getAtIndex(ValueLayout.JAVA_DOUBLE, ((long) i * BLOCK_SIZE_Double + k));
                    double a1 = pA.getAtIndex(ValueLayout.JAVA_DOUBLE, ((long) (i + 1) * BLOCK_SIZE_Double + k));
                    double b0 = pB.getAtIndex(ValueLayout.JAVA_DOUBLE, ((long) j * BLOCK_SIZE_Double + k));
                    double b1 = pB.getAtIndex(ValueLayout.JAVA_DOUBLE, ((long) (j + 1) * BLOCK_SIZE_Double + k));
                    sum00 += a0 * b0; sum01 += a0 * b1; sum10 += a1 * b0; sum11 += a1 * b1;
                }
                long idx00 = ((long) i * BLOCK_SIZE_Double + j); long idx01 = ((long) i * BLOCK_SIZE_Double + j + 1);
                long idx10 = ((long) (i + 1) * BLOCK_SIZE_Double + j); long idx11 = ((long) (i + 1) * BLOCK_SIZE_Double + j + 1);
                pC.setAtIndex(ValueLayout.JAVA_DOUBLE, idx00, sum00 + pC.getAtIndex(ValueLayout.JAVA_DOUBLE, idx00));
                pC.setAtIndex(ValueLayout.JAVA_DOUBLE, idx01, sum01 + pC.getAtIndex(ValueLayout.JAVA_DOUBLE, idx01));
                pC.setAtIndex(ValueLayout.JAVA_DOUBLE, idx10, sum10 + pC.getAtIndex(ValueLayout.JAVA_DOUBLE, idx10));
                pC.setAtIndex(ValueLayout.JAVA_DOUBLE, idx11, sum11 + pC.getAtIndex(ValueLayout.JAVA_DOUBLE, idx11));
            }
        }
    }

    static class FJ_Packed_Double extends RecursiveAction {
        MemorySegment A, B_T, C; int n, m, p, startRow, endRow;
        FJ_Packed_Double(MemorySegment A, MemorySegment B_T, MemorySegment C, int n, int m, int p, int startRow, int endRow) {
            this.A = A; this.B_T = B_T; this.C = C; this.n = n; this.m = m; this.p = p; this.startRow = startRow; this.endRow = endRow;
        }
        @Override
        protected void compute() {
            if (endRow - startRow <= Math.max(BLOCK_SIZE_Double, 128)) {
                MemorySegment pA = threadLocalA_Double.get(); MemorySegment pB = threadLocalB_Double.get(); MemorySegment pC = threadLocalC_Double.get();
                for (int iBlock = startRow; iBlock < endRow; iBlock += BLOCK_SIZE_Double) {
                    for (int jBlock = 0; jBlock < p; jBlock += BLOCK_SIZE_Double) {
                        pC.fill((byte) 0);
                        for (int kBlock = 0; kBlock < m; kBlock += BLOCK_SIZE_Double) {
                            packBlock_Double(A, pA, m, iBlock, kBlock, n, m);
                            packBlock_Double(B_T, pB, m, jBlock, kBlock, p, m);
                            packedKernel2x2_Double(pA, pB, pC);
                        }
                        unpackBlock_Double(pC, C, p, iBlock, jBlock, n, p); 
                    }
                }
            } else {
                int mid = startRow + (endRow - startRow) / 2;
                mid -= mid % BLOCK_SIZE_Double; 
                invokeAll(new FJ_Packed_Double(A, B_T, C, n, m, p, startRow, mid), new FJ_Packed_Double(A, B_T, C, n, m, p, mid, endRow));
            }
        }
    }

    static class AVX2Double extends RecursiveAction{
        MemorySegment A,B_T,C; int n,m,p,startRow,endRow;
        AVX2Double(MemorySegment A,MemorySegment B_T,MemorySegment C,int n,int m,int p,int startRow,int endRow){
            this.A = A; this.B_T = B_T; this.C = C; this.n = n; this.m = m; this.p = p; this.startRow = startRow; this.endRow = endRow;
        }
        @Override
        protected void compute(){
            if(endRow-startRow<=THRESHOLD){
                int safeRowEnd=endRow-((endRow-startRow)%2); int safeColEnd=p-(p%2);
                for(int i=startRow;i<safeRowEnd;i+=2){
                    for(int j=0;j<safeColEnd;j+=2){
                        hybridKernel2x2_Double(A, B_T, C, m, p, i, j);
                    }
                }
                if (safeRowEnd < endRow) {
                    for (int j = 0; j < safeColEnd; j++) { scalarDotProduct_Double(A, B_T, C, m, p, safeRowEnd, j); }
                }
                if (safeColEnd < p) {
                    for (int i = startRow; i < safeRowEnd; i++) { scalarDotProduct_Double(A, B_T, C, m, p, i, safeColEnd); }
                }
                if (safeRowEnd < endRow && safeColEnd < p) {
                    scalarDotProduct_Double(A, B_T, C, m, p, safeRowEnd, safeColEnd);
                }
            } else {
                int mid = startRow + (endRow - startRow) / 2;
                invokeAll(new AVX2Double(A, B_T, C, n, m, p, startRow, mid), new AVX2Double(A, B_T, C, n, m, p, mid, endRow));
            }
        }
    }

    private static void hybridKernel2x2_Double(MemorySegment A, MemorySegment B_T, MemorySegment C, int m, int p, int i, int j){
        var vSum00 = DoubleVector.zero(SPECIESDB); var vSum01 = DoubleVector.zero(SPECIESDB);
        var vSum10 = DoubleVector.zero(SPECIESDB); var vSum11 = DoubleVector.zero(SPECIESDB);
        long k = 0; long loopBound = SPECIESDB.loopBound(m);
        for(;k<loopBound;k+=SPECIESDB.length()){
            var vA0 = DoubleVector.fromMemorySegment(SPECIESDB, A, ((long) (i) * m + k) * 8L, ByteOrder.nativeOrder());
            var vA1 = DoubleVector.fromMemorySegment(SPECIESDB, A, ((long) (i + 1) * m + k) * 8L, ByteOrder.nativeOrder());
            var vB0 = DoubleVector.fromMemorySegment(SPECIESDB, B_T, ((long) (j) * m + k) * 8L, ByteOrder.nativeOrder());
            var vB1 = DoubleVector.fromMemorySegment(SPECIESDB, B_T, ((long) (j + 1) * m + k) * 8L, ByteOrder.nativeOrder());
            vSum00 = vA0.fma(vB0, vSum00); vSum01 = vA0.fma(vB1, vSum01);
            vSum10 = vA1.fma(vB0, vSum10); vSum11 = vA1.fma(vB1, vSum11);
        }
        double sum00 = vSum00.reduceLanes(VectorOperators.ADD); double sum01 = vSum01.reduceLanes(VectorOperators.ADD);
        double sum10 = vSum10.reduceLanes(VectorOperators.ADD); double sum11 = vSum11.reduceLanes(VectorOperators.ADD);
        for (; k < m; k++) {
            double a0 = A.getAtIndex(ValueLayout.JAVA_DOUBLE, ((long) (i) * m + k)); double a1 = A.getAtIndex(ValueLayout.JAVA_DOUBLE, ((long) (i + 1) * m + k));
            double b0 = B_T.getAtIndex(ValueLayout.JAVA_DOUBLE, ((long) (j) * m + k)); double b1 = B_T.getAtIndex(ValueLayout.JAVA_DOUBLE, ((long) (j + 1) * m + k));
            sum00 += a0 * b0; sum01 += a0 * b1; sum10 += a1 * b0; sum11 += a1 * b1;
        }
        C.setAtIndex(ValueLayout.JAVA_DOUBLE, ((long) (i) * p + j), sum00); C.setAtIndex(ValueLayout.JAVA_DOUBLE, ((long) (i) * p + j + 1), sum01);
        C.setAtIndex(ValueLayout.JAVA_DOUBLE, ((long) (i + 1) * p + j), sum10); C.setAtIndex(ValueLayout.JAVA_DOUBLE, ((long) (i + 1) * p + j + 1), sum11);
    }

    private static void scalarDotProduct_Double(MemorySegment A, MemorySegment B_T, MemorySegment C, int m, int p, int i, int j) {
        double sum = 0.0;
        for (int k = 0; k < m; k++) {
            double a = A.getAtIndex(ValueLayout.JAVA_DOUBLE, ((long) i * m + k));
            double b = B_T.getAtIndex(ValueLayout.JAVA_DOUBLE, ((long) j * m + k));
            sum += a * b;
        }
        C.setAtIndex(ValueLayout.JAVA_DOUBLE, ((long) i * p + j), sum);
    }

    private static MemorySegment fastTranspose2D_Double(MemorySegment src, Arena arena, int rows, int cols) {
        MemorySegment dst = arena.allocate((long) rows * cols * 8L);
        java.util.stream.IntStream.range(0, rows).parallel().forEach(i -> {
            for (int j = 0; j < cols; j++) {
                double val = src.getAtIndex(ValueLayout.JAVA_DOUBLE, (long) i * cols + j);
                dst.setAtIndex(ValueLayout.JAVA_DOUBLE, (long) j * rows + i, val);
            }
        });
        return dst;
    }


    private static final int BLOCK_SIZE_Int = 64;
    private static final ThreadLocal<MemorySegment> threadLocalA_Int = ThreadLocal.withInitial(() -> Arena.ofAuto().allocate((long) BLOCK_SIZE_Int * BLOCK_SIZE_Int * 4L));
    private static final ThreadLocal<MemorySegment> threadLocalB_Int = ThreadLocal.withInitial(() -> Arena.ofAuto().allocate((long) BLOCK_SIZE_Int * BLOCK_SIZE_Int * 4L));
    private static final ThreadLocal<MemorySegment> threadLocalC_Int = ThreadLocal.withInitial(() -> Arena.ofAuto().allocate((long) BLOCK_SIZE_Int * BLOCK_SIZE_Int * 4L));

    public static NDArray matmulInt(NDArray a, NDArray b, NDArray resArray) {
        int n = a.shape[0]; int m = a.shape[1]; int p = b.shape[1];
        try(Arena arena = Arena.ofShared()) {
            MemorySegment memA = a.isContiguous() ? a.data : a.contiguous(arena).data;
            MemorySegment memB_T = fastTranspose2D_Int(b.data, arena, m, p);
            MemorySegment memC = resArray.data;

            if (n >= 1024 || m >= 1024 || p >= 1024) {
                POOL.invoke(new FJ_Packed_Int(memA, memB_T, memC, n, m, p, 0, n));
            } else {
                POOL.invoke(new AVX2Int(memA, memB_T, memC, n, m, p, 0, n));
            }
        }
        return resArray;
    }

    private static void packBlock_Int(MemorySegment src, MemorySegment dest, int srcCols, int rowStart, int colStart, int maxRow, int maxCol) {
        for (int r = 0; r < BLOCK_SIZE_Int; r++) {
            if (rowStart + r < maxRow) {
                int validCols = Math.min(BLOCK_SIZE_Int, maxCol - colStart);
                MemorySegment.copy(src, ((long) (rowStart + r) * srcCols + colStart) * 4L, dest, (long) r * BLOCK_SIZE_Int * 4L, validCols * 4L);
                if (validCols < BLOCK_SIZE_Int) {
                    dest.asSlice(((long) r * BLOCK_SIZE_Int + validCols) * 4L, (BLOCK_SIZE_Int - validCols) * 4L).fill((byte) 0);
                }
            } else {
                dest.asSlice((long) r * BLOCK_SIZE_Int * 4L, BLOCK_SIZE_Int * 4L).fill((byte) 0);
            }
        }
    }

    private static void unpackBlock_Int(MemorySegment src, MemorySegment dest, int destCols, int rowStart, int colStart, int maxRow, int maxCol) {
        for (int r = 0; r < BLOCK_SIZE_Int; r++) {
            if (rowStart + r < maxRow) {
                int validCols = Math.min(BLOCK_SIZE_Int, maxCol - colStart);
                MemorySegment.copy(src, (long) r * BLOCK_SIZE_Int * 4L, dest, ((long) (rowStart + r) * destCols + colStart) * 4L, validCols * 4L);
            }
        }
    }

    public static void packedKernel2x2_Int(MemorySegment pA, MemorySegment pB, MemorySegment pC) {
        int loopBound = SPECIESINT.loopBound(BLOCK_SIZE_Int);
        for (int i = 0; i < BLOCK_SIZE_Int; i += 2) {
            for (int j = 0; j < BLOCK_SIZE_Int; j += 2) {
                var vSum00 = IntVector.zero(SPECIESINT); var vSum01 = IntVector.zero(SPECIESINT);
                var vSum10 = IntVector.zero(SPECIESINT); var vSum11 = IntVector.zero(SPECIESINT);
                int k = 0;
                for (; k < loopBound; k += SPECIESINT.length()) {
                    var vA0 = IntVector.fromMemorySegment(SPECIESINT, pA, ((long) i * BLOCK_SIZE_Int + k) * 4L, ByteOrder.nativeOrder());
                    var vA1 = IntVector.fromMemorySegment(SPECIESINT, pA, ((long) (i + 1) * BLOCK_SIZE_Int + k) * 4L, ByteOrder.nativeOrder());
                    var vB0 = IntVector.fromMemorySegment(SPECIESINT, pB, ((long) j * BLOCK_SIZE_Int + k) * 4L, ByteOrder.nativeOrder());
                    var vB1 = IntVector.fromMemorySegment(SPECIESINT, pB, ((long) (j + 1) * BLOCK_SIZE_Int + k) * 4L, ByteOrder.nativeOrder());
                    
                    vSum00 = vSum00.add(vA0.mul(vB0)); vSum01 = vSum01.add(vA0.mul(vB1));
                    vSum10 = vSum10.add(vA1.mul(vB0)); vSum11 = vSum11.add(vA1.mul(vB1));
                }
                int sum00 = vSum00.reduceLanes(VectorOperators.ADD); int sum01 = vSum01.reduceLanes(VectorOperators.ADD);
                int sum10 = vSum10.reduceLanes(VectorOperators.ADD); int sum11 = vSum11.reduceLanes(VectorOperators.ADD);
                for (; k < BLOCK_SIZE_Int; k++) {
                    int a0 = pA.getAtIndex(ValueLayout.JAVA_INT, ((long) i * BLOCK_SIZE_Int + k));
                    int a1 = pA.getAtIndex(ValueLayout.JAVA_INT, ((long) (i + 1) * BLOCK_SIZE_Int + k));
                    int b0 = pB.getAtIndex(ValueLayout.JAVA_INT, ((long) j * BLOCK_SIZE_Int + k));
                    int b1 = pB.getAtIndex(ValueLayout.JAVA_INT, ((long) (j + 1) * BLOCK_SIZE_Int + k));
                    sum00 += a0 * b0; sum01 += a0 * b1; sum10 += a1 * b0; sum11 += a1 * b1;
                }
                long idx00 = ((long) i * BLOCK_SIZE_Int + j); long idx01 = ((long) i * BLOCK_SIZE_Int + j + 1);
                long idx10 = ((long) (i + 1) * BLOCK_SIZE_Int + j); long idx11 = ((long) (i + 1) * BLOCK_SIZE_Int + j + 1);
                pC.setAtIndex(ValueLayout.JAVA_INT, idx00, sum00 + pC.getAtIndex(ValueLayout.JAVA_INT, idx00));
                pC.setAtIndex(ValueLayout.JAVA_INT, idx01, sum01 + pC.getAtIndex(ValueLayout.JAVA_INT, idx01));
                pC.setAtIndex(ValueLayout.JAVA_INT, idx10, sum10 + pC.getAtIndex(ValueLayout.JAVA_INT, idx10));
                pC.setAtIndex(ValueLayout.JAVA_INT, idx11, sum11 + pC.getAtIndex(ValueLayout.JAVA_INT, idx11));
            }
        }
    }

    static class FJ_Packed_Int extends RecursiveAction {
        MemorySegment A, B_T, C; int n, m, p, startRow, endRow;
        FJ_Packed_Int(MemorySegment A, MemorySegment B_T, MemorySegment C, int n, int m, int p, int startRow, int endRow) {
            this.A = A; this.B_T = B_T; this.C = C; this.n = n; this.m = m; this.p = p; this.startRow = startRow; this.endRow = endRow;
        }
        @Override
        protected void compute() {
            if (endRow - startRow <= Math.max(BLOCK_SIZE_Int, 128)) {
                MemorySegment pA = threadLocalA_Int.get(); MemorySegment pB = threadLocalB_Int.get(); MemorySegment pC = threadLocalC_Int.get();
                for (int iBlock = startRow; iBlock < endRow; iBlock += BLOCK_SIZE_Int) {
                    for (int jBlock = 0; jBlock < p; jBlock += BLOCK_SIZE_Int) {
                        pC.fill((byte) 0);
                        for (int kBlock = 0; kBlock < m; kBlock += BLOCK_SIZE_Int) {
                            packBlock_Int(A, pA, m, iBlock, kBlock, n, m);
                            packBlock_Int(B_T, pB, m, jBlock, kBlock, p, m);
                            packedKernel2x2_Int(pA, pB, pC);
                        }
                        unpackBlock_Int(pC, C, p, iBlock, jBlock, n, p); 
                    }
                }
            } else {
                int mid = startRow + (endRow - startRow) / 2;
                mid -= mid % BLOCK_SIZE_Int; 
                invokeAll(new FJ_Packed_Int(A, B_T, C, n, m, p, startRow, mid), new FJ_Packed_Int(A, B_T, C, n, m, p, mid, endRow));
            }
        }
    }

    static class AVX2Int extends RecursiveAction{
        MemorySegment A,B_T,C; int n,m,p,startRow,endRow;
        AVX2Int(MemorySegment A,MemorySegment B_T,MemorySegment C,int n,int m,int p,int startRow,int endRow){
            this.A = A; this.B_T = B_T; this.C = C; this.n = n; this.m = m; this.p = p; this.startRow = startRow; this.endRow = endRow;
        }
        @Override
        protected void compute(){
            if(endRow-startRow<=THRESHOLD){
                int safeRowEnd=endRow-((endRow-startRow)%2); int safeColEnd=p-(p%2);
                for(int i=startRow;i<safeRowEnd;i+=2){
                    for(int j=0;j<safeColEnd;j+=2){
                        hybridKernel2x2_Int(A, B_T, C, m, p, i, j);
                    }
                }
                if (safeRowEnd < endRow) {
                    for (int j = 0; j < safeColEnd; j++) { scalarDotProduct_Int(A, B_T, C, m, p, safeRowEnd, j); }
                }
                if (safeColEnd < p) {
                    for (int i = startRow; i < safeRowEnd; i++) { scalarDotProduct_Int(A, B_T, C, m, p, i, safeColEnd); }
                }
                if (safeRowEnd < endRow && safeColEnd < p) {
                    scalarDotProduct_Int(A, B_T, C, m, p, safeRowEnd, safeColEnd);
                }
            } else {
                int mid = startRow + (endRow - startRow) / 2;
                invokeAll(new AVX2Int(A, B_T, C, n, m, p, startRow, mid), new AVX2Int(A, B_T, C, n, m, p, mid, endRow));
            }
        }
    }

    private static void hybridKernel2x2_Int(MemorySegment A, MemorySegment B_T, MemorySegment C, int m, int p, int i, int j){
        var vSum00 = IntVector.zero(SPECIESINT); var vSum01 = IntVector.zero(SPECIESINT);
        var vSum10 = IntVector.zero(SPECIESINT); var vSum11 = IntVector.zero(SPECIESINT);
        long k = 0; long loopBound = SPECIESINT.loopBound(m);
        for(;k<loopBound;k+=SPECIESINT.length()){
            var vA0 = IntVector.fromMemorySegment(SPECIESINT, A, ((long) (i) * m + k) * 4L, ByteOrder.nativeOrder());
            var vA1 = IntVector.fromMemorySegment(SPECIESINT, A, ((long) (i + 1) * m + k) * 4L, ByteOrder.nativeOrder());
            var vB0 = IntVector.fromMemorySegment(SPECIESINT, B_T, ((long) (j) * m + k) * 4L, ByteOrder.nativeOrder());
            var vB1 = IntVector.fromMemorySegment(SPECIESINT, B_T, ((long) (j + 1) * m + k) * 4L, ByteOrder.nativeOrder());
            vSum00 = vSum00.add(vA0.mul(vB0)); vSum01 = vSum01.add(vA0.mul(vB1));
            vSum10 = vSum10.add(vA1.mul(vB0)); vSum11 = vSum11.add(vA1.mul(vB1));
        }
        int sum00 = vSum00.reduceLanes(VectorOperators.ADD); int sum01 = vSum01.reduceLanes(VectorOperators.ADD);
        int sum10 = vSum10.reduceLanes(VectorOperators.ADD); int sum11 = vSum11.reduceLanes(VectorOperators.ADD);
        for (; k < m; k++) {
            int a0 = A.getAtIndex(ValueLayout.JAVA_INT, ((long) (i) * m + k)); int a1 = A.getAtIndex(ValueLayout.JAVA_INT, ((long) (i + 1) * m + k));
            int b0 = B_T.getAtIndex(ValueLayout.JAVA_INT, ((long) (j) * m + k)); int b1 = B_T.getAtIndex(ValueLayout.JAVA_INT, ((long) (j + 1) * m + k));
            sum00 += a0 * b0; sum01 += a0 * b1; sum10 += a1 * b0; sum11 += a1 * b1;
        }
        C.setAtIndex(ValueLayout.JAVA_INT, ((long) (i) * p + j), sum00); C.setAtIndex(ValueLayout.JAVA_INT, ((long) (i) * p + j + 1), sum01);
        C.setAtIndex(ValueLayout.JAVA_INT, ((long) (i + 1) * p + j), sum10); C.setAtIndex(ValueLayout.JAVA_INT, ((long) (i + 1) * p + j + 1), sum11);
    }

    private static void scalarDotProduct_Int(MemorySegment A, MemorySegment B_T, MemorySegment C, int m, int p, int i, int j) {
        int sum = 0;
        for (int k = 0; k < m; k++) {
            int a = A.getAtIndex(ValueLayout.JAVA_INT, ((long) i * m + k));
            int b = B_T.getAtIndex(ValueLayout.JAVA_INT, ((long) j * m + k));
            sum += a * b;
        }
        C.setAtIndex(ValueLayout.JAVA_INT, ((long) i * p + j), sum);
    }

    private static MemorySegment fastTranspose2D_Int(MemorySegment src, Arena arena, int rows, int cols) {
        MemorySegment dst = arena.allocate((long) rows * cols * 4L);
        java.util.stream.IntStream.range(0, rows).parallel().forEach(i -> {
            for (int j = 0; j < cols; j++) {
                int val = src.getAtIndex(ValueLayout.JAVA_INT, (long) i * cols + j);
                dst.setAtIndex(ValueLayout.JAVA_INT, (long) j * rows + i, val);
            }
        });
        return dst;
    }


}