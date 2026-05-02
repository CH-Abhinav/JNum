package jnum.jnumops;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import jnum.NDArray;
import jnum.jnumops.LinalgOps.AVX2Float;

public class LinalgOps {

    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    /*private static final int VL = SPECIES.length();
    private static final long FLOAT_BYTES = ValueLayout.JAVA_FLOAT.byteSize();
    private static final ByteOrder ORDER = ByteOrder.nativeOrder();*/
    private static final ForkJoinPool POOL = ForkJoinPool.commonPool();
    private static final int THRESHOLD = 64;
    

    // Non-instantiable utility class
    private LinalgOps() {
        throw new AssertionError();
    }

    public static NDArray matmulFloat(NDArray a, NDArray b, NDArray resArray) {
        String arch=System.getProperty("os.arch");
        int vLen=SPECIES.length();
        int n = a.shape[0];
        int m = a.shape[1];
        int p = b.shape[1];

        try(Arena arena=Arena.ofShared()){
            MemorySegment memA = a.isContiguous() ? a.data : a.contiguous(arena).data;
            MemorySegment memB = fastTranspose2D(b.data, arena, m, p);
            MemorySegment memC = resArray.data;
            // TODO: AVX-512 kernel (vLen>=16) and NEON kernel (aarch64)
            if(arch.contains("aarch64")){
                POOL.invoke(new AVX2Float(memA, memB, memC, n, m, p, 0, n));
            }
            else{
                if(vLen>=16){
                    POOL.invoke(new AVX2Float(memA, memB, memC, n, m, p, 0, n));
                }
                else{
                    POOL.invoke(new AVX2Float(memA, memB, memC, n, m, p, 0, n));
                }
            }
        }
        return resArray;
    }

    static class AVX2Float extends RecursiveAction{
        MemorySegment A,B_T,C;
        int n,m,p,startRow,endRow;
        AVX2Float(MemorySegment A,MemorySegment B_T,MemorySegment C,int n,int m,int p,int startRow,int endRow){
            this.A = A; this.B_T = B_T; this.C = C;
            this.n = n; this.m = m; this.p = p;
            this.startRow = startRow; this.endRow = endRow;
        }
        @Override
        protected void compute(){
            if(endRow-startRow<=THRESHOLD){
                int safeRowEnd=endRow-((endRow-startRow)%2);
                int safeColEnd=p-(p%2);
                for(int i=startRow;i<safeRowEnd;i+=2){
                    for(int j=0;j<safeColEnd;j+=2){
                        hybridKernel2x2_Float(A, B_T, C, m, p, i, j);
                    }
                }
                if (safeRowEnd < endRow) {
                    for (int j = 0; j < safeColEnd; j++) {
                        scalarDotProduct_Float(A, B_T, C, m, p, safeRowEnd, j);
                    }
                }
                if (safeColEnd < p) {
                    for (int i = startRow; i < safeRowEnd; i++) {
                        scalarDotProduct_Float(A, B_T, C, m, p, i, safeColEnd);
                    }
                }
                if (safeRowEnd < endRow && safeColEnd < p) {
                    scalarDotProduct_Float(A, B_T, C, m, p, safeRowEnd, safeColEnd);
                }
            }
            else{
                int mid = startRow + (endRow - startRow) / 2;
                invokeAll(
                    new AVX2Float(A, B_T, C, n, m, p, startRow, mid),
                    new AVX2Float(A, B_T, C, n, m, p, mid, endRow)
                );
            }
        }
    }

    private static void hybridKernel2x2_Float(MemorySegment A, MemorySegment B_T, MemorySegment C, int m, int p, int i, int j){
        var vSum00 = FloatVector.zero(SPECIES); var vSum01 = FloatVector.zero(SPECIES);
        var vSum10 = FloatVector.zero(SPECIES); var vSum11 = FloatVector.zero(SPECIES);
        long k = 0;
        long loopBound = SPECIES.loopBound(m);
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
            float a0 = A.getAtIndex(ValueLayout.JAVA_FLOAT, ((long) (i) * m + k));
            float a1 = A.getAtIndex(ValueLayout.JAVA_FLOAT, ((long) (i + 1) * m + k));
            float b0 = B_T.getAtIndex(ValueLayout.JAVA_FLOAT, ((long) (j) * m + k));
            float b1 = B_T.getAtIndex(ValueLayout.JAVA_FLOAT, ((long) (j + 1) * m + k));
            sum00 += a0 * b0; sum01 += a0 * b1; sum10 += a1 * b0; sum11 += a1 * b1;
        }

        C.setAtIndex(ValueLayout.JAVA_FLOAT, ((long) (i) * p + j), sum00);
        C.setAtIndex(ValueLayout.JAVA_FLOAT, ((long) (i) * p + j + 1), sum01);
        C.setAtIndex(ValueLayout.JAVA_FLOAT, ((long) (i + 1) * p + j), sum10);
        C.setAtIndex(ValueLayout.JAVA_FLOAT, ((long) (i + 1) * p + j + 1), sum11);
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

    private static MemorySegment fastTranspose2D(MemorySegment src, Arena arena, int rows, int cols) {
        MemorySegment dst = arena.allocate((long) rows * cols * 4L);
        java.util.stream.IntStream.range(0, rows).parallel().forEach(i -> {
            for (int j = 0; j < cols; j++) {
                float val = src.getAtIndex(ValueLayout.JAVA_FLOAT, (long) i * cols + j);
                dst.setAtIndex(ValueLayout.JAVA_FLOAT, (long) j * rows + i, val);
            }
        });
        
        return dst;
    }

}