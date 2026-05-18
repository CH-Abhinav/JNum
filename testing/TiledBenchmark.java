import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.Random;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

public class PackedTiledBenchmark {

    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final int BLOCK_SIZE = 64; // The perfect L1 Cache Block

    // ==============================================================================
    // THE SECRET WEAPON: Thread-Local Memory Arenas (Zero Garbage Collection)
    // Each thread gets permanent 64KB buffers to pack memory into.
    // ==============================================================================
    private static final ThreadLocal<MemorySegment> threadLocalA = ThreadLocal.withInitial(() -> Arena.ofAuto().allocate((long) BLOCK_SIZE * BLOCK_SIZE * 4L));
    private static final ThreadLocal<MemorySegment> threadLocalB = ThreadLocal.withInitial(() -> Arena.ofAuto().allocate((long) BLOCK_SIZE * BLOCK_SIZE * 4L));
    private static final ThreadLocal<MemorySegment> threadLocalC = ThreadLocal.withInitial(() -> Arena.ofAuto().allocate((long) BLOCK_SIZE * BLOCK_SIZE * 4L));

    // Fast copy utility to slice a 128x128 block out of the giant matrix
    private static void packBlock(MemorySegment src, MemorySegment dest, int n, int rowStart, int colStart) {
        for (int r = 0; r < BLOCK_SIZE; r++) {
            MemorySegment.copy(src, ((long) (rowStart + r) * n + colStart) * 4L, dest, (long) r * BLOCK_SIZE * 4L, BLOCK_SIZE * 4L);
        }
    }

    // Fast copy utility to write the finished 128x128 C block back to RAM
    private static void unpackBlock(MemorySegment src, MemorySegment dest, int n, int rowStart, int colStart) {
        for (int r = 0; r < BLOCK_SIZE; r++) {
            MemorySegment.copy(src, (long) r * BLOCK_SIZE * 4L, dest, ((long) (rowStart + r) * n + colStart) * 4L, BLOCK_SIZE * 4L);
        }
    }

    // ==============================================================================
    // 1. THE PACKED MICRO-KERNEL
    // Operates ONLY on the tiny 128x128 continuous blocks. Zero Cache Misses.
    // ==============================================================================
    public static void packedKernel2x2(MemorySegment pA, MemorySegment pB, MemorySegment pC) {
        int loopBound = SPECIES.loopBound(BLOCK_SIZE);
        
        for (int i = 0; i < BLOCK_SIZE; i += 2) {
            for (int j = 0; j < BLOCK_SIZE; j += 2) {
                var vSum00 = FloatVector.zero(SPECIES); var vSum01 = FloatVector.zero(SPECIES);
                var vSum10 = FloatVector.zero(SPECIES); var vSum11 = FloatVector.zero(SPECIES);

                int k = 0;
                for (; k < loopBound; k += SPECIES.length()) {
                    long offsetA0 = ((long) i * BLOCK_SIZE + k) * 4L;
                    long offsetA1 = ((long) (i + 1) * BLOCK_SIZE + k) * 4L;
                    var vA0 = FloatVector.fromMemorySegment(SPECIES, pA, offsetA0, ByteOrder.nativeOrder());
                    var vA1 = FloatVector.fromMemorySegment(SPECIES, pA, offsetA1, ByteOrder.nativeOrder());

                    long offsetB0 = ((long) j * BLOCK_SIZE + k) * 4L;
                    long offsetB1 = ((long) (j + 1) * BLOCK_SIZE + k) * 4L;
                    var vB0 = FloatVector.fromMemorySegment(SPECIES, pB, offsetB0, ByteOrder.nativeOrder());
                    var vB1 = FloatVector.fromMemorySegment(SPECIES, pB, offsetB1, ByteOrder.nativeOrder());

                    vSum00 = vA0.fma(vB0, vSum00); vSum01 = vA0.fma(vB1, vSum01);
                    vSum10 = vA1.fma(vB0, vSum10); vSum11 = vA1.fma(vB1, vSum11);
                }

                float sum00 = vSum00.reduceLanes(VectorOperators.ADD); float sum01 = vSum01.reduceLanes(VectorOperators.ADD);
                float sum10 = vSum10.reduceLanes(VectorOperators.ADD); float sum11 = vSum11.reduceLanes(VectorOperators.ADD);

                for (; k < BLOCK_SIZE; k++) {
                    float a0 = pA.getAtIndex(ValueLayout.JAVA_FLOAT, ((long) i * BLOCK_SIZE + k));
                    float a1 = pA.getAtIndex(ValueLayout.JAVA_FLOAT, ((long) (i + 1) * BLOCK_SIZE + k));
                    float b0 = pB.getAtIndex(ValueLayout.JAVA_FLOAT, ((long) j * BLOCK_SIZE + k));
                    float b1 = pB.getAtIndex(ValueLayout.JAVA_FLOAT, ((long) (j + 1) * BLOCK_SIZE + k));
                    sum00 += a0 * b0; sum01 += a0 * b1;
                    sum10 += a1 * b0; sum11 += a1 * b1;
                }

                // Accumulate directly into the tiny pC block!
                long idx00 = ((long) i * BLOCK_SIZE + j); long idx01 = ((long) i * BLOCK_SIZE + j + 1);
                long idx10 = ((long) (i + 1) * BLOCK_SIZE + j); long idx11 = ((long) (i + 1) * BLOCK_SIZE + j + 1);
                
                pC.setAtIndex(ValueLayout.JAVA_FLOAT, idx00, sum00 + pC.getAtIndex(ValueLayout.JAVA_FLOAT, idx00));
                pC.setAtIndex(ValueLayout.JAVA_FLOAT, idx01, sum01 + pC.getAtIndex(ValueLayout.JAVA_FLOAT, idx01));
                pC.setAtIndex(ValueLayout.JAVA_FLOAT, idx10, sum10 + pC.getAtIndex(ValueLayout.JAVA_FLOAT, idx10));
                pC.setAtIndex(ValueLayout.JAVA_FLOAT, idx11, sum11 + pC.getAtIndex(ValueLayout.JAVA_FLOAT, idx11));
            }
        }
    }

    // ==============================================================================
    // 2. THE PACKED MACRO-KERNEL (v0.0.7 Experiment)
    // ==============================================================================
    static class FJ_Packed extends RecursiveAction {
        MemorySegment A, B_T, C; int n, startRow, endRow;
        FJ_Packed(MemorySegment A, MemorySegment B_T, MemorySegment C, int n, int startRow, int endRow) {
            this.A = A; this.B_T = B_T; this.C = C; this.n = n; this.startRow = startRow; this.endRow = endRow;
        }
        @Override
        protected void compute() {
            if (endRow - startRow <= BLOCK_SIZE) {
                // Fetch the permanent memory blocks for THIS specific thread
                MemorySegment pA = threadLocalA.get();
                MemorySegment pB = threadLocalB.get();
                MemorySegment pC = threadLocalC.get();

                for (int iBlock = startRow; iBlock < endRow; iBlock += BLOCK_SIZE) {
                    for (int jBlock = 0; jBlock < n; jBlock += BLOCK_SIZE) {
                        pC.fill((byte) 0); // Reset local C accumulator

                        for (int kBlock = 0; kBlock < n; kBlock += BLOCK_SIZE) {
                            packBlock(A, pA, n, iBlock, kBlock);
                            packBlock(B_T, pB, n, jBlock, kBlock);
                            packedKernel2x2(pA, pB, pC);
                        }
                        // Write the perfectly calculated tile back to Main RAM just ONCE.
                        unpackBlock(pC, C, n, iBlock, jBlock); 
                    }
                }
            } else {
                int mid = startRow + (endRow - startRow) / 2;
                mid -= mid % BLOCK_SIZE;
                invokeAll(new FJ_Packed(A, B_T, C, n, startRow, mid), new FJ_Packed(A, B_T, C, n, mid, endRow));
            }
        }
    }

    // ==============================================================================
    // 3. THE NAIVE MACRO-KERNEL (From v0.0.6)
    // ==============================================================================
    public static void hybridKernel2x2(MemorySegment A, MemorySegment B_T, MemorySegment C, int n, int i, int j) {
        var vSum00 = FloatVector.zero(SPECIES); var vSum01 = FloatVector.zero(SPECIES);
        var vSum10 = FloatVector.zero(SPECIES); var vSum11 = FloatVector.zero(SPECIES);
        int k = 0; int loopBound = SPECIES.loopBound(n);
        for (; k < loopBound; k += SPECIES.length()) {
            long offsetA0 = ((long) i * n + k) * 4L; long offsetA1 = ((long) (i + 1) * n + k) * 4L;
            var vA0 = FloatVector.fromMemorySegment(SPECIES, A, offsetA0, ByteOrder.nativeOrder());
            var vA1 = FloatVector.fromMemorySegment(SPECIES, A, offsetA1, ByteOrder.nativeOrder());
            long offsetB0 = ((long) j * n + k) * 4L; long offsetB1 = ((long) (j + 1) * n + k) * 4L;
            var vB0 = FloatVector.fromMemorySegment(SPECIES, B_T, offsetB0, ByteOrder.nativeOrder());
            var vB1 = FloatVector.fromMemorySegment(SPECIES, B_T, offsetB1, ByteOrder.nativeOrder());
            vSum00 = vA0.fma(vB0, vSum00); vSum01 = vA0.fma(vB1, vSum01);
            vSum10 = vA1.fma(vB0, vSum10); vSum11 = vA1.fma(vB1, vSum11);
        }
        float sum00 = vSum00.reduceLanes(VectorOperators.ADD); float sum01 = vSum01.reduceLanes(VectorOperators.ADD);
        float sum10 = vSum10.reduceLanes(VectorOperators.ADD); float sum11 = vSum11.reduceLanes(VectorOperators.ADD);
        for (; k < n; k++) {
            float a0 = A.getAtIndex(ValueLayout.JAVA_FLOAT, ((long) i * n + k)); float a1 = A.getAtIndex(ValueLayout.JAVA_FLOAT, ((long) (i + 1) * n + k));
            float b0 = B_T.getAtIndex(ValueLayout.JAVA_FLOAT, ((long) j * n + k)); float b1 = B_T.getAtIndex(ValueLayout.JAVA_FLOAT, ((long) (j + 1) * n + k));
            sum00 += a0 * b0; sum01 += a0 * b1; sum10 += a1 * b0; sum11 += a1 * b1;
        }
        C.setAtIndex(ValueLayout.JAVA_FLOAT, ((long) i * n + j), sum00); C.setAtIndex(ValueLayout.JAVA_FLOAT, ((long) i * n + j + 1), sum01);
        C.setAtIndex(ValueLayout.JAVA_FLOAT, ((long) (i + 1) * n + j), sum10); C.setAtIndex(ValueLayout.JAVA_FLOAT, ((long) (i + 1) * n + j + 1), sum11);
    }

    static class FJ_Naive extends RecursiveAction {
        MemorySegment A, B_T, C; int n, startRow, endRow;
        FJ_Naive(MemorySegment A, MemorySegment B_T, MemorySegment C, int n, int startRow, int endRow) {
            this.A = A; this.B_T = B_T; this.C = C; this.n = n; this.startRow = startRow; this.endRow = endRow;
        }
        @Override
        protected void compute() {
            if (endRow - startRow <= 32) {
                for (int i = startRow; i < endRow; i += 2) {
                    for (int j = 0; j < n; j += 2) {
                        hybridKernel2x2(A, B_T, C, n, i, j);
                    }
                }
            } else {
                int mid = startRow + (endRow - startRow) / 2; mid -= mid % 2;
                invokeAll(new FJ_Naive(A, B_T, C, n, startRow, mid), new FJ_Naive(A, B_T, C, n, mid, endRow));
            }
        }
    }

    // ==============================================================================
    // BENCHMARK RUNNER
    // ==============================================================================
    public static void main(String[] args) {
        int n = 4096; // Must be a multiple of BLOCK_SIZE (128) for this test
        
        try (Arena arena = Arena.ofShared(); ForkJoinPool pool = ForkJoinPool.commonPool()) {
            long bytes = (long) n * n * 4L;
            MemorySegment memA = arena.allocate(bytes); MemorySegment memB_T = arena.allocate(bytes); 
            MemorySegment memC = arena.allocate(bytes);

            Random r = new Random();
            for (long i = 0; i < n * n; i++) {
                memA.setAtIndex(ValueLayout.JAVA_FLOAT, i, r.nextFloat());
                memB_T.setAtIndex(ValueLayout.JAVA_FLOAT, i, r.nextFloat());
            }

            System.out.println("Warming up kernels...");
            for (int i = 0; i < 1; i++) {
                pool.invoke(new FJ_Naive(memA, memB_T, memC, n, 0, n));
                pool.invoke(new FJ_Packed(memA, memB_T, memC, n, 0, n));
            }

            System.out.println("\n====== v0.0.6 NAIVE MACRO-KERNEL ======");
            for (int runs = 1; runs <= 1; runs++) {
                long start = System.nanoTime();
                pool.invoke(new FJ_Naive(memA, memB_T, memC, n, 0, n));
                long end = System.nanoTime();
                System.out.printf("Run %d: %.2f ms\n", runs, (end - start) / 1e6);
            }

            System.out.println("\n====== v0.0.7 MEMORY-PACKED TILED KERNEL ======");
            for (int runs = 1; runs <= 1; runs++) {
                long start = System.nanoTime();
                pool.invoke(new FJ_Packed(memA, memB_T, memC, n, 0, n));
                long end = System.nanoTime();
                System.out.printf("Run %d: %.2f ms\n", runs, (end - start) / 1e6);
            }
        }
    }
}
