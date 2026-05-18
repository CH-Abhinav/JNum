import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

public class V1PreviewBenchmark {

    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final int CORES = Runtime.getRuntime().availableProcessors();

    // ==============================================================================
    // THE 2x2 SIMD MICRO-KERNEL (The undisputed champion)
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

    // ==============================================================================
    // BASELINE: v0.0.6 ForkJoin (Work-Stealing Overhead)
    // ==============================================================================
    static class FJ_Naive extends RecursiveAction {
        MemorySegment A, B_T, C; int n, startRow, endRow;
        FJ_Naive(MemorySegment A, MemorySegment B_T, MemorySegment C, int n, int startRow, int endRow) {
            this.A = A; this.B_T = B_T; this.C = C; this.n = n; this.startRow = startRow; this.endRow = endRow;
        }
        @Override
        protected void compute() {
            if (endRow - startRow <= 32) {
                for (int i = startRow; i < endRow; i += 2) {
                    for (int j = 0; j < n; j += 2) hybridKernel2x2(A, B_T, C, n, i, j);
                }
            } else {
                int mid = startRow + (endRow - startRow) / 2; mid -= mid % 2;
                invokeAll(new FJ_Naive(A, B_T, C, n, startRow, mid), new FJ_Naive(A, B_T, C, n, mid, endRow));
            }
        }
    }

    // ==============================================================================
    // EXPERIMENT: v0.1.0 Static Thread Pool (Pinned Memory Regions)
    // ==============================================================================
    public static void runStaticThreadPool(MemorySegment A, MemorySegment B_T, MemorySegment C, int n, ExecutorService executor) throws Exception {
        int rowsPerThread = n / CORES;
        // Ensure even rows for the 2x2 kernel
        rowsPerThread -= rowsPerThread % 2; 
        
        List<Callable<Void>> tasks = new ArrayList<>();
        
        for (int t = 0; t < CORES; t++) {
            final int startRow = t * rowsPerThread;
            final int endRow = (t == CORES - 1) ? n : startRow + rowsPerThread;
            
            tasks.add(() -> {
                for (int i = startRow; i < endRow; i += 2) {
                    for (int j = 0; j < n; j += 2) {
                        hybridKernel2x2(A, B_T, C, n, i, j);
                    }
                }
                return null;
            });
        }
        
        // Block until all threads finish their specific pinned tasks
        executor.invokeAll(tasks);
    }

    // ==============================================================================
    // BENCHMARK RUNNER
    // ==============================================================================
    public static void main(String[] args) throws Exception {
        int n = 10240; // Test 4096 first, then try 8192
        System.out.println("Cores Detected: " + CORES);
        
        try (Arena arena = Arena.ofShared()) {
            long bytes = (long) n * n * 4L;
            
            // EXPERIMENT 2: 64-Byte Cache Aligned Memory Allocation
            MemorySegment memA = arena.allocate(bytes, 64); 
            MemorySegment memB_T = arena.allocate(bytes, 64); 
            MemorySegment memC = arena.allocate(bytes, 64);

            Random r = new Random();
            for (long i = 0; i < n * n; i++) {
                memA.setAtIndex(ValueLayout.JAVA_FLOAT, i, r.nextFloat());
                memB_T.setAtIndex(ValueLayout.JAVA_FLOAT, i, r.nextFloat());
            }

            ForkJoinPool fjPool = ForkJoinPool.commonPool();
            ExecutorService staticPool = Executors.newFixedThreadPool(CORES);

            System.out.println("Warming up kernels...");
            fjPool.invoke(new FJ_Naive(memA, memB_T, memC, n, 0, n));
            runStaticThreadPool(memA, memB_T, memC, n, staticPool);

            System.out.println("\n====== v0.0.6: ForkJoin + Aligned Memory ======");
            for (int runs = 1; runs <= 3; runs++) {
                long start = System.nanoTime();
                fjPool.invoke(new FJ_Naive(memA, memB_T, memC, n, 0, n));
                long end = System.nanoTime();
                System.out.printf("Run %d: %.2f ms\n", runs, (end - start) / 1e6);
            }

            System.out.println("\n====== v0.1.0 PREVIEW: Static Pool + Aligned Memory ======");
            for (int runs = 1; runs <= 3; runs++) {
                long start = System.nanoTime();
                runStaticThreadPool(memA, memB_T, memC, n, staticPool);
                long end = System.nanoTime();
                System.out.printf("Run %d: %.2f ms\n", runs, (end - start) / 1e6);
            }
            
            staticPool.shutdown();
        }
    }
}