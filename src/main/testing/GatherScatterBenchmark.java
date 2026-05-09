package testing;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Random;
import jnum.NDArray;
import jnum.jnumops.ArithmaticOps;
import jnum.DType; 


public class GatherScatterBenchmark {

    public static void main(String[] args) {
        // N = 4096 means a 4096 x 4096 matrix (16.7 Million floats)
        int N = 1024;
        int[] shape = {N, N};
        
        // Standard Row-Major Strides
        int[] contigStrides = {N, 1};
        // Transposed Column-Major Strides (Forces Gather/Scatter)
        int[] transposedStrides = {1, N};

        System.out.println("Initializing N=" + N + " Gather/Scatter Benchmark...");

        try (Arena arena = Arena.ofShared()) {
            long bytes = (long) N * N * 4L;
            MemorySegment memA = arena.allocate(bytes, 64);
            MemorySegment memB = arena.allocate(bytes, 64);
            MemorySegment memC = arena.allocate(bytes, 64);

            // Fill with random noise
            Random r = new Random();
            for (long i = 0; i < N * N; i++) {
                memA.setAtIndex(ValueLayout.JAVA_FLOAT, i, r.nextFloat());
                memB.setAtIndex(ValueLayout.JAVA_FLOAT, i, r.nextFloat());
            }

            // 1. Create the NDArrays
            // Note: Replace DType.FLOAT32 with whatever your type enum actually is
           NDArray A = NDArray.rand(DType.FLOAT, N, N);
        NDArray B_Contig = NDArray.rand(DType.FLOAT, N, N);
        NDArray C = NDArray.zeros(DType.FLOAT, N, N);

        // 2. USE YOUR TRANSPOSE METHOD (This automatically swaps the strides!)
        NDArray B_Strided = B_Contig.transpose();

            System.out.println("Warming up JVM...");
            for (int i = 0; i < 3; i++) {
                ArithmaticOps.addFloat(A, B_Contig, C);
                ArithmaticOps.addFloat(A, B_Strided, C);
            }

            // ==========================================================
            // TEST 1: CONTIGUOUS SIMD (The Baseline)
            // ==========================================================
            System.out.println("\n====== 1. CONTIGUOUS SIMD (A + B) ======");
            for (int runs = 1; runs <= 10; runs++) {
                long start = System.nanoTime();
                ArithmaticOps.addFloat(A, B_Contig, C);
                long elapsed = System.nanoTime() - start;
                System.out.printf("Run %d: %.2f ms\n", runs, elapsed / 1e6);
            }

            // ==========================================================
            // TEST 2: GATHER/SCATTER STRIDED SIMD (The V0.0.7 Upgrade)
            // ==========================================================
            System.out.println("\n====== 2. GATHER/SCATTER SIMD (A + B_Transposed) ======");
            for (int runs = 1; runs <= 10; runs++) {
                long start = System.nanoTime();
                ArithmaticOps.addFloat(A, B_Strided, C);
                long elapsed = System.nanoTime() - start;
                System.out.printf("Run %d: %.2f ms\n", runs, elapsed / 1e6);
            }
        }
    }
}
