package testing;

import jnum.NDArray;
import jnum.DType;
import jnum.jnumops.LinalgOps;

public class benchmark2 {

    static final int SIZE = 1024;
    static final int WARMUP_RUNS = 3;
    static final int MEASURE_RUNS = 5;

    public static void main(String[] args) {
        System.out.println("--- JNum FMA Matmul Benchmark (" + SIZE + "x" + SIZE + ") ---");

        NDArray A = NDArray.zeros(DType.FLOAT, new int[]{SIZE, SIZE});
        NDArray B = NDArray.zeros(DType.FLOAT, new int[]{SIZE, SIZE});
        NDArray C = NDArray.zeros(DType.FLOAT, new int[]{SIZE, SIZE});

        // Optional: Populate A and B here if you have a random generator for NDArray

        System.out.print("Warming up... ");
        for (int r = 0; r < WARMUP_RUNS; r++) LinalgOps.matmulFloat(A, B, C);

        long totalTime = 0;
        for (int r = 0; r < MEASURE_RUNS; r++) {
            long start = System.nanoTime();
            LinalgOps.matmulFloat(A, B, C);
            totalTime += (System.nanoTime() - start);
        }
        
        System.out.println("\nJNum FMA Avg Time: " + (totalTime / MEASURE_RUNS) / 1_000_000.0 + " ms.");
    }
}