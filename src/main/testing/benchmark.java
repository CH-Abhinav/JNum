package testing;
import jnum.DType;
import jnum.NDArray;

public class benchmark {
    public static void main(String[] args) {
        long v1=System.nanoTime();
        int size = 100_000_000;
        int warmupIterations = 10;
        int measureIterations = 1000;

        // Setup: Assuming you have a ones() or random() allocator
        NDArray a = NDArray.rand(DType.FLOAT, size); 
        NDArray b = NDArray.rand(DType.FLOAT, size);
        System.out.println("Starting JNum Warmup (Waiting for C2 Compiler)...");
        double blackhole = 0;
        for (int i = 0; i < warmupIterations; i++) {
            blackhole += a.dot(b);
        }

        System.out.println("Starting JNum Measurement...");
        long startTime = System.nanoTime();
        
        for (int i = 0; i < measureIterations; i++) {
            blackhole += a.dot(b);
        }
        
        long endTime = System.nanoTime();
        
        double totalMs = (endTime - startTime) / 1_000_000.0;
        double avgMs = totalMs / measureIterations;
        System.out.println("JNum Average Dot Product Time: " + avgMs + " ms");
        System.out.println("Result (Prevents Dead Code): " + blackhole);
    }
}