package testing;

import java.lang.foreign.Arena;
import jnum.DType;
import jnum.NDArray;

public class Main {
    public static void main(String[] args) {
        // 10 Million elements for stress testing
        int SIZE = 100_000_000;
        int DIM = 3162; // ~10M elements for 2D arrays

        // Using ofShared() as requested for custom memory control
        try (Arena arena = Arena.ofShared()) {
            
            // 1D Arrays
            NDArray a = NDArray.ones(arena, DType.FLOAT, SIZE);
            NDArray b = NDArray.ones(arena, DType.FLOAT, SIZE);
            
            // 2D Arrays for Transpose
            NDArray matA = NDArray.ones(arena, DType.FLOAT, DIM, DIM);
            NDArray matB = NDArray.ones(arena, DType.FLOAT, DIM, DIM);

            System.out.println("Warming up JVM JIT Compiler...");
            for (int i = 0; i < 5; i++) {
                a.dot(b);
                a.add(b);
                a.sub(b);
                a.sin();
                a.cos();
                a.addi(b);
            }

            System.out.println("\n--- JNum v0.0.4 Benchmark ---");
            System.out.println("Elements: " + SIZE + "\n");

            // 1. Dot Product
            long t0 = System.nanoTime();
            a.dot(b);
            long t1 = System.nanoTime();
            System.out.printf("1. Dot Product (1D)          : %.2f ms\n", (t1 - t0) / 1e6);

            // 2. Transpose + Element-wise Dot
            // Note: dot() strictly requires 1D. We use mul().sum() on transposed 2D views 
            // to benchmark the Contiguous Safety Shield / Strided Fallback performance.
            long t2 = System.nanoTime();
            NDArray tA = matA.transpose();
            NDArray tB = matB.transpose();
            tA.mul(tB).sum(); 
            long t3 = System.nanoTime();
            System.out.printf("2. Transpose + Math (2D)     : %.2f ms\n", (t3 - t2) / 1e6);

            // 3. Add (2-Lane Unrolled) vs Sub (1-Lane)
            long t4 = System.nanoTime();
            a.add(b);
            long t5 = System.nanoTime();
            a.sub(b);
            long t6 = System.nanoTime();
            System.out.printf("3a. Add (2-Lane Unrolled)    : %.2f ms\n", (t5 - t4) / 1e6);
            System.out.printf("3b. Sub (1-Lane Standard)    : %.2f ms\n", (t6 - t5) / 1e6);

            // 4. Sin (4-Lane Unrolled) vs Cos (1-Lane)
            long t7 = System.nanoTime();
            a.sin();
            long t8 = System.nanoTime();
            a.cos();
            long t9 = System.nanoTime();
            System.out.printf("4a. Sin (4-Lane Unrolled)    : %.2f ms\n", (t8 - t7) / 1e6);
            System.out.printf("4b. Cos (1-Lane Standard)    : %.2f ms\n", (t9 - t8) / 1e6);

            // 5. In-Place vs Out-of-Place
            long t10 = System.nanoTime();
            a.addi(b); // Does not allocate new MemorySegment
            long t11 = System.nanoTime();
            a.add(b);  // Allocates new MemorySegment via NDArray.zeros()
            long t12 = System.nanoTime();
            System.out.printf("5a. In-Place (addi)          : %.2f ms\n", (t11 - t10) / 1e6);
            System.out.printf("5b. Out-of-Place (add)       : %.2f ms\n", (t12 - t11) / 1e6);
        }
    }
}