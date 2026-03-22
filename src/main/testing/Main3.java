package testing;

import jnum.DType;
import jnum.NDArray;

public class Main3 {
    public static void main(String[] args) {
        int size = 100_000_000;
        int runs = 50;
        
        System.out.println("=== JNum v0.0.3 Multi-Type Hardware Benchmark ===");
        System.out.println("Array Size: 150,000,000 elements\n");

        // --- ALLOCATION ---
        System.out.print("Allocating 32-bit Float Arrays (600 MB)... ");
        NDArray floatA = NDArray.ones(DType.FLOAT, size);
        NDArray floatB = NDArray.ones(DType.FLOAT, size);
        System.out.println("Done.");

        System.out.print("Allocating 32-bit Int Arrays (600 MB)... ");
        NDArray intA = NDArray.ones(DType.INTEGER, size);
        NDArray intB = NDArray.ones(DType.INTEGER, size);
        System.out.println("Done.");

        System.out.print("Allocating 64-bit Double Arrays (1.2 GB)... ");
        NDArray doubleA = NDArray.ones(DType.DOUBLE, size);
        NDArray doubleB = NDArray.ones(DType.DOUBLE, size);
        System.out.println("Done.\n");

        // --- WARMUP ---
        System.out.print("Warming up JVM C2 Compiler (JIT)... ");
        for (int i = 0; i < 5; i++) {
            floatA.add(floatB, floatA);
            intA.add(intB, intA);
            doubleA.add(doubleB, doubleA);
        }
        System.out.println("Done.\n");

        long start, end, duration, total;

        // ---------------------------------------------------------
        // TEST 1: FLOAT32 (AVX 8-Lane SIMD)
        // ---------------------------------------------------------
        System.out.println("--- Test 1: FLOAT32 Math (add) ---");
        total = 0;
        for (int i = 0; i < runs; i++) {
            start = System.nanoTime();
            floatA.add(floatB, floatA);
            end = System.nanoTime();
            duration = (end - start) / 1_000_000;
            total += duration;
        }
        System.out.println("Float32 Average: " + (total / runs) + " ms\n");

        // ---------------------------------------------------------
        // TEST 2: INT32 (AVX 8-Lane SIMD)
        // ---------------------------------------------------------
        System.out.println("--- Test 2: INT32 Math (add) ---");
        total = 0;
        for (int i = 0; i < runs; i++) {
            start = System.nanoTime();
            intA.add(intB, intA);
            end = System.nanoTime();
            duration = (end - start) / 1_000_000;
            total += duration;
        }
        System.out.println("Int32 Average: " + (total / runs) + " ms\n");

        // ---------------------------------------------------------
        // TEST 3: FLOAT64 DOUBLE (AVX 4-Lane SIMD)
        // ---------------------------------------------------------
        System.out.println("--- Test 3: FLOAT64 DOUBLE Math (add) ---");
        total = 0;
        for (int i = 0; i < runs; i++) {
            start = System.nanoTime();
            doubleA.add(doubleB, doubleA);
            end = System.nanoTime();
            duration = (end - start) / 1_000_000;
            total += duration;
        }
        System.out.println("Float64 Average: " + (total / runs) + " ms\n");
        
        System.out.println("Benchmark complete.");
    }
}