package testing;

import jnum.NDArray;

public class Main2 {
    public static void main(String[] args) {
        int size = 100_000_000;
        System.out.println("=== JNum v0.0.2 Performance Benchmark ===");
        System.out.println("Array Size: 100,000,000 elements (400 MB each)\n");

        System.out.print("Allocating memory... ");
        NDArray a = NDArray.ones(size);
        NDArray b = NDArray.ones(size);
        NDArray c = NDArray.zeros(size);
        System.out.println("Done.");

        System.out.print("Warming up JVM C2 Compiler (20 iterations)... ");
        for (int i = 0; i < 5; i++) {
            a.add(b, c);
            c.add(5.0f, c);
            c.sum();
        }
        System.out.println("Done.\n");

        int runs = 100;
        long start, end, duration;
        long total;
        float guard = 0; // Prevents the compiler from deleting our math loops

        // ---------------------------------------------------------
        // TEST 1: Array Math
        // ---------------------------------------------------------
        System.out.println("--- Test 1: Array Math (A.add(B, C)) ---");
        total = 0;
        for (int i = 0; i < runs; i++) {
            start = System.nanoTime();
            
            a.add(b, c); // Zero-allocation addition
            guard += c.getFlat(0); 
            
            end = System.nanoTime();
            duration = (end - start) / 1_000_000;
            total += duration;
            System.out.println("Run " + (i + 1) + ": " + duration + " ms");
        }
        System.out.println("Average: " + (total / runs) + " ms\n");

        // ---------------------------------------------------------
        // TEST 2: Scalar Math
        // ---------------------------------------------------------
        System.out.println("--- Test 2: Scalar Math (A.add(5.0f, C)) ---");
        total = 0;
        for (int i = 0; i < runs; i++) {
            start = System.nanoTime();
            
            a.add(5.0f, c); // Hardware broadcasting
            guard += c.getFlat(0);
            
            end = System.nanoTime();
            duration = (end - start) / 1_000_000;
            total += duration;
            System.out.println("Run " + (i + 1) + ": " + duration + " ms");
        }
        System.out.println("Average: " + (total / runs) + " ms\n");

        // ---------------------------------------------------------
        // TEST 3: Reduction
        // ---------------------------------------------------------
        System.out.println("--- Test 3: Hardware Reduction (c.sum()) ---");
        total = 0;
        for (int i = 0; i < runs; i++) {
            start = System.nanoTime();
            
            float sumResult = c.avg(); // Horizontal lane crushing
            guard += sumResult;
            
            end = System.nanoTime();
            duration = (end - start) / 1_000_000;
            total += duration;
            System.out.println("Run " + (i + 1) + ": " + duration + " ms");
        }
        System.out.println("Average: " + (total / runs) + " ms\n");
        
        System.out.println("Benchmark complete. (Guard: " + guard + ")");
    }
}