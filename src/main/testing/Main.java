package testing;

import jnum.NDArray;

public class Main {
    public static void main(String[] args) {
        int size = 100_000_000;

        System.out.println("Allocating arrays...");
        NDArray a = NDArray.zeros(size);
        NDArray b = NDArray.zeros(size); // (Using zeros so we don't hit the slow 'ones' loop)
        NDArray c = NDArray.zeros(size); 

        System.out.println("Warming up JVM (20 runs)...");
        for (int i = 0; i < 20; i++) {
            a.add(b, c); 
            c.getFlat(0); 
        }

        System.out.println("Starting Benchmark...");
        int runs = 10;
        long total = 0;

        for (int i = 0; i < runs; i++) {
            long start = System.nanoTime();

            a.add(b, c); // Zero allocations happening here!

            float guard = c.getFlat(0); 
            long end = System.nanoTime();
            long duration = (end - start) / 1_000_000;
            total += duration;

            System.out.println("Run " + (i + 1) + ": " + duration + " ms");
        }
        System.out.println("---------------------------------");
        System.out.println("Average: " + (total / runs) + " ms");
    }
}