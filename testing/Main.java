package testing;

import jnum.NDArray;

public class Main {

    public static void main(String[] args) {

        int size = 100_000_000;

        System.out.println("Allocating arrays...");
        NDArray a = NDArray.zeros(size);
        NDArray b = NDArray.ones(size);

        System.out.println("Warming up JVM (20 runs)...");
        for (int i = 0; i < 20; i++) {
            NDArray tmp = a.add(b);
            tmp.getFlat(0); 
        }

        System.out.println("Starting Benchmark...");

        int runs = 10;
        long total = 0;

        for (int i = 0; i < runs; i++) {

            long start = System.nanoTime();

            NDArray c = a.add(b);

            float guard = c.getFlat(0); 

            long end = System.nanoTime();

            long duration = end - start;
            total += duration;

            System.out.println(
                "Run " + (i + 1) + ": " + duration / 1_000_000 + " ms"
            );
        }

        System.out.println("---------------------------------");
        System.out.println("Average: " + (total / runs) / 1_000_000 + " ms");
    }
}