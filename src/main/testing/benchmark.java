
package testing;

import jnum.NDArray;

public class Benchmark {
    public static void main(String[] args) {
        // 100 Million Elements (Compute-Bound Test)
        int size = 100_000_000; 
        System.out.println("Initializing " + size + " element arrays...");

        // 1. Setup the Raw Data
        float[] rawData = new float[size];
        for (int i = 0; i < size; i++) {
            rawData[i] = (float) (Math.random() * 10f); // Random floats
        }

        // Load it into your JNum Architecture
        NDArray jnumArray = NDArray.from(rawData, size);

        // 2. The Warmup Phase (Crucial)
        // This forces the C2 JIT Compiler to optimize the code into bare-metal instructions.
        System.out.println("Warming up C2 JIT Compiler...");
        for (int i = 0; i < 3; i++) {
            jnumArray.sin();
            standardJavaSin(rawData);
        }

        // 3. The Silicon Race
        System.out.println("--- Starting Hardware Benchmark ---");

        // JNum Vector API Benchmark
        long startJNum = System.nanoTime();
        NDArray jnumResult = jnumArray.sin();
        long endJNum = System.nanoTime();
        double jnumMs = (endJNum - startJNum) / 1_000_000.0;
        System.out.println("JNum Vector API (sin) : " + jnumMs + " ms");

        // Standard Java Benchmark
        long startJava = System.nanoTime();
        float[] javaResult = standardJavaSin(rawData);
        long endJava = System.nanoTime();
        double javaMs = (endJava - startJava) / 1_000_000.0;
        System.out.println("Standard Java (Math.sin): " + javaMs + " ms");

        // The Verdict
        double speedup = javaMs / jnumMs;
        System.out.printf("Hardware Acceleration   : %.2fx Faster\n", speedup);
    }

    // Standard single-threaded Java baseline
    private static float[] standardJavaSin(float[] data) {
        float[] res = new float[data.length];
        for (int i = 0; i < data.length; i++) {
            res[i] = (float) Math.sin(data[i]);
        }
        return res;
    }
}