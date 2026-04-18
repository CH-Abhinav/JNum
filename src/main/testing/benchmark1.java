package testing;

public class benchmark1{

    static final int SIZE = 1024;
    static final int WARMUP_RUNS = 3;
    static final int MEASURE_RUNS = 5;

    public static void main(String[] args) {
        System.out.println("--- Naive Java Matmul Benchmark (" + SIZE + "x" + SIZE + ") ---");

        float[][] A = new float[SIZE][SIZE];
        float[][] B = new float[SIZE][SIZE];
        float[][] C = new float[SIZE][SIZE];

        for (int i = 0; i < SIZE; i++) {
            for (int j = 0; j < SIZE; j++) {
                A[i][j] = (float) Math.random();
                B[i][j] = (float) Math.random();
            }
        }

        System.out.print("Warming up... ");
        for (int r = 0; r < WARMUP_RUNS; r++) execute(A, B, C);

        long totalTime = 0;
        for (int r = 0; r < MEASURE_RUNS; r++) {
            long start = System.nanoTime();
            execute(A, B, C);
            totalTime += (System.nanoTime() - start);
        }
        
        System.out.println("\nNaive Avg Time: " + (totalTime / MEASURE_RUNS) / 1_000_000.0 + " ms. (Sanity: " + C[0][0] + ")");
    }

    private static void execute(float[][] A, float[][] B, float[][] C) {
        for (int i = 0; i < SIZE; i++) {
            for (int j = 0; j < SIZE; j++) {
                float sum = 0;
                for (int k = 0; k < SIZE; k++) {
                    sum += A[i][k] * B[k][j];
                }
                C[i][j] = sum;
            }
        }
    }
}