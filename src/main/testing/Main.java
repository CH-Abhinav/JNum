package testing;

import jnum.NDArray;

public class Main {
    public static void main(String[] args) {
        System.out.println("--- JNum v0.0.2 Final Integration Test ---\n");

        // 1. Ingest standard Java array (Heap to Off-Heap)
        float[] rawData = {10f, 20f, 30f, 40f, 50f, 60f};
        NDArray flatArray = NDArray.from(rawData, 6);
        System.out.println("1. Flat Array: " + flatArray);

        // 2. Zero-Copy Reshape
        NDArray matrix = flatArray.reshape(2, 3);
        System.out.println("2. Reshaped to 2x3: " + matrix);

        // 3. N-Dimensional Indexing (Get Row 1, Col 2 -> should be 60.0)
        System.out.println("3. Value at (1, 2): " + matrix.get(1, 2));

        // 4. Scalar Broadcasting (Add 5 to everything)
        NDArray plusFive = matrix.add(5.0f);
        System.out.println("4. Matrix + 5.0f: " + plusFive);

        // 5. High-Performance Math (plusFive - matrix)
        NDArray difference = plusFive.sub(matrix);
        System.out.println("5. Difference (Expected all 5s): " + difference);

        // 6. Hardware Reductions
        System.out.println("6. Max Value of Original: " + matrix.max());
        System.out.println("7. Min Value of Original: " + matrix.min());
        System.out.println("8. Sum of Original: " + matrix.sum());
        System.out.println("9. Average of Original: " + matrix.avg());
        
        System.out.println("\nAll systems nominal. Welcome to v0.0.2!");
    }
}