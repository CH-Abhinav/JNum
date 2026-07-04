package jnum;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

public class NDArrayBinaryArithmeticTest {

    @Test
    void binaryArithmeticOpsWorkForContiguousAndNonContiguousInputs() {
        assertArithmeticResults(
            createContiguousOperands(),
            new float[][]{{10f, 20f, 30f}, {40f, 50f, 60f}},
            new float[][]{{2f, 4f, 6f}, {8f, 10f, 12f}},
            new float[][]{{12f, 24f, 36f}, {48f, 60f, 72f}},
            new float[][]{{8f, 16f, 24f}, {32f, 40f, 48f}},
            new float[][]{{20f, 80f, 180f}, {320f, 500f, 720f}},
            new float[][]{{5f, 5f, 5f}, {5f, 5f, 5f}}
        );

        assertArithmeticResults(
            createNonContiguousOperands(),
            new float[][]{{10f, 40f}, {20f, 50f}, {30f, 60f}},
            new float[][]{{2f, 8f}, {4f, 10f}, {6f, 12f}},
            new float[][]{{12f, 48f}, {24f, 60f}, {36f, 72f}},
            new float[][]{{8f, 32f}, {16f, 40f}, {24f, 48f}},
            new float[][]{{20f, 320f}, {80f, 500f}, {180f, 720f}},
            new float[][]{{5f, 5f}, {5f, 5f}, {5f, 5f}}
        );
    }

    private static void assertArithmeticResults(
        OperandPair operands,
        float[][] expectedLeft,
        float[][] expectedRight,
        float[][] expectedAdd,
        float[][] expectedSub,
        float[][] expectedMul,
        float[][] expectedDiv
    ) {
        assertMatrixEquals(expectedLeft, operands.left());
        assertMatrixEquals(expectedRight, operands.right());
        assertMatrixEquals(expectedAdd, operands.left().add(operands.right()));
        assertMatrixEquals(expectedSub, operands.left().sub(operands.right()));
        assertMatrixEquals(expectedMul, operands.left().mul(operands.right()));
        assertMatrixEquals(expectedDiv, operands.left().div(operands.right()));
    }

    private static OperandPair createContiguousOperands() {
        NDArray left = NDArray.from(new float[]{
            10f, 20f, 30f,
            40f, 50f, 60f
        }, 2, 3);
        NDArray right = NDArray.from(new float[]{
            2f, 4f, 6f,
            8f, 10f, 12f
        }, 2, 3);
        return new OperandPair(left, right);
    }

    private static OperandPair createNonContiguousOperands() {
        NDArray left = NDArray.from(new float[]{
            10f, 20f, 30f,
            40f, 50f, 60f
        }, 2, 3).transpose();
        NDArray right = NDArray.from(new float[]{
            2f, 4f, 6f,
            8f, 10f, 12f
        }, 2, 3).transpose();
        return new OperandPair(left, right);
    }

    private static void assertMatrixEquals(float[][] expected, NDArray actual) {
        assertArrayEquals(new int[]{expected.length, expected[0].length}, actual.getShape());
        for (int row = 0; row < expected.length; row++) {
            for (int col = 0; col < expected[row].length; col++) {
                assertEquals(expected[row][col], actual.getFloat(row, col), 1e-5f);
            }
        }
    }

    private record OperandPair(NDArray left, NDArray right) {
    }
}
